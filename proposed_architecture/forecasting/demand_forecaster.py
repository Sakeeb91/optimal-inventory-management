import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemandForecaster:
    """
    Enhanced demand forecasting system for inventory optimization
    
    Features:
    - Multiple forecasting methods (statistical, ML-based)
    - Seasonality detection and modeling
    - Demand distribution fitting
    - Forecast evaluation and selection
    """
    
    def __init__(self):
        """Initialize the demand forecaster"""
        self.data = None
        self.fitted_models = {}
        self.best_model = None
        self.forecasts = None
        self.forecast_distribution = None
    
    def load_data(self, data, date_column='date', demand_column='demand'):
        """
        Load historical demand data
        
        Parameters:
        -----------
        data : pandas.DataFrame or str
            DataFrame containing historical demand data or path to CSV file
        date_column : str
            Name of the column containing dates
        demand_column : str
            Name of the column containing demand values
        """
        if isinstance(data, str):
            # Load data from CSV
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
        
        # Ensure date column is datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Set date as index
        self.data = self.data.set_index(date_column)
        
        # Rename demand column if needed
        if demand_column != 'demand':
            self.data = self.data.rename(columns={demand_column: 'demand'})
        
        # Sort by date
        self.data = self.data.sort_index()
        
        logger.info(f"Loaded data with {len(self.data)} observations")
        
        # Basic statistics
        self.data_stats = {
            'mean': self.data['demand'].mean(),
            'std': self.data['demand'].std(),
            'min': self.data['demand'].min(),
            'max': self.data['demand'].max(),
            'periods': len(self.data)
        }
        
        return self.data
    
    def analyze_time_series(self, plot=False):
        """
        Analyze the time series data to detect patterns
        
        Parameters:
        -----------
        plot : bool
            Whether to generate plots
            
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Check for stationarity
        adf_result = adfuller(self.data['demand'].values)
        is_stationary = adf_result[1] < 0.05
        
        # Try to detect seasonality
        try:
            # Use seasonal decomposition to detect seasonality
            # Try different seasonal periods (weekly, monthly, quarterly)
            seasonal_periods = [7, 30, 90]
            best_seasonal_period = None
            max_seasonal_strength = 0
            
            for period in seasonal_periods:
                if len(self.data) >= 2 * period:
                    decomposition = seasonal_decompose(
                        self.data['demand'], 
                        model='additive', 
                        period=period,
                        extrapolate_trend='freq'
                    )
                    
                    # Calculate seasonal strength
                    seasonal_strength = 1 - (np.var(decomposition.resid) / 
                                            np.var(decomposition.seasonal + decomposition.resid))
                    
                    if seasonal_strength > max_seasonal_strength:
                        max_seasonal_strength = seasonal_strength
                        best_seasonal_period = period
            
            has_seasonality = max_seasonal_strength > 0.3
            
        except Exception as e:
            logger.warning(f"Error in seasonality detection: {str(e)}")
            has_seasonality = False
            best_seasonal_period = None
            max_seasonal_strength = 0
        
        # Detect trend
        if len(self.data) >= 2:
            x = np.arange(len(self.data))
            y = self.data['demand'].values
            slope, _, r_value, p_value, _ = stats.linregress(x, y)
            has_trend = p_value < 0.05
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            trend_strength = r_value ** 2  # R-squared
        else:
            has_trend = False
            trend_direction = None
            trend_strength = 0
        
        # Assemble results
        results = {
            'is_stationary': is_stationary,
            'has_seasonality': has_seasonality,
            'seasonal_period': best_seasonal_period,
            'seasonal_strength': max_seasonal_strength,
            'has_trend': has_trend,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }
        
        # Generate plots if requested
        if plot:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Original time series
            self.data['demand'].plot(ax=axes[0], title='Demand Time Series')
            axes[0].set_ylabel('Demand')
            
            # ACF/PACF plots would be added here
            
            # If seasonality detected, show decomposition
            if has_seasonality and best_seasonal_period:
                decomposition = seasonal_decompose(
                    self.data['demand'],
                    model='additive',
                    period=best_seasonal_period,
                    extrapolate_trend='freq'
                )
                
                decomposition.trend.plot(ax=axes[1], title='Trend Component')
                axes[1].set_ylabel('Trend')
                
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
                axes[2].set_ylabel('Seasonality')
            
            plt.tight_layout()
            plt.show()
        
        logger.info(f"Time series analysis completed: {results}")
        return results
    
    def fit_distribution(self, plot=False):
        """
        Fit a probability distribution to the demand data
        
        Parameters:
        -----------
        plot : bool
            Whether to generate plots
            
        Returns:
        --------
        tuple
            (distribution_name, distribution_params)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        demand = self.data['demand'].values
        
        # Try fitting different distributions
        distributions = [
            ('poisson', stats.poisson),
            ('normal', stats.norm),
            ('negative_binomial', stats.nbinom),
            ('gamma', stats.gamma)
        ]
        
        best_distribution = None
        best_params = None
        best_sse = float('inf')
        
        for dist_name, distribution in distributions:
            try:
                # Fit distribution
                params = distribution.fit(demand)
                
                # Calculate theoretical PDF/PMF
                if dist_name == 'poisson':
                    theoretical = distribution.pmf(np.arange(max(demand) + 1), *params)
                    hist_data = np.bincount(demand.astype(int)) / len(demand)
                    x_data = np.arange(len(hist_data))
                    # Pad theoretical with zeros if needed
                    if len(theoretical) < len(hist_data):
                        theoretical = np.pad(theoretical, (0, len(hist_data) - len(theoretical)))
                    else:
                        theoretical = theoretical[:len(hist_data)]
                    
                elif dist_name == 'negative_binomial':
                    theoretical = distribution.pmf(np.arange(max(demand) + 1), *params)
                    hist_data = np.bincount(demand.astype(int)) / len(demand)
                    x_data = np.arange(len(hist_data))
                    # Pad theoretical with zeros if needed
                    if len(theoretical) < len(hist_data):
                        theoretical = np.pad(theoretical, (0, len(hist_data) - len(theoretical)))
                    else:
                        theoretical = theoretical[:len(hist_data)]
                    
                else:
                    # For continuous distributions
                    x_data = np.linspace(min(demand), max(demand), 100)
                    theoretical = distribution.pdf(x_data, *params)
                    hist_data, bin_edges = np.histogram(demand, bins=20, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    # Need to interpolate to compare
                    from scipy.interpolate import interp1d
                    f = interp1d(x_data, theoretical, bounds_error=False, fill_value=0)
                    theoretical = f(bin_centers)
                    x_data = bin_centers
                
                # Calculate SSE (sum of squared errors)
                sse = np.sum((theoretical - hist_data)**2)
                
                if sse < best_sse:
                    best_sse = sse
                    best_distribution = dist_name
                    best_params = params
                
            except Exception as e:
                logger.warning(f"Error fitting {dist_name} distribution: {str(e)}")
        
        if best_distribution is None:
            logger.warning("No distribution could be fit to the data. Using empirical distribution.")
            best_distribution = 'empirical'
            best_params = None
        else:
            logger.info(f"Best fitting distribution: {best_distribution} with parameters {best_params}")
        
        # Store the results
        self.forecast_distribution = {
            'name': best_distribution,
            'params': best_params
        }
        
        # Generate plots if requested
        if plot and best_distribution != 'empirical':
            plt.figure(figsize=(10, 6))
            
            if best_distribution in ['poisson', 'negative_binomial']:
                # Discrete distributions
                x = np.arange(max(demand) + 1)
                if best_distribution == 'poisson':
                    y = stats.poisson.pmf(x, *best_params)
                else:
                    y = stats.nbinom.pmf(x, *best_params)
                
                plt.bar(x, np.bincount(demand.astype(int)) / len(demand), 
                       alpha=0.5, label='Observed')
                plt.plot(x, y, 'r-', label=f'Fitted {best_distribution}')
                
            else:
                # Continuous distributions
                x = np.linspace(min(demand), max(demand), 100)
                if best_distribution == 'normal':
                    y = stats.norm.pdf(x, *best_params)
                elif best_distribution == 'gamma':
                    y = stats.gamma.pdf(x, *best_params)
                
                plt.hist(demand, bins=20, density=True, alpha=0.5, label='Observed')
                plt.plot(x, y, 'r-', label=f'Fitted {best_distribution}')
            
            plt.legend()
            plt.title(f'Demand Distribution Fitting: {best_distribution}')
            plt.xlabel('Demand')
            plt.ylabel('Probability')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return best_distribution, best_params
    
    def train_forecast_models(self, train_ratio=0.8):
        """
        Train multiple forecasting models
        
        Parameters:
        -----------
        train_ratio : float
            Ratio of data to use for training (0 to 1)
            
        Returns:
        --------
        dict
            Dictionary of trained models with evaluation metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Train-test split
        train_size = int(len(self.data) * train_ratio)
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]
        
        # Time series analysis to guide model selection
        analysis = self.analyze_time_series(plot=False)
        
        # Define models to try
        models_to_fit = []
        
        # Always include naive forecast (for baseline)
        models_to_fit.append(('naive', {}))
        
        # Simple moving average
        models_to_fit.append(('moving_average', {'window': 4}))
        
        # Exponential smoothing with/without trend and seasonality
        if analysis['has_trend'] and analysis['has_seasonality']:
            # Triple exponential smoothing (Holt-Winters)
            models_to_fit.append((
                'holt_winters', 
                {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': analysis['seasonal_period']}
            ))
        elif analysis['has_trend']:
            # Double exponential smoothing (Holt)
            models_to_fit.append(('holt', {'trend': 'add'}))
        else:
            # Simple exponential smoothing
            models_to_fit.append(('ses', {}))
        
        # ARIMA models
        # Simplified model selection - in practice, would use auto.arima or similar
        if analysis['is_stationary']:
            models_to_fit.append(('arima', {'order': (1, 0, 1)}))
        else:
            models_to_fit.append(('arima', {'order': (1, 1, 1)}))
        
        if analysis['has_seasonality']:
            # Would use SARIMA in practice
            logger.info("Seasonality detected - would use SARIMA in full implementation")
        
        # Train each model and evaluate on test set
        model_results = {}
        forecast_horizon = len(test_data)
        
        for model_name, model_params in models_to_fit:
            try:
                logger.info(f"Training {model_name} model")
                
                if model_name == 'naive':
                    # Last value forecast
                    forecast = np.full(forecast_horizon, train_data['demand'].iloc[-1])
                
                elif model_name == 'moving_average':
                    # Simple moving average
                    window = model_params['window']
                    forecast = np.full(forecast_horizon, train_data['demand'].rolling(window=window).mean().iloc[-1])
                
                elif model_name == 'ses':
                    # Simple exponential smoothing
                    model = ExponentialSmoothing(
                        train_data['demand'],
                        trend=None,
                        seasonal=None
                    ).fit()
                    forecast = model.forecast(forecast_horizon)
                
                elif model_name == 'holt':
                    # Double exponential smoothing (Holt)
                    model = ExponentialSmoothing(
                        train_data['demand'],
                        trend=model_params['trend'],
                        seasonal=None
                    ).fit()
                    forecast = model.forecast(forecast_horizon)
                
                elif model_name == 'holt_winters':
                    # Triple exponential smoothing (Holt-Winters)
                    model = ExponentialSmoothing(
                        train_data['demand'],
                        trend=model_params['trend'],
                        seasonal=model_params['seasonal'],
                        seasonal_periods=model_params['seasonal_periods']
                    ).fit()
                    forecast = model.forecast(forecast_horizon)
                
                elif model_name == 'arima':
                    # ARIMA model
                    model = ARIMA(train_data['demand'], order=model_params['order']).fit()
                    forecast = model.forecast(forecast_horizon)
                
                # Ensure forecasts are non-negative for demand
                forecast = np.maximum(forecast, 0)
                
                # Calculate evaluation metrics
                actual = test_data['demand'].values
                mse = np.mean((forecast - actual) ** 2)
                mae = np.mean(np.abs(forecast - actual))
                mape = np.mean(np.abs((actual - forecast) / actual)) * 100 if np.all(actual > 0) else np.nan
                
                # Store results
                model_results[model_name] = {
                    'model': model if model_name not in ['naive', 'moving_average'] else None,
                    'params': model_params,
                    'forecast': forecast,
                    'mse': mse,
                    'mae': mae,
                    'mape': mape,
                    'actual': actual
                }
                
                logger.info(f"Model {model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Error training {model_name} model: {str(e)}")
        
        # Select best model based on MSE
        best_model_name = min(model_results, key=lambda x: model_results[x]['mse'])
        self.best_model = best_model_name
        self.fitted_models = model_results
        
        logger.info(f"Best model: {best_model_name} with MSE: {model_results[best_model_name]['mse']:.2f}")
        
        return model_results
    
    def forecast(self, periods=52, plot=False):
        """
        Generate forecasts for the specified number of periods
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        plot : bool
            Whether to generate plots
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with forecast values and prediction intervals
        """
        if not self.fitted_models or not self.best_model:
            logger.info("No trained models found. Training models first.")
            self.train_forecast_models()
        
        best_model_info = self.fitted_models[self.best_model]
        
        # Generate forecasts based on model type
        if self.best_model == 'naive':
            forecast_mean = np.full(periods, self.data['demand'].iloc[-1])
            model = None
        
        elif self.best_model == 'moving_average':
            window = best_model_info['params']['window']
            forecast_mean = np.full(periods, self.data['demand'].rolling(window=window).mean().iloc[-1])
            model = None
        
        else:
            model = best_model_info['model']
            
            if self.best_model in ['ses', 'holt', 'holt_winters']:
                # Re-fit the model on the full dataset
                if self.best_model == 'ses':
                    model = ExponentialSmoothing(
                        self.data['demand'],
                        trend=None,
                        seasonal=None
                    ).fit()
                elif self.best_model == 'holt':
                    model = ExponentialSmoothing(
                        self.data['demand'],
                        trend=best_model_info['params']['trend'],
                        seasonal=None
                    ).fit()
                elif self.best_model == 'holt_winters':
                    model = ExponentialSmoothing(
                        self.data['demand'],
                        trend=best_model_info['params']['trend'],
                        seasonal=best_model_info['params']['seasonal'],
                        seasonal_periods=best_model_info['params']['seasonal_periods']
                    ).fit()
                
                forecast_mean = model.forecast(periods)
            
            elif self.best_model == 'arima':
                # Re-fit ARIMA on full dataset
                model = ARIMA(
                    self.data['demand'], 
                    order=best_model_info['params']['order']
                ).fit()
                
                forecast_mean = model.forecast(periods)
        
        # Ensure non-negative demand forecasts
        forecast_mean = np.maximum(forecast_mean, 0)
        
        # Generate prediction intervals
        # For simplicity, using a percentage of the forecast
        # More sophisticated intervals would be model-specific
        forecast_std = forecast_mean * 0.2  # Simplified assumption
        forecast_lower = np.maximum(forecast_mean - 1.96 * forecast_std, 0)
        forecast_upper = forecast_mean + 1.96 * forecast_std
        
        # Create forecast DataFrame
        last_date = self.data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # Assume same frequency as the data
            freq = pd.infer_freq(self.data.index)
            if not freq:
                # Default to daily if frequency can't be inferred
                freq = 'D'
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
        else:
            forecast_dates = range(1, periods + 1)
        
        self.forecasts = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_bound': forecast_lower,
            'upper_bound': forecast_upper
        }, index=forecast_dates)
        
        # Set index name same as original data
        self.forecasts.index.name = self.data.index.name
        
        # Generate plot if requested
        if plot:
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(self.data.index, self.data['demand'], 'b-', label='Historical Demand')
            
            # Plot forecast
            plt.plot(self.forecasts.index, self.forecasts['forecast'], 'r-', label='Forecast')
            
            # Plot prediction intervals
            plt.fill_between(
                self.forecasts.index,
                self.forecasts['lower_bound'],
                self.forecasts['upper_bound'],
                color='r', alpha=0.2, label='95% Prediction Interval'
            )
            
            plt.title(f'Demand Forecast ({self.best_model.upper()} model)')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        logger.info(f"Generated {periods} period forecast using {self.best_model} model")
        return self.forecasts
    
    def generate_demand_parameters(self, method="fitted"):
        """
        Generate parameters for inventory optimization model
        
        Parameters:
        -----------
        method : str
            Method to use:
            - "fitted": Use fitted distribution
            - "empirical": Use empirical distribution
            - "forecast": Use forecast mean and variance
            
        Returns:
        --------
        dict
            Dictionary with demand parameters for inventory model
        """
        if method == "fitted" and self.forecast_distribution:
            # Use fitted distribution
            dist_name = self.forecast_distribution['name']
            dist_params = self.forecast_distribution['params']
            
            if dist_name == 'poisson':
                return {
                    'demand_type': 'poisson',
                    'demand_param': dist_params[0]  # Lambda
                }
            elif dist_name == 'negative_binomial':
                # Convert to uniform for now (simplified)
                mean = stats.nbinom.mean(*dist_params)
                var = stats.nbinom.var(*dist_params)
                # Approximate as uniform with same mean and variance
                width = np.sqrt(12 * var)
                low = max(0, mean - width/2)
                high = mean + width/2
                return {
                    'demand_type': 'uniform',
                    'demand_param': (int(low), int(high))
                }
            else:
                # Default to empirical for other distributions
                method = "empirical"
        
        if method == "empirical" or (method == "fitted" and not self.forecast_distribution):
            # Use empirical distribution
            demand = self.data['demand'].values
            
            # Check if data looks Poisson-like
            mean = np.mean(demand)
            var = np.var(demand)
            
            if 0.8 <= var/mean <= 1.2:
                # Close to Poisson
                return {
                    'demand_type': 'poisson',
                    'demand_param': mean
                }
            else:
                # Use uniform approximation
                low = max(0, int(np.floor(np.min(demand))))
                high = int(np.ceil(np.max(demand)))
                return {
                    'demand_type': 'uniform',
                    'demand_param': (low, high)
                }
        
        if method == "forecast" and self.forecasts is not None:
            # Use forecast mean and variance
            forecast_mean = np.mean(self.forecasts['forecast'])
            forecast_var = np.mean((self.forecasts['upper_bound'] - self.forecasts['forecast']) ** 2 / 3.84)  # Approx from 95% interval
            
            # Check if data looks Poisson-like
            if 0.8 <= forecast_var/forecast_mean <= 1.2:
                # Close to Poisson
                return {
                    'demand_type': 'poisson',
                    'demand_param': forecast_mean
                }
            else:
                # Use uniform approximation
                width = np.sqrt(12 * forecast_var)
                low = max(0, int(np.floor(forecast_mean - width/2)))
                high = int(np.ceil(forecast_mean + width/2))
                return {
                    'demand_type': 'uniform',
                    'demand_param': (low, high)
                }
        
        # Default to empirical calculation if all else fails
        logger.warning("Using default empirical method for demand parameters")
        demand = self.data['demand'].values
        mean = np.mean(demand)
        return {
            'demand_type': 'poisson',
            'demand_param': mean
        }

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Create a demand pattern with trend and seasonality
    trend = np.linspace(15, 25, 100)
    seasonality = 5 * np.sin(np.linspace(0, 2*np.pi*3, 100))
    noise = np.random.poisson(4, 100)
    demand = np.maximum(0, trend + seasonality + noise).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'demand': demand
    })
    
    # Initialize and use forecaster
    forecaster = DemandForecaster()
    forecaster.load_data(data)
    
    # Analyze time series
    forecaster.analyze_time_series(plot=True)
    
    # Fit distribution
    forecaster.fit_distribution(plot=True)
    
    # Train models
    forecaster.train_forecast_models()
    
    # Generate forecast
    forecasts = forecaster.forecast(periods=30, plot=True)
    
    # Get parameters for inventory model
    params = forecaster.generate_demand_parameters()
    print("\nOptimal Demand Parameters for Inventory Model:")
    print(params) 