import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

def load_retail_data(file_path):
    """Load retail data from an Excel file"""
    print(f"Loading data from {file_path}...")
    # Read all sheets and concatenate
    dfs = []
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        print(f"Reading sheet: {sheet_name}")
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        dfs.append(df)
    
    # Concatenate all sheets
    data = pd.concat(dfs, ignore_index=True)
    
    print(f"Data loaded: {len(data)} rows")
    return data

def clean_data(data):
    """Clean and preprocess the retail data"""
    print("Cleaning data...")
    
    # Convert InvoiceDate to datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    
    # Extract date part
    data['Date'] = data['InvoiceDate'].dt.date
    
    # Filter out canceled orders (start with 'C')
    data = data[~data['Invoice'].astype(str).str.startswith('C')]
    
    # Filter out returns (negative quantities)
    data = data[data['Quantity'] > 0]
    
    # Filter out missing stock codes or descriptions
    data = data.dropna(subset=['StockCode', 'Description'])
    
    print(f"Data cleaned: {len(data)} rows remaining")
    return data

def create_demand_time_series(data, product_code=None):
    """Create a time series of daily demand"""
    print("Creating demand time series...")
    
    if product_code:
        # Filter for specific product
        print(f"Filtering for product: {product_code}")
        data = data[data['StockCode'] == product_code]
        
        if len(data) == 0:
            print(f"No data found for product code {product_code}")
            return None
    else:
        # Get the product with the most transactions
        product_counts = data['StockCode'].value_counts()
        product_code = product_counts.index[0]
        print(f"Selected most frequent product: {product_code}")
        data = data[data['StockCode'] == product_code]
    
    # Get product description
    product_description = data['Description'].iloc[0]
    print(f"Product: {product_code} - {product_description}")
    
    # Aggregate demand by day
    daily_demand = data.groupby('Date')['Quantity'].sum().reset_index()
    
    # Create continuous date range
    min_date = daily_demand['Date'].min()
    max_date = daily_demand['Date'].max()
    
    # Create continuous date range
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    continuous_dates = pd.DataFrame({'Date': date_range})
    continuous_dates['Date'] = continuous_dates['Date'].dt.date
    
    # Merge to get continuous time series with zeros for missing dates
    continuous_demand = pd.merge(continuous_dates, daily_demand, on='Date', how='left')
    continuous_demand['Quantity'] = continuous_demand['Quantity'].fillna(0)
    
    print(f"Time series created: {len(continuous_demand)} days from {min_date} to {max_date}")
    
    # Create CSV
    output_file = f"data/demand_{product_code}.csv"
    continuous_demand.to_csv(output_file, index=False)
    print(f"Time series saved to {output_file}")
    
    return {
        'data': continuous_demand,
        'product_code': product_code,
        'description': product_description,
        'file_path': output_file
    }

def analyze_demand(demand_data):
    """Analyze demand patterns"""
    print("Analyzing demand patterns...")
    
    data = demand_data['data']
    product_code = demand_data['product_code']
    description = demand_data['description']
    
    # Basic statistics
    quantity = data['Quantity']
    stats = {
        'mean': quantity.mean(),
        'median': quantity.median(),
        'min': quantity.min(),
        'max': quantity.max(),
        'std': quantity.std(),
        'days_with_demand': (quantity > 0).sum(),
        'total_demand': quantity.sum()
    }
    
    print("\nDemand Statistics:")
    print(f"Product: {product_code} - {description}")
    print(f"Total Demand: {stats['total_demand']:.0f} units")
    print(f"Period: {len(data)} days")
    print(f"Days with demand: {stats['days_with_demand']} ({stats['days_with_demand']/len(data)*100:.1f}%)")
    print(f"Average Daily Demand: {stats['mean']:.2f} units")
    print(f"Median Daily Demand: {stats['median']:.2f} units")
    print(f"Min Daily Demand: {stats['min']:.0f} units")
    print(f"Max Daily Demand: {stats['max']:.0f} units")
    print(f"Standard Deviation: {stats['std']:.2f} units")
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Time series plot
    plt.subplot(2, 1, 1)
    plt.plot(data['Date'], data['Quantity'])
    plt.title(f"Daily Demand for Product {product_code}")
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.grid(True)
    
    # Histogram
    plt.subplot(2, 1, 2)
    plt.hist(data['Quantity'], bins=30)
    plt.title(f"Demand Distribution for Product {product_code}")
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"data/demand_{product_code}_analysis.png"
    plt.savefig(plot_file)
    print(f"Analysis plot saved to {plot_file}")
    
    return stats

def get_demand_distribution(demand_data):
    """Estimate the demand distribution parameters for InventoryDP"""
    data = demand_data['data']
    quantity = data['Quantity']
    
    # Check if Poisson is a good fit (mean ≈ variance)
    mean = quantity.mean()
    variance = quantity.var()
    
    print("\nDemand Distribution Analysis:")
    print(f"Mean: {mean:.2f}")
    print(f"Variance: {variance:.2f}")
    
    if abs(mean - variance) / mean < 0.25:  # Within 25% difference
        print("Demand approximately follows a Poisson distribution")
        return {
            'type': 'poisson',
            'param': mean
        }
    else:
        # Try uniform distribution as a fallback
        # Round to integers for discrete demand
        min_demand = int(max(0, quantity[quantity > 0].min()))
        max_demand = int(quantity.max())
        
        print("Demand doesn't follow Poisson closely; approximating with Uniform distribution")
        return {
            'type': 'uniform',
            'param': (min_demand, max_demand)
        }

def main():
    parser = argparse.ArgumentParser(description='Process retail data and create demand time series')
    parser.add_argument('--product', type=str, help='Product stock code to analyze', default=None)
    args = parser.parse_args()
    
    # Load and process data
    data = load_retail_data('data/online_retail_II.xlsx')
    cleaned_data = clean_data(data)
    
    # Create demand time series
    demand_data = create_demand_time_series(cleaned_data, args.product)
    
    if demand_data:
        # Analyze demand
        stats = analyze_demand(demand_data)
        
        # Get distribution parameters for InventoryDP
        distribution = get_demand_distribution(demand_data)
        
        print("\nParameters for InventoryDP:")
        print(f"demand_type='{distribution['type']}',")
        print(f"demand_param={distribution['param']}")

if __name__ == "__main__":
    main() 