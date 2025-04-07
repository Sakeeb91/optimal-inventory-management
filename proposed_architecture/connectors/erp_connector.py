import os
import json
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ERPConnector:
    """
    Connector for retrieving inventory and cost data from ERP systems
    
    Supported systems:
    - SAP
    - Oracle ERP
    - Microsoft Dynamics
    - NetSuite
    - Custom REST API
    """
    
    def __init__(self, erp_type="custom_api"):
        """
        Initialize the ERP connector
        
        Parameters:
        -----------
        erp_type : str
            Type of ERP system to connect to:
            - "sap" - SAP ERP
            - "oracle" - Oracle ERP
            - "dynamics" - Microsoft Dynamics
            - "netsuite" - NetSuite
            - "custom_api" - Custom REST API
        """
        self.erp_type = erp_type
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load connection configuration from environment variables"""
        self.config = {
            "api_base_url": os.getenv(f"{self.erp_type.upper()}_API_URL", ""),
            "api_key": os.getenv(f"{self.erp_type.upper()}_API_KEY", ""),
            "username": os.getenv(f"{self.erp_type.upper()}_USERNAME", ""),
            "password": os.getenv(f"{self.erp_type.upper()}_PASSWORD", ""),
            "warehouse_id": os.getenv("WAREHOUSE_ID", ""),
            "product_category": os.getenv("PRODUCT_CATEGORY", "")
        }
        
        if not self.config["api_base_url"]:
            logger.warning(f"API base URL not configured for {self.erp_type}")
    
    def _get_auth_headers(self):
        """Get authentication headers for API requests"""
        if self.erp_type == "custom_api":
            return {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
        elif self.erp_type == "sap":
            # SAP specific authentication
            return {
                "Authorization": f"Basic {self.config['api_key']}",
                "Content-Type": "application/json"
            }
        # Add other ERP systems as needed
        else:
            return {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
    
    def get_inventory_data(self, product_id=None, from_date=None, to_date=None):
        """
        Retrieve inventory data from the ERP system
        
        Parameters:
        -----------
        product_id : str, optional
            ID of the specific product to retrieve data for
        from_date : str, optional
            Start date for the data in ISO format (YYYY-MM-DD)
        to_date : str, optional
            End date for the data in ISO format (YYYY-MM-DD)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing inventory data with columns:
            - product_id
            - product_name
            - current_stock
            - available_stock
            - reserved_stock
            - min_stock
            - max_stock
            - reorder_point
            - last_updated
        """
        logger.info(f"Retrieving inventory data from {self.erp_type}")
        
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Construct API endpoint and parameters
            if self.erp_type == "custom_api":
                endpoint = f"{self.config['api_base_url']}/inventory"
                params = {
                    "warehouse_id": self.config["warehouse_id"],
                    "from_date": from_date,
                    "to_date": to_date
                }
                
                if product_id:
                    params["product_id"] = product_id
                
                # Make API request
                response = requests.get(
                    endpoint,
                    params=params,
                    headers=self._get_auth_headers()
                )
                
                # Process response
                if response.status_code == 200:
                    data = response.json()
                    return pd.DataFrame(data["inventory_items"])
                else:
                    logger.error(f"Error retrieving inventory data: {response.status_code} - {response.text}")
                    return pd.DataFrame()
            
            # Implement other ERP-specific logic here
            else:
                logger.warning(f"Retrieval method not implemented for {self.erp_type}")
                return self._get_dummy_inventory_data(product_id)
        
        except Exception as e:
            logger.error(f"Error retrieving inventory data: {str(e)}")
            return pd.DataFrame()
    
    def get_cost_parameters(self, product_id=None):
        """
        Retrieve cost parameters for the inventory model
        
        Parameters:
        -----------
        product_id : str, optional
            ID of the specific product to retrieve cost data for
            
        Returns:
        --------
        dict
            Dictionary containing cost parameters:
            - ordering_fixed_cost (K)
            - ordering_variable_cost (c)
            - holding_cost (h)
            - shortage_cost (p)
        """
        logger.info(f"Retrieving cost parameters from {self.erp_type}")
        
        try:
            # Construct API endpoint and parameters
            if self.erp_type == "custom_api":
                endpoint = f"{self.config['api_base_url']}/cost_parameters"
                params = {
                    "warehouse_id": self.config["warehouse_id"]
                }
                
                if product_id:
                    params["product_id"] = product_id
                
                # Make API request
                response = requests.get(
                    endpoint,
                    params=params,
                    headers=self._get_auth_headers()
                )
                
                # Process response
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error retrieving cost parameters: {response.status_code} - {response.text}")
                    return self._get_default_cost_parameters()
            
            # Implement other ERP-specific logic here
            else:
                logger.warning(f"Retrieval method not implemented for {self.erp_type}")
                return self._get_default_cost_parameters()
        
        except Exception as e:
            logger.error(f"Error retrieving cost parameters: {str(e)}")
            return self._get_default_cost_parameters()
    
    def get_historical_demand(self, product_id, num_periods=52):
        """
        Retrieve historical demand data for a product
        
        Parameters:
        -----------
        product_id : str
            ID of the product to retrieve demand data for
        num_periods : int, optional
            Number of time periods (default: 52 weeks)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing demand data with columns:
            - date
            - demand
        """
        logger.info(f"Retrieving historical demand data from {self.erp_type}")
        
        try:
            # Construct API endpoint and parameters
            if self.erp_type == "custom_api":
                endpoint = f"{self.config['api_base_url']}/demand_history"
                params = {
                    "product_id": product_id,
                    "num_periods": num_periods
                }
                
                # Make API request
                response = requests.get(
                    endpoint,
                    params=params,
                    headers=self._get_auth_headers()
                )
                
                # Process response
                if response.status_code == 200:
                    data = response.json()
                    return pd.DataFrame(data["demand_history"])
                else:
                    logger.error(f"Error retrieving demand history: {response.status_code} - {response.text}")
                    return self._get_dummy_demand_data(num_periods)
            
            # Implement other ERP-specific logic here
            else:
                logger.warning(f"Retrieval method not implemented for {self.erp_type}")
                return self._get_dummy_demand_data(num_periods)
        
        except Exception as e:
            logger.error(f"Error retrieving demand history: {str(e)}")
            return self._get_dummy_demand_data(num_periods)
    
    def _get_dummy_inventory_data(self, product_id=None):
        """Generate dummy inventory data for testing"""
        import numpy as np
        
        # Generate a list of products
        if product_id:
            products = [{"id": product_id, "name": f"Product {product_id}"}]
        else:
            products = [
                {"id": f"P{i:03d}", "name": f"Product {i}"} 
                for i in range(1, 11)
            ]
        
        # Generate inventory data for each product
        inventory_data = []
        for product in products:
            current_stock = np.random.randint(10, 100)
            reserved = np.random.randint(0, current_stock // 3)
            
            inventory_data.append({
                "product_id": product["id"],
                "product_name": product["name"],
                "current_stock": current_stock,
                "available_stock": current_stock - reserved,
                "reserved_stock": reserved,
                "min_stock": max(5, current_stock // 5),
                "max_stock": current_stock * 2,
                "reorder_point": max(10, current_stock // 3),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return pd.DataFrame(inventory_data)
    
    def _get_default_cost_parameters(self):
        """Get default cost parameters"""
        return {
            "ordering_fixed_cost": 10.0,
            "ordering_variable_cost": 2.0,
            "holding_cost": 1.0,
            "shortage_cost": 5.0
        }
    
    def _get_dummy_demand_data(self, num_periods=52):
        """Generate dummy demand data for testing"""
        import numpy as np
        
        # Generate dates (assuming weekly periods)
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=7*i)).strftime("%Y-%m-%d") 
                for i in range(num_periods)]
        
        # Generate demand values (Poisson distributed)
        demands = np.random.poisson(20, num_periods)
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "demand": demands
        })
        
        # Sort by date
        df = df.sort_values("date")
        
        return df

# Example usage
if __name__ == "__main__":
    erp = ERPConnector(erp_type="custom_api")
    
    # Get inventory data
    inventory_df = erp.get_inventory_data()
    print("\nInventory Data:")
    print(inventory_df.head())
    
    # Get cost parameters
    cost_params = erp.get_cost_parameters()
    print("\nCost Parameters:")
    print(json.dumps(cost_params, indent=2))
    
    # Get historical demand
    demand_df = erp.get_historical_demand(product_id="P001")
    print("\nHistorical Demand:")
    print(demand_df.head()) 