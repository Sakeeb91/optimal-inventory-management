import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from inventory_dp import InventoryDP

def load_demand_data(file_path):
    """Load the processed demand data"""
    print(f"Loading demand data from {file_path}")
    data = pd.read_csv(file_path)
    return data

def simplify_demand_distribution(demand_data, method='poisson'):
    """Create a simplified demand distribution for faster computation"""
    quantity = demand_data['Quantity']
    
    if method == 'poisson':
        # Use Poisson with the average demand
        demand_type = 'poisson'
        demand_param = quantity.mean()
        print(f"Using simplified Poisson distribution with mean = {demand_param:.2f}")
        
    elif method == 'simplified_uniform':
        # Use a uniform with fewer buckets
        demand_type = 'uniform'
        # Round to nearest 100 for large values to reduce computation
        min_demand = max(0, int(np.percentile(quantity[quantity > 0], 5)))
        max_demand = int(np.percentile(quantity, 95))  # Use 95th percentile to avoid extremes
        demand_param = (min_demand, max_demand)
        print(f"Using simplified Uniform distribution with range = {demand_param}")
        
    elif method == 'deterministic':
        # Use deterministic demand with the average
        demand_type = 'deterministic'
        demand_param = int(quantity.mean())
        print(f"Using deterministic demand with value = {demand_param}")
    
    return demand_type, demand_param

def optimize_inventory(demand_data, demand_type, demand_param, config=None):
    """Create and solve the inventory optimization model"""
    print(f"Creating inventory model with {demand_type} demand distribution")
    
    # Set default config if not provided
    if config is None:
        config = {
            'N': 12,          # 12 weeks planning horizon (reduced from 52)
            'M': 500,         # 500 max capacity (reasonable for this dataset)
            'A_max': 300,     # Reasonable max order size
            'K': 10,          # Fixed ordering cost
            'c': 2,           # Per-unit ordering cost
            'h': 1,           # Per-unit holding cost
            'p': 5,           # Per-unit shortage penalty
        }
    
    # Create the model
    dp_model = InventoryDP(
        N=config['N'],
        M=config['M'],
        A_max=config['A_max'],
        K=config['K'],
        c=config['c'],
        h=config['h'],
        p=config['p'],
        demand_type=demand_type,
        demand_param=demand_param
    )
    
    # Solve the model
    print("Solving the inventory optimization model...")
    dp_model.solve()
    
    return dp_model

def run_simulation(dp_model, initial_state=0, num_simulations=5):
    """Run simulations using the optimized policy"""
    print(f"Running {num_simulations} simulations with initial inventory {initial_state}")
    
    # Run the simulations
    sim_results = dp_model.simulate(
        initial_state=initial_state,
        num_simulations=num_simulations,
        seed=42
    )
    
    # Calculate statistics
    total_costs = sim_results['total_costs']
    print(f"Simulation Results:")
    print(f"Average total cost: {sum(total_costs)/len(total_costs):.2f}")
    print(f"Min total cost: {min(total_costs):.2f}")
    print(f"Max total cost: {max(total_costs):.2f}")
    
    return sim_results

def analyze_results(dp_model, sim_results, product_code, output_dir='output'):
    """Analyze and visualize the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the optimal policy
    print("Generating policy heatmap...")
    dp_model.plot_policy_heatmap()
    plt.savefig(f"{output_dir}/policy_heatmap_{product_code}.png", dpi=300)
    plt.close()
    
    # Plot the value function
    print("Generating value function plot...")
    dp_model.plot_value_function()
    plt.savefig(f"{output_dir}/value_function_{product_code}.png", dpi=300)
    plt.close()
    
    # Plot a sample simulation trajectory
    print("Generating simulation trajectory plot...")
    dp_model.plot_simulation_trajectory(sim_results)
    plt.savefig(f"{output_dir}/simulation_trajectory_{product_code}.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")
    
    # Print recommended policy insights
    print("\nInventory Policy Insights:")
    
    # Analyze the policy for the first stage (k=0)
    policy_stage0 = dp_model.policy[0]
    
    # Find the reorder point (smallest inventory where action > 0)
    reorder_point = None
    for idx, action in enumerate(policy_stage0):
        state = dp_model.index_to_state(idx)
        if state >= 0 and action > 0:
            reorder_point = state
            break
    
    # Find the order-up-to level (reorder point + order quantity)
    if reorder_point is not None:
        idx = dp_model.state_to_index(reorder_point)
        order_qty = policy_stage0[idx]
        order_up_to = reorder_point + order_qty
        
        print(f"Recommended (s,S) Policy:")
        print(f"  Reorder Point (s): {reorder_point} units")
        print(f"  Order-up-to Level (S): {order_up_to} units")
        print(f"  Order Quantity: {order_qty} units when inventory reaches {reorder_point}")

def main():
    parser = argparse.ArgumentParser(description='Run optimized inventory model using demand data')
    parser.add_argument('--demand_file', type=str, required=True, help='Path to demand data CSV file')
    parser.add_argument('--demand_model', type=str, default='poisson', 
                      choices=['poisson', 'simplified_uniform', 'deterministic'],
                      help='Simplified demand model to use')
    parser.add_argument('--initial_inventory', type=int, default=0, help='Initial inventory level')
    parser.add_argument('--simulations', type=int, default=5, help='Number of simulations to run')
    parser.add_argument('--planning_horizon', type=int, default=12, help='Planning horizon (weeks)')
    parser.add_argument('--max_capacity', type=int, default=500, help='Maximum warehouse capacity')
    parser.add_argument('--fixed_cost', type=float, default=10, help='Fixed ordering cost')
    parser.add_argument('--unit_cost', type=float, default=2, help='Per-unit ordering cost')
    parser.add_argument('--holding_cost', type=float, default=1, help='Per-unit holding cost')
    parser.add_argument('--shortage_penalty', type=float, default=5, help='Per-unit shortage penalty')
    
    args = parser.parse_args()
    
    # Load the demand data
    demand_data = load_demand_data(args.demand_file)
    
    # Extract product code from filename
    product_code = os.path.basename(args.demand_file).split('_')[1].split('.')[0]
    
    # Create simplified demand distribution
    demand_type, demand_param = simplify_demand_distribution(demand_data, method=args.demand_model)
    
    # Set up configuration
    config = {
        'N': args.planning_horizon,
        'M': args.max_capacity,
        'A_max': min(300, args.max_capacity // 2),  # More reasonable max order size
        'K': args.fixed_cost,
        'c': args.unit_cost,
        'h': args.holding_cost,
        'p': args.shortage_penalty,
    }
    
    # Run the optimization
    dp_model = optimize_inventory(demand_data, demand_type, demand_param, config)
    
    # Run simulations
    sim_results = run_simulation(dp_model, 
                                initial_state=args.initial_inventory, 
                                num_simulations=args.simulations)
    
    # Analyze and visualize results
    analyze_results(dp_model, sim_results, product_code)

if __name__ == "__main__":
    main() 