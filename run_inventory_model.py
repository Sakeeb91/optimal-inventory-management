import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from inventory_dp import InventoryDP

def load_demand_data(file_path):
    """Load the processed demand data"""
    print(f"Loading demand data from {file_path}")
    data = pd.read_csv(file_path)
    return data

def optimize_inventory(demand_data, demand_type, demand_param, config=None):
    """Create and solve the inventory optimization model"""
    print(f"Creating inventory model with {demand_type} demand distribution")
    
    # Set default config if not provided
    if config is None:
        config = {
            'N': 52,          # 52 weeks planning horizon
            'M': 100,         # Maximum warehouse capacity
            'A_max': 50,      # Maximum order size
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

def run_simulation(dp_model, initial_state=0, num_simulations=10):
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

def main():
    parser = argparse.ArgumentParser(description='Run inventory optimization using demand data')
    parser.add_argument('--demand_file', type=str, required=True, help='Path to demand data CSV file')
    parser.add_argument('--initial_inventory', type=int, default=0, help='Initial inventory level')
    parser.add_argument('--simulations', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--planning_horizon', type=int, default=52, help='Planning horizon (weeks)')
    parser.add_argument('--max_capacity', type=int, default=100, help='Maximum warehouse capacity')
    parser.add_argument('--fixed_cost', type=float, default=10, help='Fixed ordering cost')
    parser.add_argument('--unit_cost', type=float, default=2, help='Per-unit ordering cost')
    parser.add_argument('--holding_cost', type=float, default=1, help='Per-unit holding cost')
    parser.add_argument('--shortage_penalty', type=float, default=5, help='Per-unit shortage penalty')
    
    args = parser.parse_args()
    
    # Load the demand data
    demand_data = load_demand_data(args.demand_file)
    
    # Extract product code from filename
    product_code = os.path.basename(args.demand_file).split('_')[1].split('.')[0]
    
    # Analyze demand patterns
    quantity = demand_data['Quantity']
    mean_demand = quantity.mean()
    variance = quantity.var()
    
    # Determine demand distribution
    if abs(mean_demand - variance) / mean_demand < 0.25:  # Within 25% difference
        demand_type = 'poisson'
        demand_param = mean_demand
    else:
        # Use uniform distribution as a fallback
        min_demand = max(0, int(quantity[quantity > 0].min()))
        max_demand = int(quantity.max())
        demand_type = 'uniform'
        demand_param = (min_demand, max_demand)
    
    print(f"Using {demand_type} demand distribution with parameter(s): {demand_param}")
    
    # Set up configuration
    config = {
        'N': args.planning_horizon,
        'M': args.max_capacity,
        'A_max': min(50, int(max_demand * 3)) if 'max_demand' in locals() else 50,
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