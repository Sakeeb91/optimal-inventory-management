import numpy as np
import matplotlib.pyplot as plt
from inventory_dp import InventoryDP
import os

# Create directory for output figures
os.makedirs('figures', exist_ok=True)

def run_baseline_scenario():
    """Run baseline experiment scenario"""
    print("=== Running Baseline Scenario ===")
    
    # Initialize and solve the DP model with baseline parameters
    baseline_dp = InventoryDP(
        N=52,          # 52 weeks (1 year)
        M=100,         # Maximum warehouse capacity
        A_max=50,      # Maximum order size
        K=10,          # Fixed ordering cost
        c=2,           # Per-unit ordering cost
        h=1,           # Per-unit holding cost
        p=5,           # Per-unit shortage penalty
        demand_type='poisson',
        demand_param=20  # Mean demand (lambda for Poisson)
    )
    
    # Solve the DP
    baseline_dp.solve()
    
    # Visualization 1: Optimal Policy Heatmap
    baseline_dp.plot_policy_heatmap()
    plt.savefig('figures/baseline_policy_heatmap.png', dpi=300)
    plt.close()
    
    # Visualization 2: Value Function Plot
    baseline_dp.plot_value_function()
    plt.savefig('figures/baseline_value_function.png', dpi=300)
    plt.close()
    
    # Run simulation and create trajectory plot
    sim_results = baseline_dp.simulate(initial_state=0, num_simulations=10, seed=42)
    
    # Visualization 3: Sample Simulation Trajectory
    baseline_dp.plot_simulation_trajectory(sim_results)
    plt.savefig('figures/baseline_simulation_trajectory.png', dpi=300)
    plt.close()
    
    # Calculate and print some statistics
    total_costs = sim_results['total_costs']
    print(f"Average total cost: {np.mean(total_costs):.2f}")
    print(f"Min total cost: {np.min(total_costs):.2f}")
    print(f"Max total cost: {np.max(total_costs):.2f}")
    
    return baseline_dp, sim_results

def compare_demand_volatility(baseline_dp):
    """Compare different demand volatility scenarios"""
    print("=== Comparing Demand Volatility ===")
    
    # Initialize high volatility model (uniform demand with same mean)
    high_vol_dp = InventoryDP(
        N=52,
        M=100,
        A_max=50,
        K=10,
        c=2,
        h=1,
        p=5,
        demand_type='uniform',
        demand_param=(10, 30)  # Uniform between 10 and 30 (mean 20)
    )
    
    # Solve the DP
    high_vol_dp.solve()
    
    # Visualization 4: Compare Policy Heatmaps
    high_vol_dp.compare_policy_heatmaps(
        baseline_dp, 
        title="Policy Comparison: High vs. Low Demand Volatility"
    )
    plt.savefig('figures/volatility_policy_comparison.png', dpi=300)
    plt.close()
    
    # Run simulations for both models
    baseline_sim = baseline_dp.simulate(initial_state=0, num_simulations=1, seed=42)
    high_vol_sim = high_vol_dp.simulate(initial_state=0, num_simulations=1, seed=42)
    
    # Visualization 5: Compare Inventory Trajectories
    plt.figure(figsize=(12, 6))
    
    # Plot baseline (Poisson) inventory trajectory
    baseline_traj = baseline_sim['state_trajectories'][0]
    weeks = np.arange(baseline_dp.N + 1)
    plt.plot(weeks, baseline_traj, 'b-o', linewidth=2, label='Low Volatility (Poisson)')
    
    # Plot high volatility (Uniform) inventory trajectory
    high_vol_traj = high_vol_sim['state_trajectories'][0]
    plt.plot(weeks, high_vol_traj, 'r-^', linewidth=2, label='High Volatility (Uniform)')
    
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Week')
    plt.ylabel('Inventory Level')
    plt.title('Inventory Trajectories: Impact of Demand Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('figures/volatility_trajectory_comparison.png', dpi=300)
    plt.close()
    
    # Compare average costs
    baseline_costs = np.mean(baseline_dp.simulate(initial_state=0, num_simulations=50, seed=42)['total_costs'])
    high_vol_costs = np.mean(high_vol_dp.simulate(initial_state=0, num_simulations=50, seed=42)['total_costs'])
    
    print(f"Average cost (Low Volatility): {baseline_costs:.2f}")
    print(f"Average cost (High Volatility): {high_vol_costs:.2f}")
    print(f"Cost increase due to volatility: {high_vol_costs - baseline_costs:.2f} ({(high_vol_costs/baseline_costs - 1)*100:.2f}%)")
    
    return high_vol_dp

def compare_penalty_costs(baseline_dp):
    """Compare different penalty cost scenarios"""
    print("=== Comparing Penalty Costs ===")
    
    # Initialize high penalty cost model
    high_penalty_dp = InventoryDP(
        N=52,
        M=100,
        A_max=50,
        K=10,
        c=2,
        h=1,
        p=20,  # Higher penalty cost (p=20 vs baseline p=5)
        demand_type='poisson',
        demand_param=20
    )
    
    # Solve the DP
    high_penalty_dp.solve()
    
    # Visualization 6: Compare Policy Heatmaps
    high_penalty_dp.compare_policy_heatmaps(
        baseline_dp, 
        title="Policy Comparison: High vs. Low Penalty Cost"
    )
    plt.savefig('figures/penalty_policy_comparison.png', dpi=300)
    plt.close()
    
    # Run simulations
    baseline_sim = baseline_dp.simulate(initial_state=0, num_simulations=50, seed=42)
    high_penalty_sim = high_penalty_dp.simulate(initial_state=0, num_simulations=50, seed=42)
    
    # Compare average inventory levels
    baseline_avg_inv = np.mean([np.mean(traj) for traj in baseline_sim['state_trajectories']])
    high_penalty_avg_inv = np.mean([np.mean(traj) for traj in high_penalty_sim['state_trajectories']])
    
    print(f"Average inventory level (Low Penalty): {baseline_avg_inv:.2f}")
    print(f"Average inventory level (High Penalty): {high_penalty_avg_inv:.2f}")
    print(f"Inventory level increase: {high_penalty_avg_inv - baseline_avg_inv:.2f} ({(high_penalty_avg_inv/baseline_avg_inv - 1)*100:.2f}%)")
    
    return high_penalty_dp

def compare_deterministic_stochastic(baseline_dp):
    """Compare deterministic vs stochastic demand models"""
    print("=== Comparing Deterministic vs Stochastic Demand ===")
    
    # Initialize deterministic model with fixed demand
    deterministic_dp = InventoryDP(
        N=52,
        M=100,
        A_max=50,
        K=10,
        c=2,
        h=1,
        p=5,
        demand_type='deterministic',
        demand_param=20  # Fixed demand of 20 units each week
    )
    
    # Solve the DP
    deterministic_dp.solve()
    
    # Visualization 7: Compare Policies
    deterministic_dp.compare_policy_heatmaps(
        baseline_dp, 
        title="Policy Comparison: Deterministic vs. Stochastic Demand"
    )
    plt.savefig('figures/deterministic_stochastic_policy.png', dpi=300)
    plt.close()
    
    # Compare the cost-to-go (value function) at initial state
    stochastic_cost = baseline_dp.cost_to_go[0, baseline_dp.state_to_index(0)]
    deterministic_cost = deterministic_dp.cost_to_go[0, deterministic_dp.state_to_index(0)]
    
    print(f"Expected cost (Stochastic): {stochastic_cost:.2f}")
    print(f"Expected cost (Deterministic): {deterministic_cost:.2f}")
    print(f"Cost of uncertainty: {stochastic_cost - deterministic_cost:.2f} ({(stochastic_cost/deterministic_cost - 1)*100:.2f}%)")
    
    # Run deterministic simulation with stochastic demand to show suboptimality
    # when using deterministic policy in stochastic environment
    det_policy_stoch_demand = []
    
    # Use deterministic policy in stochastic environment (wrong model)
    current_state = 0
    state_traj = [current_state]
    action_traj = []
    total_cost = 0
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    for k in range(baseline_dp.N):
        # Get action from deterministic policy
        action = deterministic_dp.policy[k, deterministic_dp.state_to_index(current_state)]
        action_traj.append(action)
        
        # But face stochastic (Poisson) demand
        demand = np.random.poisson(baseline_dp.demand_param)
        demand = min(demand, baseline_dp.D_max)
        
        # Calculate costs
        order_cost = baseline_dp.K * (action > 0) + baseline_dp.c * action
        
        # Update state
        next_state = current_state + action - demand
        
        # Constrain next_state to valid range
        if next_state < baseline_dp.state_min:
            next_state = baseline_dp.state_min
        elif next_state > baseline_dp.state_max:
            next_state = baseline_dp.state_max
        
        # Calculate holding/shortage cost
        holding_cost = baseline_dp.h * max(0, next_state)
        shortage_cost = baseline_dp.p * max(0, -next_state)
        
        # Update total cost
        stage_cost = order_cost + holding_cost + shortage_cost
        total_cost += stage_cost
        
        # Update state
        current_state = next_state
        state_traj.append(current_state)
    
    # Add terminal cost
    terminal_cost = baseline_dp.h * max(0, current_state) + baseline_dp.p * max(0, -current_state)
    total_cost += terminal_cost
    
    # Also run stochastic policy in stochastic environment (correct model)
    stochastic_sim = baseline_dp.simulate(initial_state=0, num_simulations=1, seed=42)
    stochastic_cost = stochastic_sim['total_costs'][0]
    
    print(f"Total cost (Deterministic policy in stochastic environment): {total_cost:.2f}")
    print(f"Total cost (Stochastic policy in stochastic environment): {stochastic_cost:.2f}")
    print(f"Cost reduction from using correct model: {total_cost - stochastic_cost:.2f} ({(total_cost/stochastic_cost - 1)*100:.2f}%)")
    
    return deterministic_dp

def run_all_experiments():
    """Run all experiments and generate figures"""
    # Run baseline scenario
    baseline_dp, baseline_sim = run_baseline_scenario()
    
    # Compare different scenarios
    high_vol_dp = compare_demand_volatility(baseline_dp)
    high_penalty_dp = compare_penalty_costs(baseline_dp)
    deterministic_dp = compare_deterministic_stochastic(baseline_dp)
    
    print("\nAll experiments completed successfully.")
    print("Figures saved in the 'figures' directory.")

if __name__ == "__main__":
    run_all_experiments() 