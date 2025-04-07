import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from inventory_dp import InventoryDP
import markdown
import shutil
from run_experiments import run_baseline_scenario, compare_demand_volatility, compare_penalty_costs, compare_deterministic_stochastic

class ExperimentDashboard:
    """
    Handles experiment runs with timestamped folders and creates 
    documentation for each plot and a dashboard for the entire run.
    """
    
    def __init__(self):
        """Initialize the dashboard manager"""
        # Create a timestamped folder for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"runs_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create plots directory within the run directory
        self.plots_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # List to track all plots and their descriptions
        self.plots = []
        
        # Redirect matplotlib to save to our plots directory
        plt.rcParams["savefig.directory"] = self.plots_dir
    
    def save_plot(self, title, description, experiment_details=""):
        """
        Save the current matplotlib figure to the plots directory
        and create a markdown file with its description
        
        Parameters:
        -----------
        title : str
            Title for the plot (used in filename)
        description : str
            Short description of what the plot shows
        experiment_details : str
            Additional details about the experiment setup
        """
        # Clean the title for use in filename
        clean_title = title.lower().replace(" ", "_").replace(":", "")
        filename = f"{clean_title}.png"
        filepath = os.path.join(self.plots_dir, filename)
        
        # Save the current figure
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        # Create markdown file for this plot
        md_filename = f"{clean_title}.md"
        md_filepath = os.path.join(self.plots_dir, md_filename)
        
        with open(md_filepath, 'w') as f:
            f.write(f"# {title}\n\n")
            f.write(f"![{title}](./{filename})\n\n")
            f.write(f"## Description\n\n{description}\n\n")
            if experiment_details:
                f.write(f"## Experiment Details\n\n{experiment_details}\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add to plots list for dashboard
        self.plots.append({
            'title': title,
            'filename': filename,
            'description': description,
            'md_filename': md_filename
        })
    
    def create_dashboard(self, run_title="Inventory Control Experiment Run", run_summary=""):
        """
        Create a markdown dashboard summarizing all plots from this run
        
        Parameters:
        -----------
        run_title : str
            Title for this experiment run
        run_summary : str
            Summary of the experiment results
        """
        dashboard_path = os.path.join(self.run_dir, "dashboard.md")
        
        with open(dashboard_path, 'w') as f:
            # Header
            f.write(f"# {run_title}\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            if run_summary:
                f.write(f"## Summary\n\n{run_summary}\n\n")
            
            # Plots section
            f.write("## Plots\n\n")
            
            for i, plot in enumerate(self.plots):
                f.write(f"### {i+1}. {plot['title']}\n\n")
                f.write(f"![{plot['title']}](./plots/{plot['filename']})\n\n")
                f.write(f"{plot['description']}\n\n")
                f.write(f"[Details](./plots/{plot['md_filename']})\n\n")
                
                if i < len(self.plots) - 1:
                    f.write("---\n\n")  # Add separator between plots
        
        print(f"Dashboard created at {dashboard_path}")
        return dashboard_path
    
    def run_baseline_experiment(self):
        """Run the baseline experiment and save plots with descriptions"""
        print("=== Running Baseline Scenario ===")
        
        # Initialize and solve the DP model
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
        
        experiment_details = """
Parameters:
- Planning horizon: 52 weeks
- Maximum warehouse capacity: 100 units
- Maximum order size: 50 units
- Fixed ordering cost (K): 10
- Per-unit ordering cost (c): 2
- Per-unit holding cost (h): 1
- Per-unit shortage penalty (p): 5
- Demand follows Poisson distribution with mean 20
"""
        
        # Solve the DP
        baseline_dp.solve()
        
        # Visualization 1: Optimal Policy Heatmap
        baseline_dp.plot_policy_heatmap()
        self.save_plot(
            "Optimal Policy Heatmap", 
            "Shows how the optimal ordering quantity (action) changes based on current inventory level (state) across different weeks in the planning horizon. Each subplot represents a different week from the planning horizon.",
            experiment_details
        )
        
        # Visualization 2: Value Function Plot
        baseline_dp.plot_value_function()
        self.save_plot(
            "Value Function Plot", 
            "Displays the expected future cost (value function) for each inventory level at different stages of the planning horizon. The convex shape indicates the trade-off between holding too much inventory (increasing costs on the right) and having shortages (increasing costs on the left).",
            experiment_details
        )
        
        # Run simulation and create trajectory plot
        sim_results = baseline_dp.simulate(initial_state=0, num_simulations=10, seed=42)
        
        # Calculate and log statistics
        total_costs = sim_results['total_costs']
        stats = f"Average total cost: {np.mean(total_costs):.2f}\n"
        stats += f"Min total cost: {np.min(total_costs):.2f}\n"
        stats += f"Max total cost: {np.max(total_costs):.2f}\n"
        print(stats)
        
        simulation_details = experiment_details + f"""
Simulation details:
- 10 simulation runs
- Initial inventory: 0 units
- Random seed: 42
- {stats}
"""
        
        # Visualization 3: Sample Simulation Trajectory
        baseline_dp.plot_simulation_trajectory(sim_results)
        self.save_plot(
            "Simulation Trajectory", 
            "Illustrates the system behavior over time under the optimal policy. The top panel shows inventory levels, the middle panel shows order quantities, and the bottom panel shows realized demand. This helps visualize how the inventory control system responds to stochastic demand.",
            simulation_details
        )
        
        return baseline_dp, sim_results, stats
    
    def run_demand_volatility_experiment(self, baseline_dp):
        """Run demand volatility comparison experiment and save plots"""
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
        
        experiment_details = """
Comparison between:
1. Low volatility: Poisson demand with mean 20
2. High volatility: Uniform demand between 10 and 30 (same mean of 20 but higher variance)

Both models use:
- Planning horizon: 52 weeks
- Maximum warehouse capacity: 100 units
- Maximum order size: 50 units
- Fixed ordering cost (K): 10
- Per-unit ordering cost (c): 2
- Per-unit holding cost (h): 1
- Per-unit shortage penalty (p): 5
"""
        
        # Solve the DP
        high_vol_dp.solve()
        
        # Visualization 4: Compare Policy Heatmaps
        high_vol_dp.compare_policy_heatmaps(
            baseline_dp, 
            title="Policy Comparison: High vs. Low Demand Volatility"
        )
        self.save_plot(
            "Demand Volatility Policy Comparison", 
            "Compares the optimal ordering policies between low volatility (Poisson) and high volatility (Uniform) demand. Higher volatility generally leads to higher safety stocks to hedge against uncertainty.",
            experiment_details
        )
        
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
        
        self.save_plot(
            "Demand Volatility Trajectory Comparison", 
            "Shows how inventory levels evolve over time under different demand volatility scenarios. The high volatility case typically maintains higher inventory levels as a buffer against uncertainty.",
            experiment_details
        )
        
        # Compare average costs
        baseline_costs = np.mean(baseline_dp.simulate(initial_state=0, num_simulations=50, seed=42)['total_costs'])
        high_vol_costs = np.mean(high_vol_dp.simulate(initial_state=0, num_simulations=50, seed=42)['total_costs'])
        
        volatility_stats = f"Average cost (Low Volatility): {baseline_costs:.2f}\n"
        volatility_stats += f"Average cost (High Volatility): {high_vol_costs:.2f}\n"
        volatility_stats += f"Cost increase due to volatility: {high_vol_costs - baseline_costs:.2f} ({(high_vol_costs/baseline_costs - 1)*100:.2f}%)\n"
        print(volatility_stats)
        
        return high_vol_dp, volatility_stats
    
    def run_penalty_costs_experiment(self, baseline_dp):
        """Run penalty cost comparison experiment and save plots"""
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
        
        experiment_details = """
Comparison between:
1. Low shortage penalty: p=5
2. High shortage penalty: p=20

Both models use:
- Planning horizon: 52 weeks
- Maximum warehouse capacity: 100 units
- Maximum order size: 50 units
- Fixed ordering cost (K): 10
- Per-unit ordering cost (c): 2
- Per-unit holding cost (h): 1
- Poisson demand with mean 20
"""
        
        # Solve the DP
        high_penalty_dp.solve()
        
        # Visualization 6: Compare Policy Heatmaps
        high_penalty_dp.compare_policy_heatmaps(
            baseline_dp, 
            title="Policy Comparison: High vs. Low Penalty Cost"
        )
        self.save_plot(
            "Penalty Cost Policy Comparison", 
            "Compares the optimal ordering policies between low penalty cost (p=5) and high penalty cost (p=20). Higher shortage penalties lead to more aggressive ordering to avoid stockouts.",
            experiment_details
        )
        
        # Run simulations
        baseline_sim = baseline_dp.simulate(initial_state=0, num_simulations=50, seed=42)
        high_penalty_sim = high_penalty_dp.simulate(initial_state=0, num_simulations=50, seed=42)
        
        # Compare average inventory levels
        baseline_avg_inv = np.mean([np.mean(traj) for traj in baseline_sim['state_trajectories']])
        high_penalty_avg_inv = np.mean([np.mean(traj) for traj in high_penalty_sim['state_trajectories']])
        
        # Create a bar chart comparing average inventory levels
        plt.figure(figsize=(10, 6))
        plt.bar(['Low Penalty (p=5)', 'High Penalty (p=20)'], 
                [baseline_avg_inv, high_penalty_avg_inv], 
                color=['blue', 'red'])
        plt.ylabel('Average Inventory Level')
        plt.title('Impact of Shortage Penalty on Average Inventory Level')
        plt.grid(axis='y')
        
        self.save_plot(
            "Penalty Cost Inventory Comparison", 
            "Shows how the average inventory level changes based on the shortage penalty cost. Higher penalties lead to maintaining larger inventories to avoid stockouts.",
            experiment_details
        )
        
        penalty_stats = f"Average inventory level (Low Penalty): {baseline_avg_inv:.2f}\n"
        penalty_stats += f"Average inventory level (High Penalty): {high_penalty_avg_inv:.2f}\n"
        penalty_stats += f"Inventory level increase: {high_penalty_avg_inv - baseline_avg_inv:.2f} ({(high_penalty_avg_inv/baseline_avg_inv - 1)*100:.2f}%)\n"
        print(penalty_stats)
        
        return high_penalty_dp, penalty_stats
    
    def run_deterministic_stochastic_experiment(self, baseline_dp):
        """Run deterministic vs stochastic comparison experiment and save plots"""
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
        
        experiment_details = """
Comparison between:
1. Stochastic model: Poisson demand with mean 20
2. Deterministic model: Fixed demand of 20 units each week

Both models use:
- Planning horizon: 52 weeks
- Maximum warehouse capacity: 100 units
- Maximum order size: 50 units
- Fixed ordering cost (K): 10
- Per-unit ordering cost (c): 2
- Per-unit holding cost (h): 1
- Per-unit shortage penalty (p): 5
"""
        
        # Solve the DP
        deterministic_dp.solve()
        
        # Visualization 7: Compare Policies
        deterministic_dp.compare_policy_heatmaps(
            baseline_dp, 
            title="Policy Comparison: Deterministic vs. Stochastic Demand"
        )
        self.save_plot(
            "Deterministic vs Stochastic Policy Comparison", 
            "Compares the optimal ordering policies between deterministic and stochastic demand models. The stochastic model typically maintains higher inventory levels to hedge against uncertainty.",
            experiment_details
        )
        
        # Compare the cost-to-go (value function) at initial state
        stochastic_cost = baseline_dp.cost_to_go[0, baseline_dp.state_to_index(0)]
        deterministic_cost = deterministic_dp.cost_to_go[0, deterministic_dp.state_to_index(0)]
        
        # Create bar chart comparing expected costs
        plt.figure(figsize=(10, 6))
        plt.bar(['Stochastic Model', 'Deterministic Model'], 
                [stochastic_cost, deterministic_cost], 
                color=['blue', 'green'])
        plt.ylabel('Expected Total Cost')
        plt.title('Expected Cost Comparison: Deterministic vs. Stochastic Models')
        plt.grid(axis='y')
        
        self.save_plot(
            "Deterministic vs Stochastic Cost Comparison", 
            "Compares the expected total cost between deterministic and stochastic demand models. The stochastic model has higher expected costs due to the need to hedge against uncertainty.",
            experiment_details
        )
        
        # Run deterministic simulation with stochastic demand to show suboptimality
        # Use deterministic policy in stochastic environment (wrong model)
        current_state = 0
        state_traj_det_policy = [current_state]
        action_traj_det_policy = []
        total_cost_det_policy = 0
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        for k in range(baseline_dp.N):
            # Get action from deterministic policy
            action = deterministic_dp.policy[k, deterministic_dp.state_to_index(current_state)]
            action_traj_det_policy.append(action)
            
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
            total_cost_det_policy += stage_cost
            
            # Update state
            current_state = next_state
            state_traj_det_policy.append(current_state)
        
        # Add terminal cost
        terminal_cost = baseline_dp.h * max(0, current_state) + baseline_dp.p * max(0, -current_state)
        total_cost_det_policy += terminal_cost
        
        # Also run stochastic policy in stochastic environment (correct model)
        stochastic_sim = baseline_dp.simulate(initial_state=0, num_simulations=1, seed=42)
        stochastic_cost_sim = stochastic_sim['total_costs'][0]
        
        # Plot both trajectories for comparison
        plt.figure(figsize=(12, 6))
        weeks = np.arange(baseline_dp.N + 1)
        plt.plot(weeks, state_traj_det_policy, 'g-o', linewidth=2, label='Deterministic Policy')
        plt.plot(weeks, stochastic_sim['state_trajectories'][0], 'b-^', linewidth=2, label='Stochastic Policy')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Week')
        plt.ylabel('Inventory Level')
        plt.title('Inventory Trajectories: Deterministic vs. Stochastic Policies')
        plt.legend()
        plt.grid(True)
        
        self.save_plot(
            "Deterministic vs Stochastic Policy Trajectories", 
            "Compares inventory trajectories when using deterministic vs. stochastic policies in a stochastic environment. The deterministic policy tends to maintain lower inventory levels, leading to more stockouts and higher actual costs.",
            experiment_details
        )
        
        det_stoch_stats = f"Expected cost (Stochastic): {stochastic_cost:.2f}\n"
        det_stoch_stats += f"Expected cost (Deterministic): {deterministic_cost:.2f}\n"
        det_stoch_stats += f"Cost of uncertainty: {stochastic_cost - deterministic_cost:.2f} ({(stochastic_cost/deterministic_cost - 1)*100:.2f}%)\n"
        det_stoch_stats += f"Total cost (Deterministic policy in stochastic environment): {total_cost_det_policy:.2f}\n"
        det_stoch_stats += f"Total cost (Stochastic policy in stochastic environment): {stochastic_cost_sim:.2f}\n"
        det_stoch_stats += f"Cost reduction from using correct model: {total_cost_det_policy - stochastic_cost_sim:.2f} ({(total_cost_det_policy/stochastic_cost_sim - 1)*100:.2f}%)\n"
        print(det_stoch_stats)
        
        return deterministic_dp, det_stoch_stats
    
    def run_all_experiments(self):
        """Run all experiments and create a dashboard"""
        # Run baseline scenario
        baseline_dp, baseline_sim, baseline_stats = self.run_baseline_experiment()
        
        # Run comparison experiments
        high_vol_dp, volatility_stats = self.run_demand_volatility_experiment(baseline_dp)
        high_penalty_dp, penalty_stats = self.run_penalty_costs_experiment(baseline_dp)
        deterministic_dp, det_stoch_stats = self.run_deterministic_stochastic_experiment(baseline_dp)
        
        # Compile summary for dashboard
        summary = f"""
## Experiment Results Summary

### Baseline Scenario
{baseline_stats}

### Impact of Demand Volatility
{volatility_stats}

### Effect of Penalty Costs
{penalty_stats}

### Deterministic vs. Stochastic Modeling
{det_stoch_stats}

## Key Insights

1. **Policy Structure**: The optimal policy typically resembles an (s,S) policy, where orders are placed when inventory falls below a certain threshold s, bringing it up to level S.

2. **Impact of Uncertainty**: Higher demand volatility leads to higher safety stocks and increased total costs. The stochastic model maintains higher inventory levels compared to a deterministic model to hedge against uncertainty.

3. **Cost Parameters**: Higher shortage penalties result in higher average inventory levels, demonstrating the tradeoff between holding costs and shortage penalties.

4. **End-of-Horizon Effects**: The optimal policy changes as the horizon approaches, typically ordering less in later stages to avoid excess inventory at the end.

5. **Value of Stochastic Modeling**: Using a deterministic model in a stochastic environment leads to suboptimal performance. The project quantifies this cost of model misspecification.
"""
        
        # Create dashboard
        dashboard_path = self.create_dashboard("Dynamic Programming for Optimal Inventory Control", summary)
        
        print("\nAll experiments completed successfully.")
        print(f"Results saved in the '{self.run_dir}' directory.")
        print(f"Dashboard available at: {dashboard_path}")
        
        return dashboard_path

if __name__ == "__main__":
    # Run all experiments and create dashboard
    dashboard = ExperimentDashboard()
    dashboard.run_all_experiments() 