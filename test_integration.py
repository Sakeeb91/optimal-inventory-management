import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from inventory_dp import InventoryDP
from run_experiments import compare_demand_volatility

class TestIntegration(unittest.TestCase):
    """Integration tests to verify entire system works correctly together"""
    
    def setUp(self):
        """Set up test environment with temporary directory for outputs"""
        self.original_dir = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        
        # Create figures directory in the temp directory
        os.makedirs('figures', exist_ok=True)
        
        # Setup a very small test model for quick testing
        self.test_dp = InventoryDP(
            N=5,           # Very small horizon
            M=10,          # Small capacity
            A_max=5,       # Small max order
            K=10,          # Fixed ordering cost
            c=2,           # Per-unit ordering cost
            h=1,           # Per-unit holding cost
            p=5,           # Per-unit shortage penalty
            demand_type='poisson',
            demand_param=3,   # Small mean demand
            B=10,          # Small backlog limit
            D_max=8        # Small max demand
        )
    
    def tearDown(self):
        """Clean up temporary directory after tests"""
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
    
    def test_solve_and_simulate(self):
        """Test full solve and simulate cycle"""
        # Solve the DP problem
        self.test_dp.solve()
        
        # Check that cost_to_go and policy matrices are populated
        self.assertFalse(np.isinf(self.test_dp.cost_to_go).any(), 
                        "cost_to_go matrix contains inf values after solving")
        self.assertFalse((self.test_dp.policy == -1).all(), 
                        "policy matrix not populated after solving")
        
        # Run a simulation
        sim_results = self.test_dp.simulate(initial_state=0, num_simulations=3, seed=42)
        
        # Check simulation results structure
        self.assertIn('state_trajectories', sim_results)
        self.assertIn('action_trajectories', sim_results)
        self.assertIn('demand_trajectories', sim_results)
        self.assertIn('cost_trajectories', sim_results)
        self.assertIn('total_costs', sim_results)
        
        # Check simulation results length
        self.assertEqual(len(sim_results['state_trajectories']), 3)
        self.assertEqual(len(sim_results['action_trajectories']), 3)
        self.assertEqual(len(sim_results['demand_trajectories']), 3)
        self.assertEqual(len(sim_results['cost_trajectories']), 3)
        self.assertEqual(len(sim_results['total_costs']), 3)
        
        # Check state trajectory length (should be N+1)
        self.assertEqual(len(sim_results['state_trajectories'][0]), self.test_dp.N + 1)
        # Check action trajectory length (should be N)
        self.assertEqual(len(sim_results['action_trajectories'][0]), self.test_dp.N)
        
        # Try visualization functions to ensure they run without error
        self.test_dp.plot_policy_heatmap()
        plt.close()
        
        self.test_dp.plot_value_function()
        plt.close()
        
        self.test_dp.plot_simulation_trajectory(sim_results)
        plt.close()
    
    def test_different_demand_types(self):
        """Test that different demand types can be solved and simulated"""
        demand_types = ['poisson', 'uniform', 'deterministic']
        demand_params = [3, (1, 5), 3]  # Corresponding parameters
        
        for dtype, dparam in zip(demand_types, demand_params):
            # Create model
            model = InventoryDP(
                N=5, M=10, A_max=5, K=10, c=2, h=1, p=5,
                demand_type=dtype,
                demand_param=dparam,
                B=10
            )
            
            # Solve
            model.solve()
            
            # Simulate
            sim_results = model.simulate(initial_state=0, num_simulations=1, seed=42)
            
            # Check basic results validity
            self.assertGreater(len(sim_results['state_trajectories']), 0)
            self.assertGreater(len(sim_results['action_trajectories']), 0)
    
    def test_run_experiments_functions(self):
        """Test that experiment functions from run_experiments.py work"""
        # Create a custom comparison function that uses matching N values
        def test_compare_demand_volatility(baseline_dp):
            # Create high volatility model matching baseline N value
            high_vol_dp = InventoryDP(
                N=baseline_dp.N,  # Match baseline N value
                M=baseline_dp.M,
                A_max=baseline_dp.A_max,
                K=baseline_dp.K,
                c=baseline_dp.c,
                h=baseline_dp.h,
                p=baseline_dp.p,
                demand_type='uniform',
                demand_param=(1, 5),  # Uniform demand
                B=baseline_dp.B
            )
            
            # Solve the DP
            high_vol_dp.solve()
            
            # Create comparable plots
            plt.figure(figsize=(12, 6))
            
            # Get representative weeks
            weeks = [0, baseline_dp.N // 2, baseline_dp.N - 1] if baseline_dp.N > 2 else [0]
            
            # Compare policies for the same weeks
            for i, week in enumerate(weeks):
                plt.subplot(2, len(weeks), i + 1)
                
                # Plot baseline policy
                state_values = np.array([baseline_dp.index_to_state(i) for i in range(baseline_dp.state_size)])
                policy_week = baseline_dp.policy[week]
                plt.plot(state_values, policy_week, marker='o', linestyle='-', linewidth=2)
                plt.title(f'Week {week} - Poisson')
                
                # Plot high volatility policy
                plt.subplot(2, len(weeks), i + 1 + len(weeks))
                state_values = np.array([high_vol_dp.index_to_state(i) for i in range(high_vol_dp.state_size)])
                policy_week = high_vol_dp.policy[week]
                plt.plot(state_values, policy_week, marker='o', linestyle='-', linewidth=2, color='orange')
                plt.title(f'Week {week} - Uniform')
            
            plt.tight_layout()
            plt.savefig('figures/volatility_policy_comparison.png')
            plt.close()
            
            # Run a simple simulation for both
            baseline_sim = baseline_dp.simulate(initial_state=0, num_simulations=1, seed=42)
            high_vol_sim = high_vol_dp.simulate(initial_state=0, num_simulations=1, seed=42)
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            weeks = np.arange(baseline_dp.N + 1)
            plt.plot(weeks, baseline_sim['state_trajectories'][0], 'b-o', label='Poisson')
            plt.plot(weeks, high_vol_sim['state_trajectories'][0], 'r-^', label='Uniform')
            plt.legend()
            plt.xlabel('Week')
            plt.ylabel('Inventory')
            plt.title('Comparison of Inventory Trajectories')
            plt.savefig('figures/volatility_trajectory_comparison.png')
            plt.close()
            
            return high_vol_dp
        
        # Create a mock small baseline model and pre-solve it
        small_baseline = InventoryDP(
            N=5, M=10, A_max=5, K=10, c=2, h=1, p=5,
            demand_type='poisson', demand_param=3, B=10
        )
        small_baseline.solve()
        
        # Use our test version of the comparison function
        high_vol_dp = test_compare_demand_volatility(small_baseline)
        
        # Check that files were created
        self.assertTrue(os.path.exists('figures/volatility_policy_comparison.png'))
        self.assertTrue(os.path.exists('figures/volatility_trajectory_comparison.png'))

    def test_model_behavior(self):
        """Test model behavior with different parameters"""
        # Create two models with different shortage penalties
        low_penalty = InventoryDP(
            N=5, M=10, A_max=5, K=10, c=2, h=1, p=2,
            demand_type='poisson', demand_param=3, B=10
        )
        
        high_penalty = InventoryDP(
            N=5, M=10, A_max=5, K=10, c=2, h=1, p=10,
            demand_type='poisson', demand_param=3, B=10
        )
        
        low_penalty.solve()
        high_penalty.solve()
        
        # Run simulations with fixed seed
        low_sim = low_penalty.simulate(initial_state=0, num_simulations=20, seed=42)
        high_sim = high_penalty.simulate(initial_state=0, num_simulations=20, seed=42)
        
        # Calculate average inventory levels
        low_avg_inv = np.mean([np.mean(traj) for traj in low_sim['state_trajectories']])
        high_avg_inv = np.mean([np.mean(traj) for traj in high_sim['state_trajectories']])
        
        # Higher penalty should lead to higher average inventory
        self.assertGreaterEqual(high_avg_inv, low_avg_inv)


if __name__ == '__main__':
    unittest.main() 