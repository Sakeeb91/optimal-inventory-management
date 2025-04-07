import unittest
import numpy as np
from inventory_dp import InventoryDP

class TestInventoryDP(unittest.TestCase):
    
    def setUp(self):
        """Set up a small test instance of InventoryDP"""
        # Create a small instance for faster testing
        self.test_dp = InventoryDP(
            N=10,          # 10 weeks for faster tests
            M=20,          # Small capacity
            A_max=10,      # Small max order
            K=10,          # Fixed ordering cost
            c=2,           # Per-unit ordering cost
            h=1,           # Per-unit holding cost
            p=5,           # Per-unit shortage penalty
            demand_type='poisson',
            demand_param=5,  # Lower demand for simpler tests
            B=20,          # Smaller backlog size
            D_max=15       # Set explicit D_max for testing
        )
        
        # Also set up a deterministic instance for predictable results
        self.deterministic_dp = InventoryDP(
            N=10,
            M=20,
            A_max=10,
            K=10,
            c=2,
            h=1,
            p=5,
            demand_type='deterministic',
            demand_param=5,  # Fixed demand of 5 units
            B=20,
            D_max=5
        )
        
    def test_initialization(self):
        """Test that initialization sets parameters correctly"""
        # Check parameter setting
        self.assertEqual(self.test_dp.N, 10)
        self.assertEqual(self.test_dp.M, 20)
        self.assertEqual(self.test_dp.A_max, 10)
        self.assertEqual(self.test_dp.K, 10)
        self.assertEqual(self.test_dp.c, 2)
        self.assertEqual(self.test_dp.h, 1)
        self.assertEqual(self.test_dp.p, 5)
        self.assertEqual(self.test_dp.demand_type, 'poisson')
        self.assertEqual(self.test_dp.demand_param, 5)
        self.assertEqual(self.test_dp.D_max, 15)
        
        # Check data structure initialization
        self.assertEqual(self.test_dp.cost_to_go.shape, (11, 41))  # (N+1, state_size)
        self.assertEqual(self.test_dp.policy.shape, (10, 41))      # (N, state_size)
        
        # Check state bounds
        self.assertEqual(self.test_dp.state_min, -20)
        self.assertEqual(self.test_dp.state_max, 20)
        self.assertEqual(self.test_dp.state_size, 41)  # From -20 to 20 inclusive
    
    def test_demand_probabilities(self):
        """Test demand probability calculation"""
        # Check that probabilities sum to 1
        self.assertAlmostEqual(sum(self.test_dp.demand_probs), 1.0, places=10)
        
        # For deterministic case, probability should be 1 at param value and 0 elsewhere
        self.assertEqual(self.deterministic_dp.demand_probs[5], 1.0)
        self.assertEqual(sum(self.deterministic_dp.demand_probs), 1.0)
        
        # Create a uniform distribution instance for testing
        uniform_dp = InventoryDP(
            N=10, M=20, A_max=10, K=10, c=2, h=1, p=5,
            demand_type='uniform',
            demand_param=(3, 7),  # Uniform between 3 and 7
            D_max=10
        )
        
        # Check uniform distribution has equal probabilities
        prob_per_value = 1.0 / 5  # 5 values (3, 4, 5, 6, 7)
        for i in range(3, 8):
            self.assertAlmostEqual(uniform_dp.demand_probs[i], prob_per_value, places=10)
        
        # Check rest are zero
        for i in range(3):
            self.assertEqual(uniform_dp.demand_probs[i], 0.0)
        for i in range(8, 11):
            self.assertEqual(uniform_dp.demand_probs[i], 0.0)
    
    def test_state_index_conversion(self):
        """Test conversion between state values and array indices"""
        test_states = [-20, -10, 0, 10, 20]  # Sample states
        for state in test_states:
            index = self.test_dp.state_to_index(state)
            self.assertEqual(self.test_dp.index_to_state(index), state)
        
        # Boundary checks
        self.assertEqual(self.test_dp.state_to_index(self.test_dp.state_min), 0)
        self.assertEqual(self.test_dp.state_to_index(self.test_dp.state_max), self.test_dp.state_size - 1)
    
    def test_feasible_actions(self):
        """Test feasible action determination"""
        # For state = 15, with M = 20, A_max = 10, should return 0 to 5
        actions = list(self.test_dp.get_feasible_actions(15))
        self.assertEqual(actions, [0, 1, 2, 3, 4, 5])
        
        # For state = -5, with M = 20, A_max = 10, should return 0 to 10
        actions = list(self.test_dp.get_feasible_actions(-5))
        self.assertEqual(actions, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # For state = 20, should return only 0 (already at capacity)
        actions = list(self.test_dp.get_feasible_actions(20))
        self.assertEqual(actions, [0])
        
        # For state > M, should return empty list
        actions = list(self.test_dp.get_feasible_actions(21))
        self.assertEqual(actions, [])
    
    def test_terminal_cost(self):
        """Test terminal cost calculation"""
        self.test_dp.calculate_terminal_cost()
        
        # Check a few sample terminal costs
        # At state 10 (positive): should be h * 10 = 1 * 10 = 10
        self.assertEqual(self.test_dp.cost_to_go[10, self.test_dp.state_to_index(10)], 10)
        
        # At state -10 (negative): should be p * 10 = 5 * 10 = 50
        self.assertEqual(self.test_dp.cost_to_go[10, self.test_dp.state_to_index(-10)], 50)
        
        # At state 0: should be 0
        self.assertEqual(self.test_dp.cost_to_go[10, self.test_dp.state_to_index(0)], 0)
    
    def test_solve_deterministic(self):
        """Test solving a deterministic instance"""
        # For deterministic case, solutions should be exact
        self.deterministic_dp.solve()
        
        # Get expected cost at state 0, stage 0
        cost_at_zero = self.deterministic_dp.cost_to_go[0, self.deterministic_dp.state_to_index(0)]
        
        # This is a deterministic problem. Let's manually calculate one cost
        # For the deterministic case with demand=5 each period:
        # - If we're at state 0, we need to order to prevent backlog
        # - With high fixed cost (K=10), we should order in batches
        # - Optimal policy should order some positive quantity
        policy_at_zero = self.deterministic_dp.policy[0, self.deterministic_dp.state_to_index(0)]
        self.assertGreater(policy_at_zero, 0)  # Should order something
    
    def test_simulation_consistency(self):
        """Test that simulation with same seed gives consistent results"""
        # Solve the deterministic instance
        self.deterministic_dp.solve()
        
        # Run two simulations with the same seed
        sim1 = self.deterministic_dp.simulate(initial_state=0, num_simulations=3, seed=42)
        sim2 = self.deterministic_dp.simulate(initial_state=0, num_simulations=3, seed=42)
        
        # Results should be identical
        for key in sim1:
            for i in range(len(sim1[key])):
                np.testing.assert_array_equal(sim1[key][i], sim2[key][i])
        
        # Run another simulation with different seed
        sim3 = self.deterministic_dp.simulate(initial_state=0, num_simulations=3, seed=43)
        
        # Check deterministic case always follows the same trajectory
        # (since demand is fixed, even different seeds should give same results)
        for key in sim1:
            for i in range(len(sim1[key])):
                np.testing.assert_array_equal(sim1[key][i], sim3[key][i])
    
    def test_s_S_policy_structure(self):
        """Test if the optimal policy follows an s-S structure for certain cases"""
        # Create a simple instance likely to have a clear s-S structure
        dp = InventoryDP(
            N=10, M=20, A_max=10, K=20,  # High fixed cost favors s-S
            c=1, h=1, p=10,  # High penalty compared to holding
            demand_type='deterministic',
            demand_param=3,
            D_max=3
        )
        
        dp.solve()
        
        # Check a few stages in the beginning (end stages might have end-effects)
        for stage in [0, 1, 2]:
            # Get policy for this stage
            policy = dp.policy[stage]
            
            # Find all states where ordering occurs
            ordering_states = [dp.index_to_state(i) for i in range(len(policy)) if policy[i] > 0]
            
            if ordering_states:  # Only if there are states where ordering occurs
                # Get lowest and highest states where ordering happens
                min_order_state = min(ordering_states)
                
                # Get the order-up-to levels
                order_up_to = {s: s + policy[dp.state_to_index(s)] for s in ordering_states}
                
                # For classic s-S, all orders should bring inventory to same level S
                order_up_to_values = list(order_up_to.values())
                
                # If all order-up-to values are the same, it's a perfect s-S
                # But we'll allow for some variation due to boundary effects
                if len(set(order_up_to_values)) <= 2:
                    # Check there's a threshold s below which we always order
                    for s in range(dp.state_min, min_order_state + 1):
                        if dp.state_to_index(s) < len(policy):  # Ensure within bounds
                            self.assertGreater(policy[dp.state_to_index(s)], 0, 
                                            f"Expected ordering at state {s}, stage {stage}")


if __name__ == '__main__':
    unittest.main() 