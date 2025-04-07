import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import time

class InventoryDP:
    """Dynamic Programming solution for optimal stochastic inventory control"""
    
    def __init__(self, N=52, M=100, A_max=50, K=10, c=2, h=1, p=5, demand_type='poisson', 
                 demand_param=20, B=100, D_max=None):
        """
        Initialize the inventory problem parameters
        
        Parameters:
        -----------
        N : int
            Number of weeks (stages) in the planning horizon
        M : int
            Maximum warehouse capacity
        A_max : int
            Maximum order size
        K : float
            Fixed ordering cost
        c : float
            Per-unit ordering cost
        h : float
            Per-unit holding cost
        p : float
            Per-unit shortage/backlog penalty
        demand_type : str
            Type of demand distribution ('poisson', 'uniform', 'deterministic')
        demand_param : float or tuple
            Parameter for the demand distribution
            - For 'poisson': lambda parameter (mean)
            - For 'uniform': tuple (min, max) for discrete uniform
            - For 'deterministic': fixed demand value
        B : int
            Maximum backlog allowed (absolute value)
        D_max : int
            Maximum possible demand (inferred from distribution if None)
        """
        self.N = N
        self.M = M
        self.A_max = A_max
        self.K = K
        self.c = c
        self.h = h
        self.p = p
        self.demand_type = demand_type
        self.demand_param = demand_param
        self.B = B
        
        # Set up state space
        self.state_min = -B
        self.state_max = M
        self.state_size = M + B + 1
        
        # Set up demand distribution
        if D_max is None:
            if demand_type == 'poisson':
                # Set D_max to a high percentile of the Poisson distribution
                self.D_max = poisson.ppf(0.9999, demand_param).astype(int)
            elif demand_type == 'uniform':
                self.D_max = demand_param[1]
            elif demand_type == 'deterministic':
                self.D_max = demand_param
            else:
                raise ValueError(f"Unsupported demand_type: {demand_type}")
        else:
            self.D_max = D_max
            
        self.demand_probs = self._compute_demand_probabilities()
        
        # Initialize data structures for DP
        self.cost_to_go = np.full((N+1, self.state_size), np.inf)
        self.policy = np.full((N, self.state_size), -1, dtype=int)
        
    def _compute_demand_probabilities(self):
        """Compute probability mass function for demand"""
        demands = np.arange(self.D_max + 1)
        
        if self.demand_type == 'poisson':
            probs = poisson.pmf(demands, self.demand_param)
            # Normalize to ensure the sum is 1
            return probs / np.sum(probs)
        
        elif self.demand_type == 'uniform':
            min_demand, max_demand = self.demand_param
            probs = np.zeros(self.D_max + 1)
            probs[min_demand:(max_demand+1)] = 1.0 / (max_demand - min_demand + 1)
            return probs
        
        elif self.demand_type == 'deterministic':
            probs = np.zeros(self.D_max + 1)
            probs[self.demand_param] = 1.0
            return probs
        
        else:
            raise ValueError(f"Unsupported demand_type: {self.demand_type}")
    
    def state_to_index(self, state):
        """Convert state value to array index"""
        return state - self.state_min
    
    def index_to_state(self, index):
        """Convert array index to state value"""
        return index + self.state_min
    
    def calculate_terminal_cost(self):
        """Calculate the terminal cost for all possible end states"""
        for idx in range(self.state_size):
            state = self.index_to_state(idx)
            # Terminal cost is holding cost for positive inventory
            # or penalty cost for backlog
            self.cost_to_go[self.N, idx] = self.h * max(0, state) + self.p * max(0, -state)
    
    def get_feasible_actions(self, state):
        """Get the set of feasible actions for a given state"""
        # Can't order more than A_max
        # Can't exceed capacity M
        max_order = min(self.A_max, self.M - state)
        if max_order < 0:
            # If state > M (shouldn't happen), return empty list
            return []
        return range(max_order + 1)
    
    def solve(self):
        """Solve the DP problem with backward recursion"""
        start_time = time.time()
        
        # Calculate terminal costs
        self.calculate_terminal_cost()
        
        # Backward recursion
        for k in range(self.N - 1, -1, -1):
            print(f"Solving stage {k}...")
            
            for idx in range(self.state_size):
                state = self.index_to_state(idx)
                min_cost = np.inf
                best_action = -1
                
                # For each feasible action
                for action in self.get_feasible_actions(state):
                    # Calculate ordering cost
                    order_cost = self.K * (action > 0) + self.c * action
                    
                    # Calculate expected future cost
                    expected_future_cost = 0
                    for demand in range(self.D_max + 1):
                        # Next state after demand realization
                        next_state = state + action - demand
                        
                        # Ensure next_state is within bounds
                        if next_state < self.state_min:
                            # If below min, apply high penalty and clamp
                            penalty = self.p * (self.state_min - next_state)
                            next_state = self.state_min
                        elif next_state > self.state_max:
                            # If above max, apply high penalty and clamp
                            penalty = self.h * (next_state - self.state_max)
                            next_state = self.state_max
                        else:
                            penalty = 0
                        
                        # Calculate holding/shortage cost
                        holding_cost = self.h * max(0, next_state)
                        shortage_cost = self.p * max(0, -next_state)
                        
                        # Get future optimal cost
                        future_cost = self.cost_to_go[k+1, self.state_to_index(next_state)]
                        
                        # Add to expected cost, weighted by demand probability
                        stage_cost = holding_cost + shortage_cost + penalty
                        expected_future_cost += self.demand_probs[demand] * (stage_cost + future_cost)
                    
                    # Total cost for this action
                    total_cost = order_cost + expected_future_cost
                    
                    # Update if this action has lower cost
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_action = action
                
                # Store optimal cost and action
                self.cost_to_go[k, idx] = min_cost
                self.policy[k, idx] = best_action
        
        elapsed_time = time.time() - start_time
        print(f"DP solution completed in {elapsed_time:.2f} seconds")
        
    def simulate(self, initial_state=0, num_simulations=1, seed=None):
        """
        Simulate the system using the optimal policy
        
        Parameters:
        -----------
        initial_state : int
            Starting inventory level
        num_simulations : int
            Number of simulation runs
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        all_results = {
            'state_trajectories': [],
            'action_trajectories': [],
            'demand_trajectories': [],
            'cost_trajectories': [],
            'total_costs': []
        }
        
        for sim in range(num_simulations):
            # Initialize simulation
            current_state = initial_state
            total_cost = 0
            
            state_trajectory = [current_state]
            action_trajectory = []
            demand_trajectory = []
            cost_trajectory = []
            
            for k in range(self.N):
                # Get optimal action from policy
                state_idx = self.state_to_index(current_state)
                action = self.policy[k, state_idx]
                
                # Generate random demand for this week
                if self.demand_type == 'poisson':
                    demand = np.random.poisson(self.demand_param)
                    # Cap demand at D_max if necessary
                    demand = min(demand, self.D_max)
                elif self.demand_type == 'uniform':
                    min_demand, max_demand = self.demand_param
                    demand = np.random.randint(min_demand, max_demand + 1)
                elif self.demand_type == 'deterministic':
                    demand = self.demand_param
                
                # Calculate costs
                order_cost = self.K * (action > 0) + self.c * action
                
                # Update state
                next_state = current_state + action - demand
                
                # Constrain next_state to valid range if needed
                if next_state < self.state_min:
                    next_state = self.state_min
                elif next_state > self.state_max:
                    next_state = self.state_max
                
                # Calculate holding/shortage cost
                holding_cost = self.h * max(0, next_state)
                shortage_cost = self.p * max(0, -next_state)
                
                # Update total cost
                stage_cost = order_cost + holding_cost + shortage_cost
                total_cost += stage_cost
                
                # Update state
                current_state = next_state
                
                # Record trajectory
                action_trajectory.append(action)
                demand_trajectory.append(demand)
                state_trajectory.append(current_state)
                cost_trajectory.append(stage_cost)
            
            # Add terminal cost (if any)
            terminal_cost = self.h * max(0, current_state) + self.p * max(0, -current_state)
            total_cost += terminal_cost
            
            # Store results for this simulation
            all_results['state_trajectories'].append(state_trajectory)
            all_results['action_trajectories'].append(action_trajectory)
            all_results['demand_trajectories'].append(demand_trajectory)
            all_results['cost_trajectories'].append(cost_trajectory)
            all_results['total_costs'].append(total_cost)
        
        return all_results
    
    def plot_policy_heatmap(self, weeks=None, figsize=(10, 6), title_suffix=""):
        """
        Plot the optimal policy as a heatmap for selected weeks
        
        Parameters:
        -----------
        weeks : list
            List of weeks to plot. If None, selects a few representative weeks.
        figsize : tuple
            Figure size
        title_suffix : str
            Additional text to add to the title
        """
        if weeks is None:
            # Choose a few representative weeks
            weeks = [0, self.N // 4, self.N // 2, 3 * self.N // 4, self.N - 1]
        
        state_values = np.array([self.index_to_state(i) for i in range(self.state_size)])
        
        plt.figure(figsize=figsize)
        
        for i, week in enumerate(weeks):
            plt.subplot(1, len(weeks), i + 1)
            
            # Get policy for this week
            policy_week = self.policy[week]
            
            # Plot as a function of state
            plt.plot(state_values, policy_week, marker='o', linestyle='-', linewidth=2)
            
            plt.xlabel('Inventory Level')
            plt.ylabel('Optimal Order Quantity')
            plt.title(f'Week {week}')
            plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'Optimal Policy Across Planning Horizon{title_suffix}', fontsize=16)
        plt.subplots_adjust(top=0.85)
        
    def plot_value_function(self, weeks=None, figsize=(10, 6), title_suffix=""):
        """
        Plot the value function for selected weeks
        
        Parameters:
        -----------
        weeks : list
            List of weeks to plot. If None, selects a few representative weeks.
        figsize : tuple
            Figure size
        title_suffix : str
            Additional text to add to the title
        """
        if weeks is None:
            # Choose a few representative weeks
            weeks = [0, self.N // 4, self.N // 2, 3 * self.N // 4, self.N - 1]
        
        state_values = np.array([self.index_to_state(i) for i in range(self.state_size)])
        
        plt.figure(figsize=figsize)
        
        for i, week in enumerate(weeks):
            plt.subplot(1, len(weeks), i + 1)
            
            # Get cost-to-go for this week
            cost_week = self.cost_to_go[week]
            
            # Plot as a function of state
            plt.plot(state_values, cost_week, marker='.', linestyle='-', linewidth=2)
            
            plt.xlabel('Inventory Level')
            plt.ylabel('Expected Future Cost')
            plt.title(f'Week {week}')
            plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'Value Function Across Planning Horizon{title_suffix}', fontsize=16)
        plt.subplots_adjust(top=0.85)
    
    def plot_simulation_trajectory(self, sim_results, sim_index=0, figsize=(12, 8), title_suffix=""):
        """
        Plot a single simulation trajectory
        
        Parameters:
        -----------
        sim_results : dict
            Simulation results from simulate()
        sim_index : int
            Index of the simulation to plot
        figsize : tuple
            Figure size
        title_suffix : str
            Additional text to add to the title
        """
        state_traj = sim_results['state_trajectories'][sim_index]
        action_traj = sim_results['action_trajectories'][sim_index]
        demand_traj = sim_results['demand_trajectories'][sim_index]
        
        weeks = np.arange(self.N + 1)
        weeks_actions = np.arange(self.N)
        
        plt.figure(figsize=figsize)
        
        # Plot inventory levels
        plt.subplot(3, 1, 1)
        plt.plot(weeks, state_traj, 'b-o', linewidth=2, label='Inventory Level')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Week')
        plt.ylabel('Inventory Level')
        plt.title('Inventory Level Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot orders
        plt.subplot(3, 1, 2)
        plt.bar(weeks_actions, action_traj, color='g')
        plt.xlabel('Week')
        plt.ylabel('Order Quantity')
        plt.title('Order Quantities Over Time')
        plt.grid(True)
        
        # Plot demand
        plt.subplot(3, 1, 3)
        plt.bar(weeks_actions, demand_traj, color='r')
        plt.xlabel('Week')
        plt.ylabel('Demand')
        plt.title('Demand Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'Simulation Trajectory{title_suffix}', fontsize=16)
        plt.subplots_adjust(top=0.9)
    
    def compare_policy_heatmaps(self, other_dp, figsize=(14, 10), title="Policy Comparison"):
        """
        Compare policies between this DP instance and another one
        
        Parameters:
        -----------
        other_dp : InventoryDP
            Another solved DP instance to compare with
        figsize : tuple
            Figure size
        title : str
            Plot title
        """
        # Choose a few representative weeks
        weeks = [0, self.N // 4, self.N // 2, 3 * self.N // 4, self.N - 1]
        
        state_values = np.array([self.index_to_state(i) for i in range(self.state_size)])
        
        plt.figure(figsize=figsize)
        
        for i, week in enumerate(weeks):
            plt.subplot(2, len(weeks), i + 1)
            
            # Get policy for this week (this DP)
            policy_week = self.policy[week]
            
            # Plot as a function of state
            plt.plot(state_values, policy_week, marker='o', linestyle='-', linewidth=2)
            
            plt.xlabel('Inventory Level')
            plt.ylabel('Optimal Order Quantity')
            plt.title(f'Week {week} - Scenario 1')
            plt.grid(True)
            
            plt.subplot(2, len(weeks), i + 1 + len(weeks))
            
            # Get policy for this week (other DP)
            other_policy_week = other_dp.policy[week]
            other_state_values = np.array([other_dp.index_to_state(i) for i in range(other_dp.state_size)])
            
            # Plot as a function of state
            plt.plot(other_state_values, other_policy_week, marker='o', linestyle='-', linewidth=2, color='orange')
            
            plt.xlabel('Inventory Level')
            plt.ylabel('Optimal Order Quantity')
            plt.title(f'Week {week} - Scenario 2')
            plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9) 