from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the DP model (will be moved to API layer in production)
from inventory_dp import InventoryDP

app = Flask(__name__)

# Cache for storing model results
model_cache = {}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/parameters', methods=['GET'])
def get_default_parameters():
    """Return default model parameters"""
    params = {
        'N': 52,          # Planning horizon (weeks)
        'M': 100,         # Maximum inventory
        'A_max': 50,      # Maximum order size
        'K': 10,          # Fixed ordering cost
        'c': 2,           # Per-unit ordering cost
        'h': 1,           # Per-unit holding cost
        'p': 5,           # Per-unit shortage penalty
        'demand_type': 'poisson',
        'demand_param': 20,  # Mean demand
        'initial_state': 0   # Initial inventory
    }
    return jsonify(params)

@app.route('/api/solve', methods=['POST'])
def solve_model():
    """Solve the DP model with provided parameters"""
    data = request.json
    
    # Create a unique key for caching
    cache_key = json.dumps(data, sort_keys=True)
    
    # Check if we already have results for these parameters
    if cache_key in model_cache:
        return jsonify(model_cache[cache_key])
    
    # Initialize the DP model with parameters from request
    dp = InventoryDP(
        N=data.get('N', 52),
        M=data.get('M', 100),
        A_max=data.get('A_max', 50),
        K=data.get('K', 10),
        c=data.get('c', 2),
        h=data.get('h', 1),
        p=data.get('p', 5),
        demand_type=data.get('demand_type', 'poisson'),
        demand_param=data.get('demand_param', 20)
    )
    
    # Solve the model
    dp.solve()
    
    # Run a simulation
    initial_state = data.get('initial_state', 0)
    sim_results = dp.simulate(initial_state=initial_state, num_simulations=5)
    
    # Prepare policy data for visualization
    policy_data = []
    for week in range(dp.N):
        for state_idx in range(dp.state_size):
            state = dp.index_to_state(state_idx)
            action = dp.policy[week, state_idx]
            if action >= 0:  # Only include valid actions
                policy_data.append({
                    'week': week,
                    'state': state,
                    'action': action
                })
    
    # Prepare value function data
    value_data = []
    for week in [0, dp.N//4, dp.N//2, 3*dp.N//4, dp.N-1]:  # Selected weeks
        for state_idx in range(dp.state_size):
            state = dp.index_to_state(state_idx)
            value = dp.cost_to_go[week, state_idx]
            if value < float('inf'):  # Only include valid values
                value_data.append({
                    'week': week,
                    'state': state,
                    'value': value
                })
    
    # Prepare simulation data
    sim_data = {
        'states': sim_results['state_trajectories'][0].tolist(),
        'actions': sim_results['action_trajectories'][0].tolist(),
        'demands': sim_results['demand_trajectories'][0].tolist(),
        'weeks': list(range(dp.N + 1)),
        'total_cost': float(sim_results['total_costs'][0]),
        'avg_inventory': float(np.mean(sim_results['state_trajectories'][0]))
    }
    
    # Calculate key metrics
    metrics = {
        'avg_cost': float(np.mean(sim_results['total_costs'])),
        'min_cost': float(np.min(sim_results['total_costs'])),
        'max_cost': float(np.max(sim_results['total_costs'])),
        'avg_inventory': float(np.mean([np.mean(traj) for traj in sim_results['state_trajectories']])),
        'stockout_rate': float(np.mean([np.sum(traj < 0) / len(traj) for traj in sim_results['state_trajectories']]))
    }
    
    # Combine all results
    results = {
        'policy_data': policy_data,
        'value_data': value_data,
        'sim_data': sim_data,
        'metrics': metrics
    }
    
    # Cache the results
    model_cache[cache_key] = results
    
    return jsonify(results)

@app.route('/api/policy_heatmap', methods=['POST'])
def get_policy_heatmap():
    """Generate policy heatmap visualization"""
    data = request.json
    policy_data = data.get('policy_data', [])
    
    # Convert to DataFrame
    df = pd.DataFrame(policy_data)
    
    # Create pivot table for heatmap
    pivot = df.pivot(index='state', columns='week', values='action')
    
    # Create heatmap using Plotly
    fig = px.imshow(
        pivot,
        labels=dict(x="Week", y="Inventory Level", color="Order Quantity"),
        title="Optimal Ordering Policy Heatmap",
        color_continuous_scale="Viridis"
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graphJSON)

@app.route('/api/value_function', methods=['POST'])
def get_value_function():
    """Generate value function visualization"""
    data = request.json
    value_data = data.get('value_data', [])
    
    # Convert to DataFrame
    df = pd.DataFrame(value_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add lines for each week
    for week in df['week'].unique():
        week_data = df[df['week'] == week]
        fig.add_trace(go.Scatter(
            x=week_data['state'],
            y=week_data['value'],
            mode='lines',
            name=f'Week {week}'
        ))
    
    # Layout
    fig.update_layout(
        title="Value Function by Week",
        xaxis_title="Inventory Level",
        yaxis_title="Expected Future Cost",
        legend_title="Week"
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graphJSON)

@app.route('/api/simulation', methods=['POST'])
def get_simulation():
    """Generate simulation visualization"""
    data = request.json
    sim_data = data.get('sim_data', {})
    
    # Create figure with subplots
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=("Inventory Level", "Order Quantity", "Demand"))
    
    # Add inventory trace
    fig.add_trace(
        go.Scatter(x=sim_data.get('weeks', []), y=sim_data.get('states', []),
                   mode='lines+markers', name='Inventory'),
        row=1, col=1
    )
    
    # Add order quantity trace
    fig.add_trace(
        go.Bar(x=sim_data.get('weeks', [])[:-1], y=sim_data.get('actions', []),
               name='Orders'),
        row=2, col=1
    )
    
    # Add demand trace
    fig.add_trace(
        go.Bar(x=sim_data.get('weeks', [])[:-1], y=sim_data.get('demands', []),
               name='Demand'),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Simulation Trajectory",
        height=800,
        showlegend=True
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graphJSON)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 