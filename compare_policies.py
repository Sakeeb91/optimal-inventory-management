import matplotlib.pyplot as plt
import numpy as np

# Policy data from our runs
models = ['Deterministic', 'Uniform', 'Poisson']
reorder_points = [0, 0, 0]  # All models had reorder point = 0
order_up_to_levels = [135, 150, 148]
avg_costs = [1680.00, 4549.60, 1777.80]
cost_ranges = [0, 1990, 90]  # Max - Min cost

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot order-up-to levels
bars1 = ax1.bar(models, order_up_to_levels, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax1.set_ylabel('Order-up-to Level (S)')
ax1.set_title('Comparison of Order-up-to Levels (S) by Demand Model')
ax1.grid(True, linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{height}', ha='center', va='bottom')

# Add safety stock calculation
avg_demand = 135
for i, model in enumerate(models):
    safety_stock = order_up_to_levels[i] - avg_demand
    if safety_stock > 0:
        ax1.text(i, avg_demand + safety_stock/2, f"+{safety_stock}\n({safety_stock/avg_demand:.0%})",
                ha='center', fontsize=10, color='white')

# Add horizontal line for average demand
ax1.axhline(y=avg_demand, color='red', linestyle='--', alpha=0.7)
ax1.text(2.1, avg_demand, f'Average Demand ({avg_demand})', 
         va='center', color='red', fontsize=10)

# Plot costs
x = np.arange(len(models))
width = 0.35

bars2 = ax2.bar(x, avg_costs, width, yerr=cost_ranges, capsize=10,
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_xlabel('Demand Model')
ax2.set_ylabel('Average Total Cost')
ax2.set_title('Comparison of Costs by Demand Model')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.grid(True, linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()

# Save the figure
plt.savefig('output/policy_comparison.png', dpi=300)
print("Comparison visualization saved to 'output/policy_comparison.png'")

# If you want to show the plot (uncomment if needed)
# plt.show() 