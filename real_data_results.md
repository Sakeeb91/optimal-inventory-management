# Inventory Optimization with Real-World Retail Data

## Dataset Overview
- **Product**: 85123A - WHITE HANGING HEART T-LIGHT HOLDER
- **Source**: Online Retail II dataset from UK-based retailer
- **Time Period**: 739 days (2009-12-01 to 2011-12-09)
- **Demand Statistics**:
  - Average Daily Demand: 135.52 units
  - Median Daily Demand: 87.00 units
  - Standard Deviation: 251.47 units
  - Min Daily Demand: 0 units
  - Max Daily Demand: 4015 units
  - Days with Demand: 603 (81.6%)

## Model Configurations
We ran the inventory optimization model with three different demand modeling approaches:

### 1. Deterministic Model
- **Demand Type**: Deterministic
- **Demand Value**: 135 units (average daily demand)
- **Planning Horizon**: 6 weeks
- **Maximum Capacity**: 300 units

### 2. Simplified Uniform Model
- **Demand Type**: Uniform distribution
- **Demand Range**: (17, 408) units (5th to 95th percentile)
- **Planning Horizon**: 6 weeks
- **Maximum Capacity**: 300 units

### 3. Poisson Model  
- **Demand Type**: Poisson distribution
- **Mean Demand**: 135.52 units
- **Planning Horizon**: 6 weeks
- **Maximum Capacity**: 300 units

## Optimization Results

### Deterministic Model Results
- **Average Total Cost**: 1680.00
- **Min Total Cost**: 1680.00
- **Max Total Cost**: 1680.00
- **Recommended Policy**:
  - Reorder Point (s): 0 units
  - Order-up-to Level (S): 135 units
  - Order Quantity: 135 units when inventory reaches 0

### Uniform Model Results
- **Average Total Cost**: 4549.60
- **Min Total Cost**: 3370.00
- **Max Total Cost**: 5360.00
- **Recommended Policy**:
  - Reorder Point (s): 0 units
  - Order-up-to Level (S): 150 units
  - Order Quantity: 150 units when inventory reaches 0

### Poisson Model Results
- **Average Total Cost**: 1777.80
- **Min Total Cost**: 1724.00
- **Max Total Cost**: 1814.00
- **Recommended Policy**:
  - Reorder Point (s): 0 units
  - Order-up-to Level (S): 148 units
  - Order Quantity: 148 units when inventory reaches 0

## Key Insights

1. **Impact of Uncertainty**: 
   - The deterministic model produced the lowest cost (1680.00), but ignores the variability in real-world demand
   - The stochastic models (Uniform and Poisson) recommend higher order-up-to levels to hedge against demand uncertainty
   - The Uniform model produced much higher costs due to its wider demand range

2. **Safety Stock Effect**:
   - Deterministic policy: Order-up-to = 135 (exactly the expected demand)
   - Poisson policy: Order-up-to = 148 (135 + 13 units of safety stock, ~10% buffer)
   - Uniform policy: Order-up-to = 150 (135 + 15 units of safety stock, ~11% buffer)

3. **Policy Structure**:
   - All three models recommend an (s,S) policy with a reorder point of 0
   - This indicates that for this product, it's optimal to place orders when inventory is depleted
   - The zero reorder point might be influenced by the relatively low holding costs compared to ordering costs

4. **Cost Variability**:
   - Deterministic model: No cost variability (always 1680.00)
   - Poisson model: Low variability (range of ~90 units or ~5%)
   - Uniform model: High variability (range of ~2000 units or ~45%)

## Recommendations for This Product

Based on the modeling results:

1. **Implement a (0, 148) Policy**: Order 148 units when inventory reaches 0, as suggested by the Poisson model
2. **Monitor Performance**: Track actual costs and adjust the order-up-to level based on actual demand patterns
3. **Consider Seasonal Patterns**: The current model doesn't account for seasonality; further analysis could reveal optimal seasonal policies
4. **Lead Time Considerations**: The model assumes immediate delivery; consider incorporating lead times in future modeling
5. **Cost Parameter Sensitivity**: Test different holding and shortage costs to understand their impact on the optimal policy 