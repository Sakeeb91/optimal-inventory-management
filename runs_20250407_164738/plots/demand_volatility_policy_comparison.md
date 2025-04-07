# Demand Volatility Policy Comparison

![Demand Volatility Policy Comparison](./demand_volatility_policy_comparison.png)

## Description

Compares the optimal ordering policies between low volatility (Poisson) and high volatility (Uniform) demand. Higher volatility generally leads to higher safety stocks to hedge against uncertainty.

## Experiment Details


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


Generated on: 2025-04-07 16:48:38
