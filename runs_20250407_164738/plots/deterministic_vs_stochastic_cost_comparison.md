# Deterministic vs Stochastic Cost Comparison

![Deterministic vs Stochastic Cost Comparison](./deterministic_vs_stochastic_cost_comparison.png)

## Description

Compares the expected total cost between deterministic and stochastic demand models. The stochastic model has higher expected costs due to the need to hedge against uncertainty.

## Experiment Details


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


Generated on: 2025-04-07 16:49:34
