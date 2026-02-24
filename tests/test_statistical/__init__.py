"""
Test sub-package for the Statistical Models module (F-006).

Tests verify that statistical distribution models produce outputs matching
expected parameters using confidence intervals with sufficient sample sizes.

Modules:
- test_payment_timing_model: 5-segment mixture model for customer payment timing
- test_amount_distributions: Log-normal amount distributions with rounding and discounts
- test_order_frequency_model: Poisson process with day-of-week effects
- test_selection_models: Pareto vendor, revenue-weighted customer, repeat/new product selection

Testing Conventions:
- Uses numpy seed=42 for reproducible statistical tests
- Statistical assertions use confidence intervals (N>=10,000 samples)
- No external API calls or Redis dependencies
"""
