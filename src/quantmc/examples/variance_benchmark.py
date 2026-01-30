import numpy as np

from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.variance_reduction import (
    variance,
    variance_reduction,
    control_variate_adjustment
)
from quantmc.analytics.statistics import standard_error


model = GBM(
    initial_price=100,
    risk_free_rate=0.05,
    volatility=0.2
)

payoff = EuropeanCall(
    strike=100
)

pricer = MonteCarloPricer(
    model=model,
    payoff=payoff,
    risk_free_rate=0.05
)


paths = 50000
time = 1

discount_factor = np.exp(
    -pricer.r * time
)


standard_payoffs = pricer.simulate_payoffs(
    time=time,
    steps=100,
    paths=paths,
    seed=42,
    antithetic=False
)


antithetic_payoffs = pricer.simulate_payoffs(
    time=time,
    steps=100,
    paths=paths,
    seed=42,
    antithetic=True
)


half = len(antithetic_payoffs) // 2

antithetic_estimates = (
    antithetic_payoffs[:half]
    +
    antithetic_payoffs[half:]
) / 2


standard_discounted = (
    standard_payoffs
    *
    discount_factor
)


antithetic_discounted = (
    antithetic_estimates
    *
    discount_factor
)


control_data = pricer.simulate_payoffs_and_prices(
    time=time,
    steps=100,
    paths=paths,
    seed=42,
    antithetic=False
)


control_payoffs = (
    control_data["payoffs"]
    *
    discount_factor
)


control_prices = (
    control_data["terminal_prices"]
    *
    discount_factor
)


control_adjusted = control_variate_adjustment(
    samples=control_payoffs,
    control_samples=control_prices,
    expected_control=100
)


standard_price = np.mean(
    standard_discounted
)

antithetic_price = np.mean(
    antithetic_discounted
)

control_price = np.mean(
    control_adjusted
)


standard_variance = variance(
    standard_discounted
)

antithetic_variance = variance(
    antithetic_discounted
)

control_variance = variance(
    control_adjusted
)


standard_error_mc = standard_error(
    standard_discounted
)

antithetic_error_mc = standard_error(
    antithetic_discounted
)

control_error_mc = standard_error(
    control_adjusted
)


variance_reduction_pct = (
    variance_reduction(
        standard_discounted,
        antithetic_discounted
    )
    *
    100
)


error_reduction_pct = (
    1
    -
    antithetic_error_mc
    /
    standard_error_mc
) * 100


control_variance_reduction_pct = (
    variance_reduction(
        standard_discounted,
        control_adjusted
    )
    *
    100
)


control_error_reduction_pct = (
    1
    -
    control_error_mc
    /
    standard_error_mc
) * 100


print("\nVariance Reduction Benchmark")
print("----------------------------")

print(
    f"{'Method':<20}"
    f"{'Price':<15}"
    f"{'Std Error':<15}"
    f"{'Variance':<15}"
)

print("-" * 65)

print(
    f"{'Standard MC':<20}"
    f"{standard_price:<15.5f}"
    f"{standard_error_mc:<15.5f}"
    f"{standard_variance:<15.5f}"
)

print(
    f"{'Antithetic MC':<20}"
    f"{antithetic_price:<15.5f}"
    f"{antithetic_error_mc:<15.5f}"
    f"{antithetic_variance:<15.5f}"
)

print(
    f"{'Control Variate':<20}"
    f"{control_price:<15.5f}"
    f"{control_error_mc:<15.5f}"
    f"{control_variance:<15.5f}"
)

print()

print(
    f"Antithetic Variance Reduction: "
    f"{variance_reduction_pct:.2f}%"
)

print(
    f"Antithetic Std Error Reduction: "
    f"{error_reduction_pct:.2f}%"
)

print()

print(
    f"Control Variate Variance Reduction: "
    f"{control_variance_reduction_pct:.2f}%"
)

print(
    f"Control Variate Std Error Reduction: "
    f"{control_error_reduction_pct:.2f}%"
)