import numpy as np

from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.variance_reduction import variance, variance_reduction


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


standard_payoffs = (
    pricer.simulate_payoffs(
        time=1,
        steps=100,
        paths=paths,
        seed=42,
        antithetic=False
    )
)


antithetic_payoffs = (
    pricer.simulate_payoffs(
        time=1,
        steps=100,
        paths=paths,
        seed=42,
        antithetic=True
    )
)


standard_variance = variance(
    standard_payoffs
)

antithetic_variance = variance(
    antithetic_payoffs
)


reduction = variance_reduction(
    standard_payoffs,
    antithetic_payoffs
)


print("\nVariance Reduction Benchmark")
print("----------------------------")

print(
    f"Standard MC Variance: {standard_variance:.6f}"
)

print(
    f"Antithetic Variance: {antithetic_variance:.6f}"
)

print(
    f"Variance Reduction: {reduction * 100:.2f}%"
)