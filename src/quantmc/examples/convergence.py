from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.convergence import convergence_analysis


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


results = convergence_analysis(
    pricer,
    time_to_expiry=1,
    steps=100,
    path_counts=[
        1000,
        5000,
        10000,
        50000
    ],
    seed=42
)


print("\nMonte Carlo Convergence Analysis")
print("--------------------------------")

for paths, stats in results.items():

    lower, upper = stats["confidence_interval"]

    print(
        f"Paths: {paths}"
    )

    print(
        f"Price: {stats['price']:.5f}"
    )

    print(
        f"Std Error: {stats['standard_error']:.5f}"
    )

    print(
        f"95% CI: ({lower:.5f}, {upper:.5f})"
    )

    print(
        f"Runtime: {stats['runtime']:.4f}s"
    )

    print()