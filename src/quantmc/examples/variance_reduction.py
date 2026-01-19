from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing import MonteCarloPricer


model = GBM(
    initial_price=100,
    risk_free_rate=0.05,
    volatility=0.2
)


option = EuropeanCall(100)


engine = MonteCarloPricer(
    model,
    option,
    0.05
)


normal = engine.price(
    time=1,
    steps=252,
    paths=10000,
    seed=42,
    antithetic=False
)


reduced = engine.price(
    time=1,
    steps=252,
    paths=10000,
    seed=42,
    antithetic=True
)


print(
    f"Normal Monte Carlo: ${normal:.5f}"
)

print(
    f"Antithetic Monte Carlo: ${reduced:.5f}"
)