from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing import MonteCarloPricer


stock = GBM(
    initial_price=100,
    risk_free_rate=0.05,
    volatility=0.2
)


option = EuropeanCall(
    strike=100
)


engine = MonteCarloPricer(
    model=stock,
    payoff=option,
    risk_free_rate=0.05
)


price = engine.price(
    time=1,
    steps=252,
    paths=100000,
    seed=42
)


print(
    f"European Call Price: ${price:.2f}"
)