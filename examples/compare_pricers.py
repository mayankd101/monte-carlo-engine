from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing import (
    MonteCarloPricer,
    BlackScholes
)


S = 100
K = 100
r = 0.05
sigma = 0.2
T = 1


# Monte Carlo

model = GBM(
    initial_price=S,
    risk_free_rate=r,
    volatility=sigma
)

option = EuropeanCall(K)

mc_engine = MonteCarloPricer(
    model,
    option,
    r
)

mc_price = mc_engine.price(
    time=T,
    steps=252,
    paths=100000,
    seed=42
)


# Black-Scholes

bs = BlackScholes(
    S,
    K,
    r,
    sigma,
    T
)

bs_price = bs.call_price()


print(f"Monte Carlo Price: ${mc_price:.4f}")
print(f"Black-Scholes Price: ${bs_price:.4f}")

print(
    f"Absolute Error: ${abs(mc_price-bs_price):.4f}"
)