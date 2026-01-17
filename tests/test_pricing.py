from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing import MonteCarloPricer


def test_monte_carlo_price():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    payoff = EuropeanCall(100)

    pricer = MonteCarloPricer(
        model,
        payoff,
        0.05
    )

    price = pricer.price(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    assert 8 < price < 13