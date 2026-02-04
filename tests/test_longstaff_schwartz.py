from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanPut
from quantmc.pricing.longstaff_schwartz import LongstaffSchwartzPricer


def test_longstaff_schwartz_initialization():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    payoff = EuropeanPut(
        strike=100
    )

    pricer = LongstaffSchwartzPricer(
        model=model,
        payoff=payoff,
        risk_free_rate=0.05
    )

    price = pricer.price(
        time=1,
        steps=50,
        paths=10000,
        seed=42
    )

    assert price > 0