from quantmc.models.gbm import GBM
from quantmc.options.payoffs import (
    EuropeanPut,
    AmericanPut
)
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.pricing.longstaff_schwartz import (
    LongstaffSchwartzPricer
)


def create_models():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    european = EuropeanPut(
        strike=100
    )

    american = AmericanPut(
        strike=100
    )

    return model, european, american


def test_american_put_exceeds_european():

    model, european, american = create_models()


    european_pricer = MonteCarloPricer(
        model=model,
        payoff=european,
        risk_free_rate=0.05
    )


    american_pricer = LongstaffSchwartzPricer(
        model=model,
        payoff=american,
        risk_free_rate=0.05
    )


    european_price = european_pricer.price(
        time=1,
        steps=50,
        paths=50000,
        seed=42
    )


    american_price = american_pricer.price(
        time=1,
        steps=50,
        paths=50000,
        seed=42
    )


    assert american_price >= european_price



def test_american_price_positive():

    model, _, american = create_models()


    pricer = LongstaffSchwartzPricer(
        model=model,
        payoff=american,
        risk_free_rate=0.05
    )


    price = pricer.price(
        time=1,
        steps=50,
        paths=10000,
        seed=42
    )


    assert price > 0



def test_longstaff_schwartz_reproducible():

    model, _, american = create_models()


    pricer = LongstaffSchwartzPricer(
        model=model,
        payoff=american,
        risk_free_rate=0.05
    )


    price1 = pricer.price(
        time=1,
        steps=50,
        paths=10000,
        seed=42
    )


    price2 = pricer.price(
        time=1,
        steps=50,
        paths=10000,
        seed=42
    )


    assert price1 == price2