from quantmc.models.heston import Heston
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer


def test_heston_price_positive():

    model = Heston(
        initial_price=100,
        initial_variance=0.04,
        risk_free_rate=0.05,
        kappa=2,
        theta=0.04,
        volatility_of_volatility=0.3,
        correlation=-0.7
    )

    payoff = EuropeanCall(
        strike=100
    )

    pricer = MonteCarloPricer(
        model=model,
        payoff=payoff,
        risk_free_rate=0.05
    )

    price = pricer.price(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    assert price > 0


def test_heston_reproducibility():

    model = Heston(
        initial_price=100,
        initial_variance=0.04,
        risk_free_rate=0.05,
        kappa=2,
        theta=0.04,
        volatility_of_volatility=0.3,
        correlation=-0.7
    )

    payoff = EuropeanCall(
        strike=100
    )

    pricer = MonteCarloPricer(
        model=model,
        payoff=payoff,
        risk_free_rate=0.05
    )

    price1 = pricer.price(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    price2 = pricer.price(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    assert price1 == price2