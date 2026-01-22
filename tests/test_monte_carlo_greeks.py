import numpy as np

from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.greeks import MonteCarloGreeks


def test_monte_carlo_delta():

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

    greeks = MonteCarloGreeks(pricer)

    delta = greeks.delta(
        time=1,
        steps=100,
        paths=20000,
        bump=0.1,
        seed=42
    )

    assert 0 < delta < 1


def test_monte_carlo_vega():

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

    greeks = MonteCarloGreeks(pricer)

    vega = greeks.vega(
        time=1,
        steps=100,
        paths=20000,
        bump=0.01,
        seed=42
    )

    assert vega > 0

def test_monte_carlo_gamma():

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

    greeks = MonteCarloGreeks(pricer)

    gamma = greeks.gamma(
        time=1,
        steps=100,
        paths=20000,
        bump=0.1,
        seed=42
    )

    assert gamma > 0