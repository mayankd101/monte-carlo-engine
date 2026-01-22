from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.greeks import (
    BlackScholesGreeks,
    MonteCarloGreeks
)
from quantmc.analytics.validation import compare_greeks


def test_greek_comparison():

    spot = 100
    strike = 100
    rate = 0.05
    volatility = 0.2
    time = 1

    model = GBM(
        initial_price=spot,
        risk_free_rate=rate,
        volatility=volatility
    )

    payoff = EuropeanCall(
        strike=strike
    )

    pricer = MonteCarloPricer(
        model=model,
        payoff=payoff,
        risk_free_rate=rate
    )

    analytical = BlackScholesGreeks(
        spot=spot,
        strike=strike,
        risk_free_rate=rate,
        volatility=volatility,
        time=time
    )

    monte_carlo = MonteCarloGreeks(
        pricer
    )

    results = compare_greeks(
        analytical,
        monte_carlo,
        time=time,
        steps=100,
        paths=50000,
        seed=42
    )

    assert abs(
        results["delta"]["black_scholes"]
        -
        results["delta"]["monte_carlo"]
    ) < 0.05

    assert abs(
        results["vega"]["black_scholes"]
        -
        results["vega"]["monte_carlo"]
    ) < 5