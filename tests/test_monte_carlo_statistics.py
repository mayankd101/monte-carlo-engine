import numpy as np

from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer


def test_price_with_statistics():

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

    result = pricer.price_with_statistics(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    assert "price" in result
    assert "standard_error" in result
    assert "confidence_interval" in result
    assert result["paths"] == 10000

    assert result["standard_error"] > 0
    assert (
        result["confidence_interval"][0]
        <
        result["price"]
        <
        result["confidence_interval"][1]
    )