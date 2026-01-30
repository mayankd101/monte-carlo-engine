from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.variance_reduction import (
    variance,
    control_variate_adjustment
)
import numpy as np


def test_antithetic_paths():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    paths = model.simulate(
        time=1,
        steps=10,
        paths=100,
        seed=42,
        antithetic=True
    )

    assert paths.shape == (
        100,
        11
    )


def test_antithetic_reduces_variance():

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

    standard_payoffs = pricer.simulate_payoffs(
        time=1,
        steps=100,
        paths=10000,
        seed=42,
        antithetic=False
    )

    antithetic_payoffs = pricer.simulate_payoffs(
        time=1,
        steps=100,
        paths=10000,
        seed=42,
        antithetic=True
    )

    antithetic_pairs = (
        antithetic_payoffs[:5000]
        +
        antithetic_payoffs[5000:]
    ) / 2

    standard_variance = variance(
        standard_payoffs
    )

    antithetic_variance = variance(
        antithetic_pairs
    )

    assert antithetic_variance < standard_variance


def test_control_variate_reduces_variance():

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

    data = pricer.simulate_payoffs_and_prices(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    discount_factor = (
        2.718281828459045
        **
        (-0.05)
    )

    discounted_payoffs = (
        data["payoffs"]
        *
        discount_factor
    )

    discounted_prices = (
        data["terminal_prices"]
        *
        discount_factor
    )

    adjusted = control_variate_adjustment(
        samples=discounted_payoffs,
        control_samples=discounted_prices,
        expected_control=100
    )

    original_variance = variance(
        discounted_payoffs
    )

    adjusted_variance = variance(
        adjusted
    )

    assert adjusted_variance < original_variance


def test_control_variate_preserves_price():

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

    data = pricer.simulate_payoffs_and_prices(
        time=1,
        steps=100,
        paths=10000,
        seed=42
    )

    discount_factor = np.exp(
        -0.05
    )

    discounted_payoffs = (
        data["payoffs"]
        *
        discount_factor
    )

    discounted_prices = (
        data["terminal_prices"]
        *
        discount_factor
    )

    adjusted = control_variate_adjustment(
        samples=discounted_payoffs,
        control_samples=discounted_prices,
        expected_control=100
    )

    original_price = discounted_payoffs.mean()

    adjusted_price = adjusted.mean()

    assert abs(
        original_price - adjusted_price
    ) < 0.5