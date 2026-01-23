from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.variance_reduction import variance


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

    # Average paired antithetic samples
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