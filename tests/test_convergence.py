from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.convergence import convergence_analysis


def test_convergence_output():

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


    results = convergence_analysis(
        pricer,
        time_to_expiry=1,
        steps=50,
        path_counts=[
            1000,
            5000
        ],
        seed=42
    )


    assert len(results) == 2


    for paths, stats in results.items():

        assert paths in [
            1000,
            5000
        ]

        assert "price" in stats
        assert "standard_error" in stats
        assert "confidence_interval" in stats
        assert "runtime" in stats


def test_more_paths_reduce_error():

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


    results = convergence_analysis(
        pricer,
        time_to_expiry=1,
        steps=100,
        path_counts=[
            1000,
            100000
        ],
        seed=42
    )


    assert (
        results[100000]["standard_error"]
        <
        results[1000]["standard_error"]
    )