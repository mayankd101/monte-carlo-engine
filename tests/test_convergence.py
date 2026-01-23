from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.convergence import convergence_analysis


def test_convergence_analysis():

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

    assert 1000 in results
    assert 5000 in results

    for paths, stats in results.items():

        assert "price" in stats
        assert "standard_error" in stats
        assert "confidence_interval" in stats
        assert "runtime" in stats

        assert stats["standard_error"] > 0
        assert stats["runtime"] > 0