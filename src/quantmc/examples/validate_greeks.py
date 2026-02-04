import numpy as np

from quantmc.models.gbm import GBM
from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.analytics.greeks import (
    BlackScholesGreeks,
    MonteCarloGreeks
)


def relative_error(
    monte_carlo,
    analytical
):

    return (
        abs(monte_carlo - analytical)
        /
        abs(analytical)
    ) * 100


def main():

    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1

    model = GBM(
        initial_price=S0,
        risk_free_rate=r,
        volatility=sigma
    )

    payoff = EuropeanCall(
        strike=K
    )

    pricer = MonteCarloPricer(
        model=model,
        payoff=payoff,
        risk_free_rate=r
    )

    mc_greeks = MonteCarloGreeks(
        pricer
    )

    bs_greeks = BlackScholesGreeks(
        spot=S0,
        strike=K,
        risk_free_rate=r,
        volatility=sigma,
        time=T
    )


    paths = 500000
    steps = 100
    seed = 42


    results = [
        (
            "Delta",
            mc_greeks.delta(
                T,
                steps,
                paths,
                seed=seed
            ),
            bs_greeks.delta()
        ),
        (
            "Pathwise Delta",
            mc_greeks.delta_pathwise(
                T,
                steps,
                paths,
                seed=seed
            ),
            bs_greeks.delta()
        ),
        (
            "Gamma",
            mc_greeks.gamma(
                T,
                steps,
                paths,
                seed=seed
            ),
            bs_greeks.gamma()
        ),
        (
            "Vega",
            mc_greeks.vega(
                T,
                steps,
                paths,
                seed=seed
            ),
            bs_greeks.vega()
        ),
        (
            "Theta",
            mc_greeks.theta(
                T,
                steps,
                paths,
                seed=seed
            ),
            bs_greeks.theta()
        ),
        (
            "Rho",
            mc_greeks.rho(
                T,
                steps,
                paths,
                seed=seed
            ),
            bs_greeks.rho()
        )
    ]


    print("\nGreek Validation Benchmark")
    print("-------------------------")

    print(
        f"{'Greek':<18}"
        f"{'Monte Carlo':<18}"
        f"{'Black-Scholes':<18}"
        f"{'Error %':<10}"
    )

    print("-" * 70)


    for name, mc, bs in results:

        print(
            f"{name:<18}"
            f"{mc:<18.6f}"
            f"{bs:<18.6f}"
            f"{relative_error(mc, bs):<10.3f}"
        )


if __name__ == "__main__":
    main()