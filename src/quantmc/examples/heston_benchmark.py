from quantmc.models.gbm import GBM
from quantmc.models.heston import Heston

from quantmc.options.payoffs import EuropeanCall
from quantmc.pricing.monte_carlo import MonteCarloPricer


def main():

    payoff = EuropeanCall(
        strike=100
    )


    gbm = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )


    heston = Heston(
        initial_price=100,
        initial_variance=0.04,
        risk_free_rate=0.05,
        kappa=2,
        theta=0.04,
        volatility_of_volatility=0.3,
        correlation=-0.7
    )


    gbm_pricer = MonteCarloPricer(
        model=gbm,
        payoff=payoff,
        risk_free_rate=0.05
    )


    heston_pricer = MonteCarloPricer(
        model=heston,
        payoff=payoff,
        risk_free_rate=0.05
    )


    gbm_price = gbm_pricer.price(
        time=1,
        steps=100,
        paths=50000,
        seed=42
    )


    heston_price = heston_pricer.price(
        time=1,
        steps=100,
        paths=50000,
        seed=42
    )


    print("\nHeston vs GBM Benchmark")
    print("----------------------")

    print(
        f"GBM Price:    {gbm_price:.5f}"
    )

    print(
        f"Heston Price: {heston_price:.5f}"
    )

    print(
        f"Difference:   "
        f"{heston_price - gbm_price:.5f}"
    )


if __name__ == "__main__":
    main()