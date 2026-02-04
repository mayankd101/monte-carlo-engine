from quantmc.models.gbm import GBM
from quantmc.options.payoffs import (
    EuropeanPut,
    AmericanPut
)
from quantmc.pricing.monte_carlo import MonteCarloPricer
from quantmc.pricing.longstaff_schwartz import (
    LongstaffSchwartzPricer
)


def main():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )


    european = EuropeanPut(
        strike=100
    )

    american = AmericanPut(
        strike=100
    )


    european_pricer = MonteCarloPricer(
        model=model,
        payoff=european,
        risk_free_rate=0.05
    )


    american_pricer = LongstaffSchwartzPricer(
        model=model,
        payoff=american,
        risk_free_rate=0.05
    )


    european_price = european_pricer.price(
        time=1,
        steps=100,
        paths=100000,
        seed=42
    )


    american_price = american_pricer.price(
        time=1,
        steps=100,
        paths=100000,
        seed=42
    )


    print("\nAmerican Option Benchmark")
    print("------------------------")

    print(
        f"European Put: {european_price:.5f}"
    )

    print(
        f"American Put: {american_price:.5f}"
    )

    print(
        f"Early Exercise Premium: "
        f"{american_price - european_price:.5f}"
    )


if __name__ == "__main__":
    main()