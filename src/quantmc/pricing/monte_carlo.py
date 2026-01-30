import numpy as np

from quantmc.analytics.statistics import (
    standard_error,
    confidence_interval
)


class MonteCarloPricer:

    def __init__(
        self,
        model,
        payoff,
        risk_free_rate
    ):
        self.model = model
        self.payoff = payoff
        self.r = risk_free_rate


    def _simulate_payoffs(
        self,
        time,
        steps,
        paths,
        seed=None,
        antithetic=False
    ):

        simulated_prices = self.model.simulate(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed,
            antithetic=antithetic
        )

        payoffs = self.payoff.calculate(
            simulated_prices
        )

        return payoffs


    def simulate_payoffs(
        self,
        time,
        steps,
        paths,
        seed=None,
        antithetic=False
    ):
        """
        Generate raw option payoff samples.

        Useful for:
        - variance analysis
        - variance reduction comparisons
        - Monte Carlo diagnostics
        """

        return self._simulate_payoffs(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed,
            antithetic=antithetic
        )

    def simulate_payoffs_and_prices(
        self,
        time,
        steps,
        paths,
        seed=None,
        antithetic=False
    ):

        simulated_prices = self.model.simulate(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed,
            antithetic=antithetic
        )

        payoffs = self.payoff.calculate(
            simulated_prices
        )

        terminal_prices = simulated_prices[:, -1]

        return {
            "payoffs": payoffs,
            "terminal_prices": terminal_prices
        }

    def price(
        self,
        time,
        steps,
        paths,
        seed=None,
        antithetic=False
    ):

        payoffs = self._simulate_payoffs(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed,
            antithetic=antithetic
        )

        expected_payoff = np.mean(payoffs)

        discounted_value = (
            np.exp(-self.r * time)
            *
            expected_payoff
        )

        return discounted_value


    def price_with_statistics(
        self,
        time,
        steps,
        paths,
        seed=None,
        antithetic=False,
        confidence=0.95
    ):

        payoffs = self._simulate_payoffs(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed,
            antithetic=antithetic
        )

        discount_factor = np.exp(
            -self.r * time
        )

        discounted_payoffs = (
            payoffs * discount_factor
        )

        price = np.mean(
            discounted_payoffs
        )

        error = standard_error(
            discounted_payoffs
        )

        interval = confidence_interval(
            price,
            error,
            confidence
        )

        return {
            "price": price,
            "standard_error": error,
            "confidence_interval": interval,
            "paths": paths
        }