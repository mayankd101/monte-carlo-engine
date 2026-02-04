import copy
import numpy as np
from scipy.stats import norm


class BlackScholesGreeks:

    def __init__(
        self,
        spot,
        strike,
        risk_free_rate,
        volatility,
        time
    ):
        self.S = spot
        self.K = strike
        self.r = risk_free_rate
        self.sigma = volatility
        self.T = time


    def _d1(self):

        return (
            np.log(self.S / self.K)
            +
            (
                self.r
                +
                0.5 * self.sigma ** 2
            )
            *
            self.T
        ) / (
            self.sigma * np.sqrt(self.T)
        )


    def _d2(self):

        return (
            self._d1()
            -
            self.sigma * np.sqrt(self.T)
        )


    def delta(self):
        return norm.cdf(self._d1())


    def gamma(self):

        return (
            norm.pdf(self._d1())
            /
            (
                self.S
                *
                self.sigma
                *
                np.sqrt(self.T)
            )
        )


    def vega(self):

        return (
            self.S
            *
            norm.pdf(self._d1())
            *
            np.sqrt(self.T)
        )


    def theta(self):

        return (
            -(
                self.S
                *
                norm.pdf(self._d1())
                *
                self.sigma
            )
            /
            (2 * np.sqrt(self.T))
            -
            self.r
            *
            self.K
            *
            np.exp(-self.r * self.T)
            *
            norm.cdf(self._d2())
        )


    def rho(self):

        return (
            self.K
            *
            self.T
            *
            np.exp(-self.r * self.T)
            *
            norm.cdf(self._d2())
        )


class MonteCarloGreeks:

    def __init__(
        self,
        pricer
    ):
        self.pricer = pricer


    def _price_with_model(
        self,
        model,
        time,
        steps,
        paths,
        seed
    ):

        original = self.pricer.model

        self.pricer.model = model

        price = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = original

        return price


    def delta(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):

        up = copy.deepcopy(self.pricer.model)
        down = copy.deepcopy(self.pricer.model)

        up.S0 += bump
        down.S0 -= bump

        return (
            self._price_with_model(
                up,
                time,
                steps,
                paths,
                seed
            )
            -
            self._price_with_model(
                down,
                time,
                steps,
                paths,
                seed
            )
        ) / (2 * bump)


    def gamma(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):

        up = copy.deepcopy(self.pricer.model)
        mid = copy.deepcopy(self.pricer.model)
        down = copy.deepcopy(self.pricer.model)

        up.S0 += bump
        down.S0 -= bump

        return (
            self._price_with_model(
                up,
                time,
                steps,
                paths,
                seed
            )
            -
            2
            *
            self._price_with_model(
                mid,
                time,
                steps,
                paths,
                seed
            )
            +
            self._price_with_model(
                down,
                time,
                steps,
                paths,
                seed
            )
        ) / (bump ** 2)


    def vega(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):

        up = copy.deepcopy(self.pricer.model)
        down = copy.deepcopy(self.pricer.model)

        up.sigma += bump
        down.sigma -= bump

        return (
            self._price_with_model(
                up,
                time,
                steps,
                paths,
                seed
            )
            -
            self._price_with_model(
                down,
                time,
                steps,
                paths,
                seed
            )
        ) / (2 * bump)


    def theta(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):

        shorter_time = time - bump

        if shorter_time <= 0:
            raise ValueError(
                "Time bump must be smaller than maturity"
            )

        price_shorter = self.pricer.price(
            shorter_time,
            steps,
            paths,
            seed=seed
        )

        price_current = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        return (
            price_shorter
            -
            price_current
        ) / bump


    def rho(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):

        up = copy.deepcopy(
            self.pricer.model
        )

        down = copy.deepcopy(
            self.pricer.model
        )

        up.r += bump
        down.r -= bump

        return (
            self._price_with_model(
                up,
                time,
                steps,
                paths,
                seed
            )
            -
            self._price_with_model(
                down,
                time,
                steps,
                paths,
                seed
            )
        ) / (2 * bump)
    
    def delta_pathwise(
        self,
        time,
        steps,
        paths,
        seed=None
    ):

        simulated_prices = self.pricer.model.simulate(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed
        )

        terminal_prices = (
            simulated_prices[:, -1]
        )

        strike = self.pricer.payoff.strike

        indicator = (
            terminal_prices > strike
        )

        delta_samples = (
            indicator
            *
            terminal_prices
            /
            self.pricer.model.S0
        )

        discount_factor = np.exp(
            -self.pricer.r * time
        )

        return (
            discount_factor
            *
            np.mean(delta_samples)
        )