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
            * self.T
        ) / (
            self.sigma * np.sqrt(self.T)
        )


    def _d2(self):

        return self._d1() - (
            self.sigma * np.sqrt(self.T)
        )


    def delta(self):
        """
        Call option delta.
        """

        return norm.cdf(self._d1())


    def gamma(self):
        """
        Gamma for call and put.
        """

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
        """
        Vega per 1.0 volatility change.
        """

        return (
            self.S
            *
            norm.pdf(self._d1())
            *
            np.sqrt(self.T)
        )


    def theta(self):
        """
        Call option theta.
        """

        d1 = self._d1()
        d2 = self._d2()

        return (
            -(
                self.S
                *
                norm.pdf(d1)
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
            norm.cdf(d2)
        )


    def rho(self):
        """
        Call option rho.
        """

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

    def __init__(self, pricer):
        self.pricer = pricer


    def delta(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):
        """
        Delta using central finite difference
        """

        import copy

        original_model = self.pricer.model

        up_model = copy.deepcopy(original_model)
        down_model = copy.deepcopy(original_model)

        up_model.S0 += bump
        down_model.S0 -= bump

        self.pricer.model = up_model
        price_up = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = down_model
        price_down = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = original_model

        return (
            price_up - price_down
        ) / (2 * bump)


    def vega(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):
        """
        Vega using central finite difference
        """

        import copy

        original_model = self.pricer.model

        up_model = copy.deepcopy(original_model)
        down_model = copy.deepcopy(original_model)

        up_model.sigma += bump
        down_model.sigma -= bump

        self.pricer.model = up_model
        price_up = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = down_model
        price_down = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = original_model

        return (
            price_up - price_down
        ) / (2 * bump)
    
    
    def gamma(
        self,
        time,
        steps,
        paths,
        bump=0.01,
        seed=None
    ):
        """
        Gamma using central finite difference
        """

        import copy

        original_model = self.pricer.model

        up_model = copy.deepcopy(original_model)
        mid_model = copy.deepcopy(original_model)
        down_model = copy.deepcopy(original_model)

        up_model.S0 += bump
        down_model.S0 -= bump

        self.pricer.model = up_model
        price_up = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = mid_model
        price_mid = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = down_model
        price_down = self.pricer.price(
            time,
            steps,
            paths,
            seed=seed
        )

        self.pricer.model = original_model

        return (
            price_up
            - 2 * price_mid
            + price_down
        ) / (bump ** 2)