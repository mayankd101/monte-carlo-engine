import numpy as np
from scipy.stats import norm


class BlackScholes:

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


    def call_price(self):

        d1 = self._d1()
        d2 = self._d2()

        return (
            self.S * norm.cdf(d1)
            -
            self.K
            * np.exp(-self.r * self.T)
            * norm.cdf(d2)
        )


    def put_price(self):

        d1 = self._d1()
        d2 = self._d2()

        return (
            self.K
            * np.exp(-self.r * self.T)
            * norm.cdf(-d2)
            -
            self.S
            * norm.cdf(-d1)
        )