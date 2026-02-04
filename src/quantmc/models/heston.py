import numpy as np


class Heston:

    def __init__(
        self,
        initial_price: float,
        initial_variance: float,
        risk_free_rate: float,
        kappa: float,
        theta: float,
        volatility_of_volatility: float,
        correlation: float
    ):
        

        self.S0 = initial_price
        self.v0 = initial_variance
        self.r = risk_free_rate

        self.kappa = kappa
        self.theta = theta
        self.xi = volatility_of_volatility
        self.rho = correlation


    def simulate(
        self,
        time,
        steps,
        paths,
        seed=None
    ):

        if seed is not None:
            np.random.seed(seed)


        dt = time / steps


        prices = np.zeros(
            (paths, steps + 1)
        )

        variance = np.zeros(
            (paths, steps + 1)
        )


        prices[:, 0] = self.S0

        variance[:, 0] = self.v0


        for t in range(1, steps + 1):

            z1 = np.random.normal(
                size=paths
            )

            z2 = np.random.normal(
                size=paths
            )


            w1 = z1

            w2 = (
                self.rho * z1
                +
                np.sqrt(
                    1 - self.rho ** 2
                )
                *
                z2
            )


            previous_variance = variance[:, t-1]


            variance[:, t] = (
                previous_variance
                +
                self.kappa
                *
                (
                    self.theta
                    -
                    previous_variance
                )
                *
                dt
                +
                self.xi
                *
                np.sqrt(
                    np.maximum(
                        previous_variance,
                        0
                    )
                )
                *
                np.sqrt(dt)
                *
                w2
            )


            variance[:, t] = np.maximum(
                variance[:, t],
                0
            )


            prices[:, t] = (
                prices[:, t-1]
                *
                np.exp(
                    (
                        self.r
                        -
                        0.5 * previous_variance
                    )
                    *
                    dt
                    +
                    np.sqrt(
                        previous_variance
                    )
                    *
                    np.sqrt(dt)
                    *
                    w1
                )
            )


        return prices