import numpy as np


class GBM:

    def __init__(
        self,
        initial_price: float,
        risk_free_rate: float,
        volatility: float
    ):
        self.S0 = initial_price
        self.r = risk_free_rate
        self.sigma = volatility


    def simulate(
        self,
        time,
        steps,
        paths,
        seed=None,
        antithetic=False
    ):


        if seed is not None:
            np.random.seed(seed)

        dt = time / steps

        if antithetic:
            if paths % 2 != 0:
                raise ValueError(
                    "Antithetic variates require an even number of paths"
                )

            half_paths = paths // 2

            Z = np.random.normal(
                size=(half_paths, steps)
            )

            Z = np.vstack(
                [
                    Z,
                    -Z
                ]
            )

        else:
            Z = np.random.normal(
                size=(paths, steps)
            )


        increments = (
            (
                self.r - 0.5 * self.sigma ** 2
            )
            * dt
            +
            self.sigma
            * np.sqrt(dt)
            * Z
        )


        prices = np.zeros(
            (paths, steps + 1)
        )

        prices[:, 0] = self.S0


        prices[:, 1:] = (
            self.S0
            *
            np.exp(
                np.cumsum(
                    increments,
                    axis=1
                )
            )
        )

        return prices