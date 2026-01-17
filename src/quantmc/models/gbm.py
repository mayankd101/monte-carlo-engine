import numpy as np


class GBM:
    """
    Geometric Brownian Motion simulator.

    Models stock prices using:

    dS = rSdt + sigma*S*dW
    """

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
        time: float,
        steps: int,
        paths: int,
        seed: int | None = None
    ):
        """
        Generate Monte Carlo stock price paths.

        Parameters:
            time:
                total simulation time in years

            steps:
                number of time steps

            paths:
                number of simulated paths

        Returns:
            numpy array of shape:
            (paths, steps + 1)
        """

        if seed is not None:
            np.random.seed(seed)

        dt = time / steps

        prices = np.zeros(
            (paths, steps + 1)
        )

        prices[:, 0] = self.S0

        random_shocks = np.random.normal(
            0,
            1,
            size=(paths, steps)
        )

        for step in range(1, steps + 1):

            prices[:, step] = (
                prices[:, step - 1]
                *
                np.exp(
                    (
                        self.r
                        - 0.5 * self.sigma ** 2
                    ) * dt
                    +
                    self.sigma
                    * np.sqrt(dt)
                    * random_shocks[:, step - 1]
                )
            )

        return prices