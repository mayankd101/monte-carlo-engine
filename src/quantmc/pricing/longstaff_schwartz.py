import numpy as np


class LongstaffSchwartzPricer:

    def __init__(
        self,
        model,
        payoff,
        risk_free_rate
    ):
        self.model = model
        self.payoff = payoff
        self.r = risk_free_rate


    def _simulate_paths(
        self,
        time,
        steps,
        paths,
        seed=None
    ):

        return self.model.simulate(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed
        )


    def _initialize_cashflows(
        self,
        simulated_paths
    ):
        """
        Initialize cashflows at maturity
        """

        return self.payoff.terminal_payoff(
            simulated_paths
        )


    def price(
        self,
        time,
        steps,
        paths,
        seed=None
    ):

        simulated_paths = self._simulate_paths(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed
        )

        cashflows = self._initialize_cashflows(
            simulated_paths
        )

        return np.mean(
            cashflows
            *
            np.exp(-self.r * time)
        )