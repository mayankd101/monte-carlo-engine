import numpy as np


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


    def price(
        self,
        time,
        steps,
        paths,
        seed=None
    ):


        simulated_prices = self.model.simulate(
            time=time,
            steps=steps,
            paths=paths,
            seed=seed
        )

        payoffs = self.payoff.calculate(
            simulated_prices
        )

        expected_payoff = np.mean(payoffs)

        discounted_value = (
            np.exp(-self.r * time)
            *
            expected_payoff
        )

        return discounted_value