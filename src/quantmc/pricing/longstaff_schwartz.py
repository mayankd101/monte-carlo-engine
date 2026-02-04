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


    def _discount(
        self,
        cashflows,
        dt
    ):

        return cashflows * np.exp(
            -self.r * dt
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


        dt = time / steps



        cashflows = self.payoff.terminal_payoff(
            simulated_paths
        )



        for t in range(
            steps - 1,
            0,
            -1
        ):

            stock_prices = (
                simulated_paths[:, t]
            )


            exercise_values = (
                self.payoff.exercise_value(
                    stock_prices
                )
            )


            in_money = (
                exercise_values > 0
            )


            if np.any(in_money):

                X = stock_prices[in_money]

                Y = self._discount(
                    cashflows[in_money],
                    dt
                )


                coefficients = np.polyfit(
                    X,
                    Y,
                    2
                )


                continuation = np.polyval(
                    coefficients,
                    X
                )


                exercise = (
                    exercise_values[in_money]
                    >
                    continuation
                )


                exercise_indices = np.where(
                    in_money
                )[0][exercise]


                cashflows[
                    exercise_indices
                ] = exercise_values[
                    exercise_indices
                ]


            cashflows = self._discount(
                cashflows,
                dt
            )


        return np.mean(
            cashflows
        )