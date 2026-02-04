from abc import ABC, abstractmethod
import numpy as np


class OptionPayoff(ABC):

    @abstractmethod
    def terminal_payoff(self, prices):
        pass

    @abstractmethod
    def exercise_value(self, prices):
        pass

    def calculate(self, prices):
        return self.terminal_payoff(prices)


class EuropeanCall(OptionPayoff):
    """
    European call option

    Payoff:
        max(S - K, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike

    def terminal_payoff(self, prices):

        terminal_prices = prices[:, -1]

        return np.maximum(
            terminal_prices - self.strike,
            0
        )

    def exercise_value(self, prices):

        return np.maximum(
            prices - self.strike,
            0
        )


class EuropeanPut(OptionPayoff):
    """
    European put option

    Payoff:
        max(K - S, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike

    def terminal_payoff(self, prices):

        terminal_prices = prices[:, -1]

        return np.maximum(
            self.strike - terminal_prices,
            0
        )

    def exercise_value(self, prices):

        return np.maximum(
            self.strike - prices,
            0
        )


class AsianCall(OptionPayoff):
    """
    Arithmetic Asian call option

    Payoff:
        max(mean(S) - K, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike

    def terminal_payoff(self, prices):

        average_price = np.mean(
            prices[:, 1:],
            axis=1
        )

        return np.maximum(
            average_price - self.strike,
            0
        )

    def exercise_value(self, prices):
        raise NotImplementedError(
            "Asian options do not have a simple intrinsic "
            "exercise value."
        )
    
class AmericanPut(OptionPayoff):
    """
    American put option

    Payoff:
        max(K - S_t, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike


    def terminal_payoff(self, prices):

        terminal_prices = prices[:, -1]

        return np.maximum(
            self.strike - terminal_prices,
            0
        )


    def exercise_value(self, prices):

        return np.maximum(
            self.strike - prices,
            0
        )