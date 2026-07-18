from abc import ABC, abstractmethod
import numpy as np


class OptionPayoff(ABC):
    """
    Abstract base class for option payoffs.
    """

    @abstractmethod
    def calculate(self, prices):
        pass


class EuropeanCall(OptionPayoff):
    """
    European call option.

    Payoff:
        max(S_T - K, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike

    def calculate(self, prices):
        terminal_prices = prices[:, -1]

        return np.maximum(
            terminal_prices - self.strike,
            0
        )


class EuropeanPut(OptionPayoff):
    """
    European put option.

    Payoff:
        max(K - S_T, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike

    def calculate(self, prices):
        terminal_prices = prices[:, -1]

        return np.maximum(
            self.strike - terminal_prices,
            0
        )


class AsianCall(OptionPayoff):
    """
    Asian call option.

    Payoff:
        max(average(S_t) - K, 0)
    """

    def __init__(self, strike: float):
        self.strike = strike

    def calculate(self, prices):
        average_price = np.mean(
            prices[:, 1:],
            axis=1
        )

        return np.maximum(
            average_price - self.strike,
            0
        )