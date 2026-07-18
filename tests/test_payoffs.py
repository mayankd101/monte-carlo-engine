import numpy as np

from quantmc.options.payoffs import (
    EuropeanCall,
    EuropeanPut,
    AsianCall
)


def test_european_call():

    prices = np.array([
        [100, 110],
        [100, 90]
    ])

    payoff = EuropeanCall(100)

    result = payoff.calculate(prices)

    assert result.tolist() == [10, 0]


def test_european_put():

    prices = np.array([
        [100, 110],
        [100, 90]
    ])

    payoff = EuropeanPut(100)

    result = payoff.calculate(prices)

    assert result.tolist() == [0, 10]


def test_asian_call():

    prices = np.array([
        [100, 105, 110],
        [100, 95, 90]
    ])

    payoff = AsianCall(100)

    result = payoff.calculate(prices)

    assert result.tolist() == [7.5, 0]