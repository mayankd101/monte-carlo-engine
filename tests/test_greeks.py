from quantmc.analytics import BlackScholesGreeks


def test_delta():

    greeks = BlackScholesGreeks(
        100,
        100,
        0.05,
        0.2,
        1
    )

    assert abs(greeks.delta() - 0.6368) < 0.001


def test_gamma():

    greeks = BlackScholesGreeks(
        100,
        100,
        0.05,
        0.2,
        1
    )

    assert abs(greeks.gamma() - 0.0188) < 0.001