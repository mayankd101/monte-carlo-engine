from quantmc.pricing import BlackScholes


def test_black_scholes_call():

    model = BlackScholes(
        spot=100,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time=1
    )

    price = model.call_price()

    assert abs(price - 10.45) < 0.01