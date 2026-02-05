from quantmc.analytics.implied_volatility import (
    black_scholes_call_price,
    implied_volatility
)


def test_implied_volatility_recovers_volatility():

    spot = 100
    strike = 100
    rate = 0.05
    time = 1

    volatility = 0.2


    price = black_scholes_call_price(
        spot,
        strike,
        rate,
        volatility,
        time
    )


    recovered = implied_volatility(
        market_price=price,
        spot=spot,
        strike=strike,
        risk_free_rate=rate,
        time=time
    )


    assert abs(
        recovered - volatility
    ) < 1e-6