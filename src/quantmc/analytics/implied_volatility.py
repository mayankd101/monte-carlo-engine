import numpy as np

from quantmc.analytics.greeks import BlackScholesGreeks


def black_scholes_call_price(
    spot,
    strike,
    risk_free_rate,
    volatility,
    time
):

    greeks = BlackScholesGreeks(
        spot=spot,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time=time
    )

    d1 = greeks._d1()
    d2 = greeks._d2()

    from scipy.stats import norm

    price = (
        spot * norm.cdf(d1)
        -
        strike
        *
        np.exp(-risk_free_rate * time)
        *
        norm.cdf(d2)
    )

    return price



def implied_volatility(
    market_price,
    spot,
    strike,
    risk_free_rate,
    time,
    initial_volatility=0.2,
    tolerance=1e-6,
    max_iterations=100
):

    volatility = initial_volatility


    for _ in range(max_iterations):

        price = black_scholes_call_price(
            spot,
            strike,
            risk_free_rate,
            volatility,
            time
        )


        greeks = BlackScholesGreeks(
            spot=spot,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time=time
        )


        vega = greeks.vega()


        difference = (
            price
            -
            market_price
        )


        if abs(difference) < tolerance:
            return volatility


        volatility -= (
            difference
            /
            vega
        )


        if volatility <= 0:
            volatility = tolerance


    raise RuntimeError(
        "Implied volatility did not converge"
    )