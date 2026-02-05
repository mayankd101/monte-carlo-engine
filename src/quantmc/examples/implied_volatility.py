from quantmc.analytics.implied_volatility import (
    black_scholes_call_price,
    implied_volatility
)


spot = 100
strike = 100
risk_free_rate = 0.05
time = 1


true_volatility = 0.20


market_price = black_scholes_call_price(
    spot=spot,
    strike=strike,
    risk_free_rate=risk_free_rate,
    volatility=true_volatility,
    time=time
)


recovered_volatility = implied_volatility(
    market_price=market_price,
    spot=spot,
    strike=strike,
    risk_free_rate=risk_free_rate,
    time=time,
)


print("\nImplied Volatility Example")
print("--------------------------")

print(
    f"Market Price: {market_price:.5f}"
)

print(
    f"True Volatility: {true_volatility:.5f}"
)

print(
    f"Recovered Volatility: {recovered_volatility:.5f}"
)

print(
    f"Error: {abs(true_volatility - recovered_volatility):.8f}"
)