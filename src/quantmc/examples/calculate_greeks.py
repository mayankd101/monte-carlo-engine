from quantmc.analytics import BlackScholesGreeks


greeks = BlackScholesGreeks(
    spot=100,
    strike=100,
    risk_free_rate=0.05,
    volatility=0.2,
    time=1
)


print(
    f"Delta: {greeks.delta():.4f}"
)

print(
    f"Gamma: {greeks.gamma():.4f}"
)

print(
    f"Vega: {greeks.vega():.4f}"
)

print(
    f"Theta: {greeks.theta():.4f}"
)

print(
    f"Rho: {greeks.rho():.4f}"
)