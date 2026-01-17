from quantmc.models.gbm import GBM
import matplotlib.pyplot as plt


model = GBM(
    initial_price=100,
    risk_free_rate=0.05,
    volatility=0.20
)


paths = model.simulate(
    time=1,
    steps=252,
    paths=100,
    seed=42
)


plt.figure(figsize=(10, 5))

plt.plot(paths.T)

plt.title("Geometric Brownian Motion Simulation")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price")

plt.show()