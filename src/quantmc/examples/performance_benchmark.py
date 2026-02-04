import time

from quantmc.models.gbm import GBM
from quantmc.performance.numba_gbm import (
    simulate_gbm_numba
)
from quantmc.models.heston import Heston
from quantmc.performance.numba_heston import (
    simulate_heston_numba
)

def main():

    paths = 500000
    steps = 252


    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )


    start = time.time()

    model.simulate(
        time=1,
        steps=steps,
        paths=paths,
        seed=42
    )

    numpy_time = (
        time.time()
        -
        start
    )


    start = time.time()

    simulate_gbm_numba(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time=1,
        steps=steps,
        paths=paths,
        seed=42
    )

    numba_time = (
        time.time()
        -
        start
    )


    print("\nMonte Carlo Performance Benchmark")
    print("--------------------------------")

    print(
        f"NumPy GBM:  {numpy_time:.4f}s"
    )

    print(
        f"Numba GBM:  {numba_time:.4f}s"
    )

    print(
        f"Speedup: "
        f"{numpy_time / numba_time:.2f}x"
    )


if __name__ == "__main__":
    main()

heston = Heston(
    initial_price=100,
    initial_variance=0.04,
    risk_free_rate=0.05,
    kappa=2,
    theta=0.04,
    volatility_of_volatility=0.3,
    correlation=-0.7
)


start = time.time()

heston.simulate(
    time=1,
    steps=252,
    paths=100000,
    seed=42
)

numpy_heston_time = (
    time.time()
    -
    start
)


start = time.time()

simulate_heston_numba(
    initial_price=100,
    initial_variance=0.04,
    risk_free_rate=0.05,
    kappa=2,
    theta=0.04,
    volatility_of_volatility=0.3,
    correlation=-0.7,
    time=1,
    steps=252,
    paths=100000,
    seed=42
)

numba_heston_time = (
    time.time()
    -
    start
)


print("\nHeston Performance")
print("------------------")

print(
    f"NumPy Heston: {numpy_heston_time:.4f}s"
)

print(
    f"Numba Heston: {numba_heston_time:.4f}s"
)

print(
    f"Speedup: "
    f"{numpy_heston_time / numba_heston_time:.2f}x"
)