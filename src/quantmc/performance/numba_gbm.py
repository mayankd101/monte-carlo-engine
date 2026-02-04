import numpy as np

from numba import njit


@njit
def _simulate_gbm_kernel(
    S0,
    r,
    sigma,
    dt,
    normals
):

    paths, steps = normals.shape

    prices = np.zeros(
        (paths, steps + 1)
    )

    for i in range(paths):

        prices[i, 0] = S0

        current_price = S0

        for j in range(steps):

            z = normals[i, j]

            increment = (
                (
                    r
                    -
                    0.5 * sigma ** 2
                )
                *
                dt
                +
                sigma
                *
                np.sqrt(dt)
                *
                z
            )

            current_price = (
                current_price
                *
                np.exp(increment)
            )

            prices[i, j + 1] = current_price


    return prices



def simulate_gbm_numba(
    initial_price,
    risk_free_rate,
    volatility,
    time,
    steps,
    paths,
    seed=None
):

    if seed is not None:
        np.random.seed(seed)


    dt = time / steps


    normals = np.random.normal(
        size=(
            paths,
            steps
        )
    )


    return _simulate_gbm_kernel(
        initial_price,
        risk_free_rate,
        volatility,
        dt,
        normals
    )