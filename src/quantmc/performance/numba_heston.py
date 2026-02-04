import numpy as np

from numba import njit, prange


@njit(parallel=True)
def _simulate_heston_kernel(
    S0,
    v0,
    r,
    kappa,
    theta,
    xi,
    rho,
    dt,
    z1,
    z2
):

    paths, steps = z1.shape

    prices = np.zeros(
        (paths, steps + 1)
    )

    variance = np.zeros(
        (paths, steps + 1)
    )


    for i in prange(paths):

        prices[i, 0] = S0
        variance[i, 0] = v0


        for t in range(1, steps + 1):

            previous_variance = variance[i, t - 1]


            w1 = z1[i, t - 1]

            w2 = (
                rho * z1[i, t - 1]
                +
                np.sqrt(
                    1 - rho ** 2
                )
                *
                z2[i, t - 1]
            )


            new_variance = (
                previous_variance
                +
                kappa
                *
                (
                    theta
                    -
                    previous_variance
                )
                *
                dt
                +
                xi
                *
                np.sqrt(
                    max(
                        previous_variance,
                        0
                    )
                )
                *
                np.sqrt(dt)
                *
                w2
            )


            if new_variance < 0:
                new_variance = 0


            variance[i, t] = new_variance


            prices[i, t] = (
                prices[i, t - 1]
                *
                np.exp(
                    (
                        r
                        -
                        0.5 * previous_variance
                    )
                    *
                    dt
                    +
                    np.sqrt(
                        previous_variance
                    )
                    *
                    np.sqrt(dt)
                    *
                    w1
                )
            )


    return prices



def simulate_heston_numba(
    initial_price,
    initial_variance,
    risk_free_rate,
    kappa,
    theta,
    volatility_of_volatility,
    correlation,
    time,
    steps,
    paths,
    seed=None
):

    if seed is not None:
        np.random.seed(seed)


    dt = time / steps


    z1 = np.random.normal(
        size=(
            paths,
            steps
        )
    )

    z2 = np.random.normal(
        size=(
            paths,
            steps
        )
    )


    return _simulate_heston_kernel(
        initial_price,
        initial_variance,
        risk_free_rate,
        kappa,
        theta,
        volatility_of_volatility,
        correlation,
        dt,
        z1,
        z2
    )