import numpy as np


def variance(samples):

    return np.var(
        samples,
        ddof=1
    )


def variance_reduction(
    baseline,
    improved
):
   

    return (
        1
        -
        variance(improved)
        /
        variance(baseline)
    )