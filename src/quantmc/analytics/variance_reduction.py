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
        1 - variance(improved) / variance(baseline)
    )


def control_variate_adjustment(
    samples,
    control_samples,
    expected_control
):
    

    covariance = np.cov(
        samples,
        control_samples,
        ddof=1
    )[0, 1]

    control_variance = np.var(
        control_samples,
        ddof=1
    )

    coefficient = (
        covariance
        /
        control_variance
    )

    adjusted_samples = (
        samples
        -
        coefficient
        *
        (
            control_samples
            -
            expected_control
        )
    )

    return adjusted_samples