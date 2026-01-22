import numpy as np


def standard_error(samples):
    """
    Estimate Monte Carlo standard error
    """
    return np.std(samples, ddof=1) / np.sqrt(len(samples))


def confidence_interval(
    estimate,
    error,
    confidence=0.95
):
    """
    Normal approximation confidence interval
    """

    z = 1.96 if confidence == 0.95 else 1.645

    return (
        estimate - z * error,
        estimate + z * error
    )