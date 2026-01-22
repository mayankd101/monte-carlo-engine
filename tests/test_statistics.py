import numpy as np

from quantmc.analytics.statistics import (
    standard_error,
    confidence_interval
)


def test_standard_error():
    samples = np.array([1, 2, 3, 4, 5])

    result = standard_error(samples)

    expected = np.std(samples, ddof=1) / np.sqrt(5)

    assert result == expected


def test_confidence_interval():
    result = confidence_interval(
        estimate=10,
        error=0.5,
        confidence=0.95
    )

    assert result == (9.02, 10.98)