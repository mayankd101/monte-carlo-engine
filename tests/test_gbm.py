import numpy as np

from quantmc.models.gbm import GBM


def test_gbm_shape():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    paths = model.simulate(
        time=1,
        steps=252,
        paths=100,
        seed=42
    )

    assert paths.shape == (100, 253)


def test_initial_price():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    paths = model.simulate(
        time=1,
        steps=252,
        paths=10,
        seed=42
    )

    assert np.all(paths[:, 0] == 100)