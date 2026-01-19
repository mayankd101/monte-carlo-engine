from quantmc.models.gbm import GBM


def test_antithetic_paths():

    model = GBM(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.2
    )

    paths = model.simulate(
        time=1,
        steps=10,
        paths=100,
        seed=42,
        antithetic=True
    )


    assert paths.shape == (
        100,
        11
    )