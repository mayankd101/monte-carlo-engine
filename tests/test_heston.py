from quantmc.models.heston import Heston


def test_heston_simulation_shape():

    model = Heston(
        initial_price=100,
        initial_variance=0.04,
        risk_free_rate=0.05,
        kappa=2,
        theta=0.04,
        volatility_of_volatility=0.3,
        correlation=-0.7
    )


    paths = model.simulate(
        time=1,
        steps=100,
        paths=1000,
        seed=42
    )


    assert paths.shape == (
        1000,
        101
    )


def test_heston_positive_prices():

    model = Heston(
        initial_price=100,
        initial_variance=0.04,
        risk_free_rate=0.05,
        kappa=2,
        theta=0.04,
        volatility_of_volatility=0.3,
        correlation=-0.7
    )


    paths = model.simulate(
        time=1,
        steps=100,
        paths=1000,
        seed=42
    )


    assert (paths > 0).all()