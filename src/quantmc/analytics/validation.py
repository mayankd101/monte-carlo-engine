def compare_greeks(
    analytical,
    monte_carlo,
    time,
    steps,
    paths,
    seed=None
):

    results = {}

    results["delta"] = {
        "black_scholes": analytical.delta(),
        "monte_carlo": monte_carlo.delta(
            time,
            steps,
            paths,
            seed=seed
        )
    }

    results["gamma"] = {
        "black_scholes": analytical.gamma(),
        "monte_carlo": monte_carlo.gamma(
            time,
            steps,
            paths,
            seed=seed
        )
    }

    results["vega"] = {
        "black_scholes": analytical.vega(),
        "monte_carlo": monte_carlo.vega(
            time,
            steps,
            paths,
            seed=seed
        )
    }

    return results