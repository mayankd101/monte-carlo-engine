import time
from quantmc.analytics.plots import (
    plot_price_convergence,
    plot_error_convergence
)


def convergence_analysis(
    pricer,
    time_to_expiry,
    steps,
    path_counts,
    seed=None
):
  

    results = {}

    for paths in path_counts:

        start = time.time()

        stats = pricer.price_with_statistics(
            time=time_to_expiry,
            steps=steps,
            paths=paths,
            seed=seed
        )

        runtime = time.time() - start

        results[paths] = {
            "price": stats["price"],
            "standard_error": stats["standard_error"],
            "confidence_interval": stats["confidence_interval"],
            "runtime": runtime
        }

    plot_price_convergence(results)
    plot_error_convergence(results)
    return results

