import time


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

    return results