import matplotlib.pyplot as plt


def plot_price_convergence(results):

    paths = list(results.keys())

    prices = [
        results[p]["price"]
        for p in paths
    ]


    plt.figure(figsize=(8, 5))

    plt.plot(
        paths,
        prices,
        marker="o"
    )

    plt.xscale("log")

    plt.xlabel(
        "Number of Paths"
    )

    plt.ylabel(
        "Estimated Price"
    )

    plt.title(
        "Monte Carlo Price Convergence"
    )

    plt.grid(True)

    plt.savefig(
        "price_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_error_convergence(results):

    paths = list(results.keys())

    errors = [
        results[p]["standard_error"]
        for p in paths
    ]


    plt.figure(figsize=(8, 5))

    plt.plot(
        paths,
        errors,
        marker="o"
    )

    plt.xscale("log")
    plt.yscale("log")


    plt.xlabel(
        "Number of Paths"
    )

    plt.ylabel(
        "Standard Error"
    )

    plt.title(
        "Monte Carlo Error Convergence"
    )

    plt.grid(True)

    plt.savefig(
        "error_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()