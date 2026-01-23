import argparse
import importlib


EXAMPLES = {
    "simulate-gbm": "quantmc.examples.simulate_gbm",
    "price-option": "quantmc.examples.price_option",
    "compare-pricers": "quantmc.examples.compare_pricers",
    "calculate-greeks": "quantmc.examples.calculate_greeks",
    "variance-reduction": "quantmc.examples.variance_reduction",
    "convergence": "quantmc.examples.convergence",
}


def main():
    parser = argparse.ArgumentParser(
        description="Quant Monte Carlo Engine CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    example_parser = subparsers.add_parser(
        "example",
        help="Run example scripts"
    )

    example_parser.add_argument(
        "name",
        choices=EXAMPLES.keys()
    )

    args = parser.parse_args()

    if args.command == "example":
        module = importlib.import_module(
            EXAMPLES[args.name]
        )

        if hasattr(module, "main"):
            module.main()
        else:
            print(
                f"{args.name} executed successfully"
            )
    else:
        parser.print_help()