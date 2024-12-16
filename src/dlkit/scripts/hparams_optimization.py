from dlkit.scripts.hparams import optimize
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization using Optuna and MLFlow."
    )
    parser.add_argument(
        "config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    optimize(args.config)


if __name__ == "__main__":
    main()
