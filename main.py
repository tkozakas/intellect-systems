import sys
import argparse

import experiment_gemini
import experiment_deepseek


def main():
    parser = argparse.ArgumentParser(
        description="Run logic reasoning experiments with different models."
    )
    parser.add_argument(
        "--experiment",
        choices=["gemini", "deepseek"],
        default="gemini",
        help="Which experiment to run. Default is 'gemini'.",
    )

    args = parser.parse_args()

    if args.experiment == "gemini":
        print("Running Gemini experiment (NL + FOL)...")
        experiment_gemini.run_experiment()
    elif args.experiment == "deepseek":
        print("Running DeepSeek experiment (NL only)...")
        experiment_deepseek.run_experiment()
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)


if __name__ == "__main__":
    main()
