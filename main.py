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
    parser.add_argument(
        "--example-id",
        type=str,
        default=None,
        help="Specific example_id from FOLIO dataset to run (e.g., 'test-1234'). If not provided, runs all samples.",
    )

    args = parser.parse_args()

    if args.experiment == "gemini":
        if args.example_id:
            print(
                f"Running Gemini experiment (NL + FOL) for example_id: {args.example_id}..."
            )
        else:
            print("Running Gemini experiment (NL + FOL)...")
        experiment_gemini.run_experiment(example_id=args.example_id)
    elif args.experiment == "deepseek":
        if args.example_id:
            print(
                f"Running DeepSeek experiment (NL only) for example_id: {args.example_id}..."
            )
        else:
            print("Running DeepSeek experiment (NL only)...")
        experiment_deepseek.run_experiment(example_id=args.example_id)
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)


if __name__ == "__main__":
    main()
