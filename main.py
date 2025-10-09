import os
import time
from typing import Literal
import argparse

from datasets import load_dataset, Dataset
import google.generativeai as genai
from google.api_core import exceptions
from google.generativeai import types
from matplotlib import pyplot as plt
from pydantic import BaseModel

HF_TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DATASET_NAME = "yale-nlp/FOLIO"
MODEL_NAME = "gemini-2.5-pro"
DATA_SPLIT = "train"
NUM_SAMPLES = 5
REQUESTS_DELAY_SECONDS = 2


class LogicAnalysis(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False"]


def get_folio_dataset(token: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=DATA_SPLIT, token=token)


def evaluate_with_gemini_structured(
        client: genai.GenerativeModel,
        request_config: types.GenerationConfig,
        premises: str,
        conclusion: str,
        actual_answer: str
) -> LogicAnalysis | None:
    prompt = f"""
        Based ONLY on the following premises, analyze the conclusion.
        Provide a brief reasoning and then state if the conclusion is logically True, False

        Premises:
        {premises}

        Conclusion:
        "{conclusion}"
        """
    print(f"Premise: {premises[:500]}...")
    try:
        response = client.generate_content(
            contents=prompt,
            generation_config=request_config
        )
        parsed_response = LogicAnalysis.model_validate_json(response.text)
        print(f"  -> Model Conclusion: {parsed_response.final_answer} (Actual: {actual_answer})")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return parsed_response
    except exceptions.GoogleAPICallError as e:
        print(f"  !! API Call Error: {e.message}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return None
    except Exception as e:
        print(f"  !! An unexpected error occurred: {e}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return None


def run_evaluation(
        client: genai.GenerativeModel,
        request_config: types.GenerationConfig,
        dataset_sample: Dataset,
        evaluation_type: Literal["NL", "FOL"]
) -> dict:
    correct_count = 0

    premise_key = "premises" if evaluation_type == "NL" else "premises-FOL"
    conclusion_key = "conclusion" if evaluation_type == "NL" else "conclusion-FOL"

    print(f"\n--- Starting Evaluation for Type: {evaluation_type} ---")

    for i, example in enumerate(dataset_sample):
        print(f"  -> Evaluating example {i + 1}/{len(dataset_sample)}...")
        correct_label = str(example["label"]).capitalize()

        analysis = evaluate_with_gemini_structured(
            client,
            request_config,
            premises=example[premise_key],
            conclusion=example[conclusion_key],
            actual_answer=correct_label
        )
        if analysis:
            if analysis.final_answer == correct_label:
                correct_count += 1

    return {
        "Correct": correct_count,
    }


def plot_results(evaluation_results: dict, total_samples: int, evaluation_type: str):
    categories = ['Correct']
    values = [evaluation_results[cat] for cat in categories]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=['green', 'gray'])
    plt.ylim(0, total_samples)
    plt.ylabel('Number of Samples')
    plt.title(f"Evaluation Results for '{evaluation_type}' on FOLIO Dataset ({total_samples} samples)")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, yval, ha='center', va='bottom')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate logical reasoning using the Gemini API on the FOLIO dataset.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["NL", "FOL"],
        help="The type of evaluation to run: 'NL' for natural language or 'FOL' for first-order logic."
    )
    args = parser.parse_args()
    evaluation_type = args.type

    hf_token = os.getenv(HF_TOKEN_ENV_VAR)
    gemini_api_key = os.getenv(GEMINI_API_KEY_ENV_VAR)
    if not hf_token or not gemini_api_key:
        raise ValueError(f"Environment variables {HF_TOKEN_ENV_VAR} and {GEMINI_API_KEY_ENV_VAR} must be set.")

    genai.configure(api_key=gemini_api_key)

    print("Loading FOLIO dataset...")
    folio_dataset = get_folio_dataset(token=hf_token)
    print(f"Dataset loaded. Running evaluation on {NUM_SAMPLES} samples.")

    if folio_dataset:
        folio_sample = folio_dataset.select(range(NUM_SAMPLES))

        request_config = types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=LogicAnalysis,
        )
        client = genai.GenerativeModel(MODEL_NAME)

        evaluation_results = run_evaluation(client, request_config, folio_sample, evaluation_type)

        print("\n--- Evaluation Complete ---")
        print(f"Results for '{evaluation_type}' evaluation:")
        print(f"  Correct: {evaluation_results['Correct']}/{NUM_SAMPLES}")

        plot_results(evaluation_results, NUM_SAMPLES, evaluation_type)


if __name__ == "__main__":
    main()
