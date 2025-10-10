import os

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Suppress gRPC warnings

import time
from typing import Literal

from datasets import load_dataset, Dataset
from google import genai
from google.api_core import exceptions
from google.genai import types
from matplotlib import pyplot as plt
from pydantic import BaseModel

# --- CONFIGURATION ---
HF_TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DATASET_NAME = "yale-nlp/FOLIO"
MODEL_NAME = "gemini-2.5-flash"
DATA_SPLIT = "train"
NUM_SAMPLES = 2
REQUESTS_DELAY_SECONDS = 1


class LogicAnalysis(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False"]


def get_folio_dataset(token: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=DATA_SPLIT, token=token, trust_remote_code=True)


def evaluate_with_gemini_structured(
        client: genai.Client,
        request_config: types.GenerateContentConfig,
        premises: str,
        conclusion: str,
        actual_answer: str
) -> LogicAnalysis | None:
    prompt = f"""
        Based ONLY on the following premises, analyze the conclusion.
        Provide a brief reasoning and then state if the conclusion is logically True or False.

        Premises:
        {premises}

        Conclusion:
        "{conclusion}"
        """
    print(f"  Premise: {premises[:100]}...")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=request_config
        )
        parsed_response = response.parsed
        print(f"    -> Model Conclusion: {parsed_response.final_answer} (Actual: {actual_answer})")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return parsed_response
    except exceptions.GoogleAPICallError as e:
        print(f"    !! API Call Error: {e.message}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return None
    except Exception as e:
        print(f"    !! An unexpected error occurred: {e}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return None


def run_evaluation(
        client: genai.Client,
        request_config: types.GenerateContentConfig,
        dataset_sample: Dataset,
) -> dict:
    nl_correct = 0
    fol_correct = 0

    for i, example in enumerate(dataset_sample):
        print(f"  -> Evaluating example {i + 1}/{len(dataset_sample)}...")
        correct_label = str(example["label"]).capitalize()

        # --- Natural Language Evaluation ---
        nl_analysis = evaluate_with_gemini_structured(
            client, request_config, example["premises"], example["conclusion"], correct_label
        )
        if nl_analysis and nl_analysis.final_answer == correct_label:
            nl_correct += 1

        # --- First-Order Logic Evaluation ---
        fol_analysis = evaluate_with_gemini_structured(
            client, request_config, example["premises-FOL"], example["conclusion-FOL"], correct_label
        )
        if fol_analysis and fol_analysis.final_answer == correct_label:
            fol_correct += 1

    return {"NL Correct": nl_correct, "FOL Correct": fol_correct}


def plot_comparison(results: dict, total_samples: int, variable_name: str):
    x_values = list(results.keys())
    nl_accuracy = [(res["NL Correct"] / total_samples) * 100 for res in results.values()]
    fol_accuracy = [(res["FOL Correct"] / total_samples) * 100 for res in results.values()]

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, nl_accuracy, marker='o', linestyle='-', label='Natural Language Accuracy')
    plt.plot(x_values, fol_accuracy, marker='s', linestyle='--', label='First-Order Logic Accuracy')

    plt.ylim(0, 100)
    plt.xlabel(variable_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Reasoning Accuracy vs. {variable_name}\n(Model: {MODEL_NAME}, Samples: {total_samples})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i, x_val in enumerate(x_values):
        plt.text(x_val, nl_accuracy[i] + 2, f"{nl_accuracy[i]:.1f}%", ha='center')
        plt.text(x_val, fol_accuracy[i] - 4, f"{fol_accuracy[i]:.1f}%", ha='center')

    plt.show()


def main():
    hf_token = os.getenv(HF_TOKEN_ENV_VAR)
    gemini_api_key = os.getenv(GEMINI_API_KEY_ENV_VAR)
    if not hf_token or not gemini_api_key:
        raise ValueError(f"{HF_TOKEN_ENV_VAR} and {GEMINI_API_KEY_ENV_VAR} must be set.")

    print("Loading and filtering FOLIO dataset...")
    loaded_folio_dataset = get_folio_dataset(token=hf_token)
    folio_dataset = loaded_folio_dataset.filter(lambda example: example['label'] != 'Uncertain')
    folio_sample = folio_dataset.select(range(NUM_SAMPLES))
    print(f"Dataset ready. Running evaluations on {NUM_SAMPLES} samples.")

    client = genai.Client(api_key=gemini_api_key)

    # --- EXPERIMENT 1: TEMPERATURE COMPARISON ---
    print("\n--- EXPERIMENT 1: TEMPERATURE COMPARISON ---")
    temperatures_to_test = [0.0, 0.3, 0.7, 1.0]
    results_by_temp = {}
    for temp in temperatures_to_test:
        print(f"\n>> Testing Temperature: {temp}")
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LogicAnalysis,
            temperature=temp
        )
        results_by_temp[temp] = run_evaluation(client, request_config, folio_sample)

    plot_comparison(results_by_temp, NUM_SAMPLES, "Temperature (Creativity)")

    # --- EXPERIMENT 2: THINKING BUDGET COMPARISON ---
    print("\n--- EXPERIMENT 2: THINKING BUDGET COMPARISON ---")
    budgets_to_test = [0, 200, 400, 800]
    results_by_budget = {}
    for budget in budgets_to_test:
        print(f"\n>> Testing Thinking Budget: {budget}")
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LogicAnalysis,
            thinking_config=types.ThinkingConfig(thinking_budget=budget)
        )
        results_by_budget[budget] = run_evaluation(client, request_config, folio_sample)

    plot_comparison(results_by_budget, NUM_SAMPLES, "Thinking Budget")

    print("\n--- All Experiments Complete ---")


if __name__ == "__main__":
    main()