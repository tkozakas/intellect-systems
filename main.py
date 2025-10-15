import os
import random

import pandas as pd

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Suppress gRPC warnings

import time
from typing import Literal

from datasets import load_dataset, Dataset
from google import genai
from google.genai import types
from matplotlib import pyplot as plt
from pydantic import BaseModel

# --- CONFIGURATION ---
HF_TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DATASET_NAME = "yale-nlp/FOLIO"
MODEL_NAME = "gemini-2.5-pro"
DATA_SPLIT = "train"
NUM_SAMPLES = 20
NUM_SAMPLES_FINAL = 50
MAX_RETRIES = 5
REQUESTS_DELAY_SECONDS = 5
CSV_OUTPUT_FILE = "final_evaluation_results.csv"


class LogicAnalysis(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False"]


class LogicAnalysisUncertain(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False", "Uncertain"]


def get_folio_dataset(token: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=DATA_SPLIT, token=token)


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

    max_retries = MAX_RETRIES
    base_delay = REQUESTS_DELAY_SECONDS

    for attempt in range(max_retries):
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

        except Exception:
            print(f"  Attempt {attempt + 1}/{max_retries}: API is unavailable or rate limit exceeded.")

            if attempt == max_retries - 1:
                print("    !! Max retries reached. Failing.")
                raise

            delay = (base_delay ** attempt) + random.uniform(0, 1)
            print(f"    -> Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

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


def run_final_evaluation_and_log(client, request_config, dataset_sample) -> pd.DataFrame:
    results_log = []
    for i, example in enumerate(dataset_sample):
        print(f"  -> Final Eval: Example {i + 1}/{len(dataset_sample)}...")
        correct_label = str(example["label"]).capitalize()

        nl_analysis = evaluate_with_gemini_structured(client, request_config, example["premises"], example["conclusion"], correct_label)
        fol_analysis = evaluate_with_gemini_structured(client, request_config, example["premises-FOL"], example["conclusion-FOL"],
                                                       correct_label)

        results_log.append({
            'story_id': example['story_id'],
            'example_id': example['example_id'],
            'premises_nl': example['premises'],
            'conclusion_nl': example['conclusion'],
            'premises_fol': example['premises-FOL'],
            'conclusion_fol': example['conclusion-FOL'],
            'correct_label': correct_label,
            'nl_model_answer': nl_analysis.final_answer if nl_analysis else 'ERROR',
            'nl_model_reasoning': nl_analysis.reasoning if nl_analysis else 'API_FAILURE',
            'fol_model_answer': fol_analysis.final_answer if fol_analysis else 'ERROR',
            'fol_model_reasoning': fol_analysis.reasoning if fol_analysis else 'API_FAILURE'
        })
    return pd.DataFrame(results_log)


def plot_answer_distribution(df: pd.DataFrame):
    labels = ['True', 'False', 'Uncertain']
    nl_counts = df['nl_model_answer'].value_counts().reindex(labels, fill_value=0)
    fol_counts = df['fol_model_answer'].value_counts().reindex(labels, fill_value=0)

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width / 2 for i in x], nl_counts, width, label='Natural Language')
    rects2 = ax.bar([i + width / 2 for i in x], fol_counts, width, label='First-Order Logic')

    ax.set_ylabel('Number of Answers')
    ax.set_title(f'Model Answer Distribution (N={len(df)})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()


def plot_final_accuracy(df: pd.DataFrame):
    total_samples = len(df)
    nl_correct = (df['nl_model_answer'] == df['correct_label']).sum()
    fol_correct = (df['fol_model_answer'] == df['correct_label']).sum()

    nl_accuracy = (nl_correct / total_samples) * 100 if total_samples > 0 else 0
    fol_accuracy = (fol_correct / total_samples) * 100 if total_samples > 0 else 0

    categories = ['Natural Language', 'First-Order Logic']
    accuracies = [nl_accuracy, fol_accuracy]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, color=['skyblue', 'lightgreen'])

    plt.ylim(0, 110)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Final model accuracy (N={total_samples})')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

    plt.show()


def plot_comparison(results: dict, total_samples: int, variable_name: str):
    x_values = list(results.keys())
    nl_accuracy = [(res["NL Correct"] / total_samples) * 100 for res in results.values()]
    fol_accuracy = [(res["FOL Correct"] / total_samples) * 100 for res in results.values()]

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, nl_accuracy, marker='o', linestyle='-', label='Natural Language Accuracy')
    plt.plot(x_values, fol_accuracy, marker='s', linestyle='--', label='First-Order Logic Accuracy')

    plt.ylim(0, 110)
    plt.xlabel(variable_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Reasoning Accuracy vs. {variable_name}\n(Model: {MODEL_NAME}, Samples: {total_samples})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i, x_val in enumerate(x_values):
        nl_val = nl_accuracy[i]
        fol_val = fol_accuracy[i]

        nl_vertical_alignment = 'top' if nl_val > 90 else 'bottom'
        fol_vertical_alignment = 'top'

        plt.text(x_val, nl_val, f" {nl_val:.1f}%", ha='center', va=nl_vertical_alignment)
        plt.text(x_val, fol_val, f" {fol_val:.1f}%", ha='center', va=fol_vertical_alignment)

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
    print(folio_sample.to_pandas())

    print(f"Dataset ready. Running evaluations on {NUM_SAMPLES} samples.")

    client = genai.Client(api_key=gemini_api_key)

    # # --- EXPERIMENT 1: TEMPERATURE COMPARISON ---
    # print("\n--- EXPERIMENT 1: TEMPERATURE COMPARISON ---")
    # temperatures_to_test = [0.0, 0.3, 0.7, 1.0]
    # results_by_temp = {}
    # for temp in temperatures_to_test:
    #     print(f"\n>> Testing Temperature: {temp}")
    #     request_config = types.GenerateContentConfig(
    #         response_mime_type="application/json",
    #         response_schema=LogicAnalysis,
    #         temperature=temp
    #     )
    #     results_by_temp[temp] = run_evaluation(client, request_config, folio_sample)
    #
    # print(results_by_temp)
    # plot_comparison(results_by_temp, NUM_SAMPLES, "Temperature (Creativity)")
    #
    # # --- EXPERIMENT 2: THINKING BUDGET COMPARISON ---
    # print("\n--- EXPERIMENT 2: THINKING BUDGET COMPARISON ---")
    # budgets_to_test = [int(8192/6), int(8192/4), int(8192/2), int(8192)]
    # results_by_budget = {}
    # for budget in budgets_to_test:
    #     print(f"\n>> Testing Thinking Budget: {budget}")
    #     request_config = types.GenerateContentConfig(
    #         response_mime_type="application/json",
    #         response_schema=LogicAnalysis,
    #         thinking_config=types.ThinkingConfig(thinking_budget=budget)
    #     )
    #     results_by_budget[budget] = run_evaluation(client, request_config, folio_sample)
    #
    # print(results_by_budget)
    # plot_comparison(results_by_budget, NUM_SAMPLES, "Thinking Budget")

    # --- EXPERIMENTAS 3: LAST ---
    print(f"\n--- EXPERIMENT 3: FINAL EVALUATION ON {NUM_SAMPLES_FINAL} SAMPLES ---")
    folio_sample_final = folio_dataset.select(range(NUM_SAMPLES_FINAL))

    final_request_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LogicAnalysisUncertain
    )

    results_df = run_final_evaluation_and_log(client, final_request_config, folio_sample_final)
    results_df.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"\nFinal evaluation results saved to {CSV_OUTPUT_FILE}")

    plot_answer_distribution(results_df)
    plot_final_accuracy(results_df)

    print("\n--- All Experiments Complete ---")


if __name__ == "__main__":
    main()