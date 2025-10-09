import os
from typing import Literal

from datasets import load_dataset, Dataset
from google import genai
from google.genai import types
from matplotlib import pyplot as plt
from pydantic import BaseModel

HF_TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DATASET_NAME = "yale-nlp/FOLIO"
MODEL_NAME = "gemini-2.5-flash"
DATA_SPLIT = "train"
NUM_SAMPLES = 5


class LogicAnalysis(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False", "Unknown"]


def get_folio_dataset(token: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=DATA_SPLIT, token=token)


def evaluate_with_gemini_structured(client: genai.Client, request_config: types.GenerateContentConfig, premises: str, conclusion: str) -> LogicAnalysis:
    prompt = f"""
        Based ONLY on the following premises, analyze the conclusion.
        Provide a brief reasoning and then state if the conclusion is logically True or False.

        Premises:
        {premises}

        Conclusion:
        "{conclusion}"
        """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=request_config
    )

    return response.parsed

def run_evaluation(client: genai.Client, request_config: types.GenerateContentConfig, dataset_sample: Dataset) -> tuple[int, int]:
    nl_correct = 0
    fol_correct = 0

    for i, example in enumerate(dataset_sample):
        print(f"Evaluating example {i + 1}/{len(dataset_sample)}...")
        correct_label = str(example["label"]).capitalize()

        try:
            nl_analysis = evaluate_with_gemini_structured(
                client,
                request_config,
                premises=example["premises"],
                conclusion=example["conclusion"]
            )
            if nl_analysis.final_answer == correct_label:
                nl_correct += 1

            fol_analysis = evaluate_with_gemini_structured(
                client,
                request_config,
                premises=example["premises-FOL"],
                conclusion=example["conclusion-FOL"]
            )
            if fol_analysis.final_answer == correct_label:
                fol_correct += 1
        except Exception as e:
            print(f"An error occurred during evaluation for example {i+1}: {e}")

    return nl_correct, fol_correct


def plot_results(nl_correct: int, fol_correct: int, total_samples: int):
    labels = ['Natural Language Premises', 'First-Order Logic Premises']
    correct_counts = [nl_correct, fol_correct]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, correct_counts, color=['skyblue', 'lightgreen'])
    ax.set_ylabel('Number of Correct Answers')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, total_samples)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom', ha='center')
    plt.show()


def main():
    hf_token = os.getenv(HF_TOKEN_ENV_VAR)
    gemini_api_key = os.getenv(GEMINI_API_KEY_ENV_VAR)
    if not hf_token or not gemini_api_key:
        raise ValueError(f"{HF_TOKEN_ENV_VAR} and {GEMINI_API_KEY_ENV_VAR} must be set.")

    request_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LogicAnalysis,
    )
    client = genai.Client()
    folio_dataset = get_folio_dataset(token=hf_token)

    if folio_dataset:
        folio_sample = folio_dataset.select(range(NUM_SAMPLES))
        nl_correct, fol_correct = run_evaluation(client, request_config, folio_sample)

        print("\n--- Evaluation Complete ---")
        print(f"Natural Language Correct: {nl_correct}/{NUM_SAMPLES}")
        print(f"First-Order Logic Correct: {fol_correct}/{NUM_SAMPLES}")

        plot_results(nl_correct, fol_correct, NUM_SAMPLES)


if __name__ == "__main__":
    main()
