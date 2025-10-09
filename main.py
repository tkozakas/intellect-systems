import os
import time
from typing import Literal

from datasets import load_dataset, Dataset
from google import genai
from google.api_core import exceptions
from google.genai import types
from matplotlib import pyplot as plt
from pydantic import BaseModel

HF_TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DATASET_NAME = "yale-nlp/FOLIO"
MODEL_NAME = "gemini-2.5-flash-lite"
DATA_SPLIT = "train"
NUM_SAMPLES = 5
REQUESTS_DELAY_SECONDS = 2


class LogicAnalysis(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False", "Unknown"]


def get_folio_dataset(token: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=DATA_SPLIT, token=token)


def evaluate_with_gemini_structured(
        client: genai.Client,
        request_config: types.GenerateContentConfig,
        premises: str,
        conclusion: str
) -> LogicAnalysis | None:
    prompt = f"""
        Based ONLY on the following premises, analyze the conclusion.
        Provide a brief reasoning and then state if the conclusion is logically True, False, or Unknown.

        Premises:
        {premises}

        Conclusion:
        "{conclusion}"
        """
    print(f"Premise: {premises}")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=request_config
        )
        print(f"  -> Model Conclusion: {response.parsed.final_answer}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return response.parsed
    except exceptions.GoogleAPICallError as e:
        print(f"  !! API Call Error: {e.message}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return None
    except Exception as e:
        print(f"  !! An unexpected error occurred: {e}")
        time.sleep(REQUESTS_DELAY_SECONDS)
        return None


def run_evaluation(
        client: genai.Client,
        request_config: types.GenerateContentConfig,
        dataset_sample: Dataset
) -> dict:
    nl_correct = 0
    fol_correct = 0
    nl_unknowns = 0
    fol_unknowns = 0

    for i, example in enumerate(dataset_sample):
        print(f"  -> Evaluating example {i + 1}/{len(dataset_sample)}...")
        correct_label = str(example["label"]).capitalize()

        # --- Natural Language Evaluation ---
        nl_analysis = evaluate_with_gemini_structured(
            client,
            request_config,
            premises=example["premises"],
            conclusion=example["conclusion"]
        )
        if nl_analysis:
            if nl_analysis.final_answer == correct_label:
                nl_correct += 1
            elif nl_analysis.final_answer == "Unknown":
                nl_unknowns += 1

        # --- First-Order Logic Evaluation ---
        fol_analysis = evaluate_with_gemini_structured(
            client,
            request_config,
            premises=example["premises-FOL"],
            conclusion=example["conclusion-FOL"]
        )
        if fol_analysis:
            if fol_analysis.final_answer == correct_label:
                fol_correct += 1
            elif fol_analysis.final_answer == "Unknown":
                fol_unknowns += 1

    return {
        "NL Correct": nl_correct,
        "FOL Correct": fol_correct,
        "NL Unknowns": nl_unknowns,
        "FOL Unknowns": fol_unknowns,
    }


def plot_results(evaluation_results, total_samples: int):
    categories = ['NL Correct', 'FOL Correct', 'NL Unknowns', 'FOL Unknowns']
    values = [evaluation_results[cat] for cat in categories]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=['blue', 'orange', 'gray', 'lightgray'])
    plt.ylim(0, total_samples)
    plt.ylabel('Number of Samples')
    plt.title('Evaluation Results on FOLIO Dataset')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, yval, ha='center', va='bottom')

    plt.show()


def main():
    hf_token = os.getenv(HF_TOKEN_ENV_VAR)
    gemini_api_key = os.getenv(GEMINI_API_KEY_ENV_VAR)
    if not hf_token or not gemini_api_key:
        raise ValueError(f"{HF_TOKEN_ENV_VAR} and {GEMINI_API_KEY_ENV_VAR} must be set.")

    print("Loading FOLIO dataset...")
    folio_dataset = get_folio_dataset(token=hf_token)
    print(f"Dataset loaded. Running evaluation on {NUM_SAMPLES} samples.")
    if folio_dataset:
        folio_sample = folio_dataset.select(range(NUM_SAMPLES))
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LogicAnalysis,
        )
        client = genai.Client(api_key=gemini_api_key)
        evaluation_results = run_evaluation(client, request_config, folio_sample)

        print(f"  Natural Language: {evaluation_results['NL Correct']} Correct, {evaluation_results['NL Unknowns']} Unknowns")
        print(f"  First-Order Logic: {evaluation_results['FOL Correct']} Correct, {evaluation_results['FOL Unknowns']} Unknowns")

        plot_results(evaluation_results, NUM_SAMPLES)


if __name__ == "__main__":
    main()
