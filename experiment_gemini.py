import os
import random
import time
from typing import Literal

import pandas as pd
from datasets import Dataset, load_dataset
from google import genai
from google.genai import types
from pydantic import BaseModel

import config

os.environ["GRPC_VERBOSITY"] = "ERROR"


class LogicAnalysisUncertain(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False", "Uncertain"]


def get_folio_dataset(token: str) -> Dataset:
    dataset = load_dataset(config.DATASET_NAME, split=config.DATA_SPLIT, token=token)
    return dataset


def evaluate_with_gemini_structured(
    client: genai.Client,
    request_config: types.GenerateContentConfig,
    premises: str,
    conclusion: str,
    actual_answer: str,
) -> LogicAnalysisUncertain | None:
    prompt = f"""Based ONLY on the following premises, analyze the conclusion. Provide a brief reasoning and then state if the conclusion is logically True, False, or Unknown.

Premises:
{premises}

Conclusion:
"{conclusion}
"""
    print(f"  Premise: {premises[:100]}...")

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL_NAME, contents=prompt, config=request_config
            )
            parsed_response = response.parsed
            print(
                f"    -> Model Conclusion: {parsed_response.final_answer} (Actual: {actual_answer})"
            )
            time.sleep(config.REQUESTS_DELAY_SECONDS)
            return parsed_response

        except Exception:
            print(
                f"  Attempt {attempt + 1}/{config.MAX_RETRIES}: API unavailable or rate limit exceeded."
            )

            if attempt == config.MAX_RETRIES - 1:
                print("    !! Max retries reached.")
                raise

            delay = (config.REQUESTS_DELAY_SECONDS**attempt) + random.uniform(0, 1)
            print(f"    -> Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    return None


def run_final_evaluation_and_log(
    client, request_config, dataset_sample
) -> pd.DataFrame:
    results_log = []
    for i, example in enumerate(dataset_sample):
        print(f"  -> Final Eval: Example {i + 1}/{len(dataset_sample)}...")
        correct_label = str(example["label"]).capitalize()

        nl_analysis = evaluate_with_gemini_structured(
            client,
            request_config,
            example["premises"],
            example["conclusion"],
            correct_label,
        )
        fol_analysis = evaluate_with_gemini_structured(
            client,
            request_config,
            example["premises-FOL"],
            example["conclusion-FOL"],
            correct_label,
        )

        results_log.append(
            {
                "story_id": example["story_id"],
                "example_id": example["example_id"],
                "premises_nl": example["premises"],
                "conclusion_nl": example["conclusion"],
                "premises_fol": example["premises-FOL"],
                "conclusion_fol": example["conclusion-FOL"],
                "correct_label": correct_label,
                "nl_model_answer": nl_analysis.final_answer if nl_analysis else "ERROR",
                "nl_model_reasoning": nl_analysis.reasoning
                if nl_analysis
                else "API_FAILURE",
                "fol_model_answer": fol_analysis.final_answer
                if fol_analysis
                else "ERROR",
                "fol_model_reasoning": fol_analysis.reasoning
                if fol_analysis
                else "API_FAILURE",
            }
        )
    return pd.DataFrame(results_log)


def run_experiment(example_id: str | None = None):
    hf_token = os.getenv(config.HF_TOKEN_ENV_VAR)
    gemini_api_key = os.getenv(config.GEMINI_API_KEY_ENV_VAR)
    if not hf_token or not gemini_api_key:
        raise ValueError(
            f"{config.HF_TOKEN_ENV_VAR} and {config.GEMINI_API_KEY_ENV_VAR} must be set."
        )

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Loading FOLIO dataset...")
    loaded_folio_dataset = get_folio_dataset(token=hf_token)

    if example_id:
        # Search entire dataset for specific example_id
        folio_sample_final = loaded_folio_dataset.filter(
            lambda example: str(example["example_id"]) == str(example_id)
        )
        if len(folio_sample_final) == 0:
            raise ValueError(f"No example found with example_id: {example_id}")
        print(
            f"Found example with ID: {example_id} (label: {folio_sample_final[0]['label']})"
        )
    else:
        # Filter out Uncertain labels only when running all samples
        folio_dataset = loaded_folio_dataset.filter(
            lambda example: example["label"] != "Uncertain"
        )
        folio_sample_final = folio_dataset.select(range(config.NUM_SAMPLES_FINAL))

    client = genai.Client(api_key=gemini_api_key)

    sample_count = len(folio_sample_final)
    print(f"\n--- GEMINI FINAL EVALUATION ON {sample_count} SAMPLE(S) ---")
    final_request_config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=LogicAnalysisUncertain
    )

    results_df = run_final_evaluation_and_log(
        client, final_request_config, folio_sample_final
    )

    if example_id:
        # Output only model answers and reasoning
        for _, row in results_df.iterrows():
            print(f"\nNL Answer: {row['nl_model_answer']}")
            print(f"NL Reasoning:\n{row['nl_model_reasoning']}")
            print(f"\nFOL Answer: {row['fol_model_answer']}")
            print(f"FOL Reasoning:\n{row['fol_model_reasoning']}\n")
    else:
        results_df.to_csv(config.GEMINI_CSV_OUTPUT, index=False)
        print(f"\nFinal evaluation results saved to {config.GEMINI_CSV_OUTPUT}")

        nl_correct = (results_df['correct_label'] == results_df['nl_model_answer']).sum()
        fol_correct = (results_df['correct_label'] == results_df['fol_model_answer']).sum()
        nl_accuracy = (nl_correct / len(results_df)) * 100
        fol_accuracy = (fol_correct / len(results_df)) * 100
        print(f"\nNL ACCURACY:  {nl_correct}/{len(results_df)} ({nl_accuracy:.2f}%)")
        print(f"FOL ACCURACY: {fol_correct}/{len(results_df)} ({fol_accuracy:.2f}%)")
