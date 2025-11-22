import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd
from datasets import Dataset, load_dataset
from openai import OpenAI
from pydantic import BaseModel

import config

os.environ["GRPC_VERBOSITY"] = "ERROR"


class LogicAnalysisUncertain(BaseModel):
    reasoning: str
    final_answer: Literal["True", "False", "Uncertain"]


def get_folio_dataset(token: str) -> Dataset:
    dataset = load_dataset(config.DATASET_NAME, split=config.DATA_SPLIT, token=token)
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Expected Dataset but got {type(dataset)}")
    return dataset


def evaluate_with_deepseek_structured(
    client: OpenAI,
    premises: str,
    conclusion: str,
    actual_answer: str,
    example_index: int = 0,
) -> LogicAnalysisUncertain | None:
    system_prompt = """You are a logic reasoning expert. The user will provide premises and a conclusion. 
Analyze whether the conclusion logically follows from the premises and output your response in JSON format.

EXAMPLE INPUT:
Premises: All humans are mortal. Socrates is a human.
Conclusion: Socrates is mortal.

EXAMPLE JSON OUTPUT:
{
    "reasoning": "The conclusion follows logically from the premises through a valid syllogism.",
    "final_answer": "True"
}

Output format:
{
    "reasoning": "<your step-by-step logical analysis>",
    "final_answer": "True" or "False"
}
"""

    user_prompt = f"""Based ONLY on the following premises, analyze the conclusion.
Provide a brief reasoning and then state if the conclusion is logically True, False or Uncertain.

Premises:
{premises}

Conclusion:
"{conclusion}"
"""

    print(f"  [Example {example_index}] Premise: {premises[:100]}...")

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.DEEPSEEK_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from API")
            response_json = json.loads(response_text)

            parsed_response = LogicAnalysisUncertain(
                reasoning=response_json.get("reasoning", ""),
                final_answer=response_json.get("final_answer", "Uncertain"),
            )

            print(
                f"    [Example {example_index}] -> Model Conclusion: {parsed_response.final_answer} (Actual: {actual_answer})"
            )
            time.sleep(config.REQUESTS_DELAY_SECONDS)
            return parsed_response

        except Exception as e:
            print(
                f"  [Example {example_index}] Attempt {attempt + 1}/{config.MAX_RETRIES}: API unavailable or rate limit exceeded. Error: {e}"
            )

            if attempt == config.MAX_RETRIES - 1:
                print(f"    [Example {example_index}] !! Max retries reached.")
                raise

            delay = (config.REQUESTS_DELAY_SECONDS**attempt) + random.uniform(0, 1)
            print(
                f"    [Example {example_index}] -> Retrying in {delay:.2f} seconds..."
            )
            time.sleep(delay)

    return None


def process_single_example(client: OpenAI, example: dict, index: int) -> dict:
    """Process a single example and return the result dictionary."""
    correct_label = str(example["label"]).capitalize()

    nl_analysis = evaluate_with_deepseek_structured(
        client, example["premises"], example["conclusion"], correct_label, index + 1
    )

    return {
        "story_id": example["story_id"],
        "example_id": example["example_id"],
        "premises_nl": example["premises"],
        "conclusion_nl": example["conclusion"],
        "premises_fol": example["premises-FOL"],
        "conclusion_fol": example["conclusion-FOL"],
        "correct_label": correct_label,
        "nl_model_answer": nl_analysis.final_answer if nl_analysis else "ERROR",
        "nl_model_reasoning": nl_analysis.reasoning if nl_analysis else "API_FAILURE",
        "fol_model_answer": "N/A",
        "fol_model_reasoning": "N/A (DeepSeek NL-only evaluation)",
        "index": index, 
    }


def run_final_evaluation_and_log(client, dataset_sample) -> pd.DataFrame:
    results_log = []

    print(
        f"Processing {len(dataset_sample)} examples with {config.MAX_WORKERS} parallel workers..."
    )

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_single_example, client, example, i): i
            for i, example in enumerate(dataset_sample)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results_log.append(result)
                print(
                    f"  ✓ Completed {len(results_log)}/{len(dataset_sample)} examples"
                )
            except Exception as e:
                print(f"  ✗ Example {index + 1} failed with error: {e}")
                example = dataset_sample[index]
                results_log.append(
                    {
                        "story_id": example["story_id"],
                        "example_id": example["example_id"],
                        "premises_nl": example["premises"],
                        "conclusion_nl": example["conclusion"],
                        "premises_fol": example["premises-FOL"],
                        "conclusion_fol": example["conclusion-FOL"],
                        "correct_label": str(example["label"]).capitalize(),
                        "nl_model_answer": "ERROR",
                        "nl_model_reasoning": f"EXCEPTION: {str(e)}",
                        "fol_model_answer": "N/A",
                        "fol_model_reasoning": "N/A (DeepSeek NL-only evaluation)",
                        "index": index,
                    }
                )

    results_log.sort(key=lambda x: x["index"])

    for result in results_log:
        result.pop("index", None)

    return pd.DataFrame(results_log)


def run_experiment():
    hf_token = os.getenv(config.HF_TOKEN_ENV_VAR)
    deepseek_api_key = os.getenv(config.DEEPSEEK_API_KEY_ENV_VAR)
    if not hf_token or not deepseek_api_key:
        raise ValueError(
            f"{config.HF_TOKEN_ENV_VAR} and {config.DEEPSEEK_API_KEY_ENV_VAR} must be set."
        )

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Loading and filtering FOLIO dataset...")
    loaded_folio_dataset = get_folio_dataset(token=hf_token)
    folio_dataset = loaded_folio_dataset.filter(
        lambda example: example["label"] != "Uncertain"
    )
    folio_sample_final = folio_dataset.select(range(config.NUM_SAMPLES_FINAL))

    client = OpenAI(api_key=deepseek_api_key, base_url=config.DEEPSEEK_BASE_URL)

    print(
        f"\n--- DEEPSEEK FINAL EVALUATION ON {config.NUM_SAMPLES_FINAL} SAMPLES (NL ONLY) ---"
    )
    results_df = run_final_evaluation_and_log(client, folio_sample_final)
    results_df.to_csv(config.DEEPSEEK_CSV_OUTPUT, index=False)
    print(f"\nFinal evaluation results saved to {config.DEEPSEEK_CSV_OUTPUT}")
    print("\n--- DeepSeek Experiment Complete ---")
