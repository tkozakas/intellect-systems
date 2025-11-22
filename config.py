import os

HF_TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEEPSEEK_API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

DATASET_NAME = "yale-nlp/FOLIO"
DATA_SPLIT = "train"
NUM_SAMPLES_FINAL = 2
MAX_RETRIES = 5
REQUESTS_DELAY_SECONDS = 5

OUTPUT_DIR = "outputs"

GEMINI_MODEL_NAME = "gemini-2.5-pro"
GEMINI_CSV_OUTPUT = os.path.join(OUTPUT_DIR, "final_evaluation_results_gemini.csv")

DEEPSEEK_MODEL_NAME = "deepseek-reasoner"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CSV_OUTPUT = os.path.join(OUTPUT_DIR, "final_evaluation_results_deepseek.csv")
