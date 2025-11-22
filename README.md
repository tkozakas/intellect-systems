# Intellect Systems - Logic Reasoning Experiments

Evaluate logic reasoning capabilities of LLMs on the FOLIO dataset using Natural Language and First-Order Logic.

## Quick Start
1. Set env variables `HF_TOKEN`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`
2. Do the thing
```bash
pip install -r requirements.txt

# Run Gemini experiment (NL + FOL)
python main.py --experiment gemini

# Run DeepSeek experiment (NL only)
python main.py --experiment deepseek
```
