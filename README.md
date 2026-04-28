# DSP Final Project: LLM Sycophancy Pipeline

## Dataset

The raw conversations live in `Data/`, one CSV per chatbot:

- `Chatbot_Prompt_Responses_Log - ChatGPT.csv`
- `Chatbot_Prompt_Responses_Log - Claude.csv`
- `Chatbot_Prompt_Responses_Log - Deepseek.csv`
- `Chatbot_Prompt_Responses_Log - Gemini.csv`

Each CSV has 5 prompts × 3 categories (`False Presumptions`, `Unethical Delusion`,
`Psychological Delusion`) × 6 turns (`Objective`, `Subjective`, `Reprompt 1`–`Reprompt 4`).
`DataLoader.parse_chatbot_log` reshapes one of these CSVs into a tidy long-format
DataFrame with columns `[prompt_num, category, turn, prompt, response]`.

## Pipeline

This repository supports a three-step workflow:

1. Hand-label a 30-row sample and validate the LLM judge
2. Run the full parser -> judge -> scorer pipeline
3. Inspect generated metrics artifacts

### 1) Hand-label 30 rows -> validate judge

The `test_scorer.ipynb` notebook walks you through:

1. Building the master annotation sheet `Data/annotations_labels.csv` (one row per
   non-Objective `(model, prompt_num, category, turn)`).
2. Drawing a fixed random sample of 30 rows into `Data/annotations_sample_30.csv`
   via `init_random_sample(n=30, seed=42)`.
3. Labeling those 30 rows interactively with `interactive_label_sample()` (or the
   non-interactive `label_next('V'|'N'|'C')` cursor flow). Allowed labels are
   `VALIDATING`, `NEUTRAL`, `CHALLENGING`.
4. Saving labels back to `Data/annotations_labels.csv`.

Once the 30 rows are labeled, run the kappa gate. Prepare a CSV containing the
long-format columns and a `human_label` column (your 30 hand labels), then run:

```bash
python validate_judge.py --csv Data/your_30_row_human_labeled.csv --judge-model gpt-4o
```

The script exits non-zero if Cohen's kappa is below `0.75`.

### 2) Run full pipeline

From the raw chatbot log CSVs (the model name is inferred from each filename):

```bash
python run_pipeline.py --csv-dir Data --judge-model gpt-4o
```

Or by passing CSVs explicitly:

```bash
python run_pipeline.py \
  --csv "Data/Chatbot_Prompt_Responses_Log - ChatGPT.csv" \
  --csv "Data/Chatbot_Prompt_Responses_Log - Claude.csv" \
  --csv "Data/Chatbot_Prompt_Responses_Log - Deepseek.csv" \
  --csv "Data/Chatbot_Prompt_Responses_Log - Gemini.csv" \
  --judge-model gpt-4o
```

This writes:

- `labeled.csv`
- `metrics_per_conversation.csv`
- `metrics_aggregated.csv`
- `transitions.csv`

To re-score an already-labeled long-format CSV (no API call):

```bash
python run_pipeline.py --csv Data/already_labeled.csv --skip-judge
```

### 3) Inspect outputs

Review:

- Per-conversation metrics in `metrics_per_conversation.csv`
- Grouped metrics in `metrics_aggregated.csv`
- Sankey transition data in `transitions.csv`

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `pandas`, `openai>=1.0`, and `scikit-learn`.
