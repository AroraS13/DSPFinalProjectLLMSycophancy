"""
Parse the Chatbot Prompt/Response log CSV into a tidy long-format DataFrame.

The source CSV has a nested layout:
  - Row 0: three category headers (False Presumption, Unethical Delusion,
    Psychological Delusion), each spanning 2 columns (prompt column + response column).
  - Each prompt group spans 10 rows, alternating label rows
    ("Objective Prompt", "Subjective Prompt", "Reprompt 1/2/3") and content rows.
  - Trailing rows after the last prompt contain experiment metadata, which we skip.

Output: one row per (prompt_num, category, turn) with columns
  [prompt_num, category, turn, prompt, response]
"""

from pathlib import Path
import pandas as pd

# The 5 turns in order, matching the 5 label/content pairs inside each 10-row block.
TURNS = ["Objective", "Subjective", "Reprompt 1", "Reprompt 2", "Reprompt 3"]

# Each category occupies 2 columns: (prompt_col_index, response_col_index).
# Column 0 holds the prompt number, so category columns start at index 1.
CATEGORY_COLS = {
    "False Presumption":     (1, 2),
    "Unethical Delusion":    (3, 4),
    "Psychological Delusion": (5, 6),
}

ROWS_PER_PROMPT = 10  # 5 turns x 2 rows (label row + content row) per turn


def parse_chatbot_log(csv_path: str | Path) -> pd.DataFrame:
    """Parse the nested chatbot log CSV into a tidy long-format DataFrame."""
    # Read with no header; we'll handle the multi-row header ourselves.
    raw = pd.read_csv(csv_path, header=None, dtype=str)

    # Find rows that mark the start of a prompt group (col 0 has a numeric value).
    prompt_start_rows = [
        (int(val), idx)
        for idx, val in raw[0].items()
        if pd.notna(val) and str(val).strip().isdigit()
    ]

    records = []
    for prompt_num, start_row in prompt_start_rows:
        # Within a 10-row block, content rows are at offsets 1, 3, 5, 7, 9
        # (the label rows sit at offsets 0, 2, 4, 6, 8).
        for turn_idx, turn_name in enumerate(TURNS):
            content_row = start_row + 1 + turn_idx * 2
            if content_row >= len(raw):
                break
            for category, (p_col, r_col) in CATEGORY_COLS.items():
                prompt_text = raw.iat[content_row, p_col]
                response_text = raw.iat[content_row, r_col]
                records.append({
                    "prompt_num": prompt_num,
                    "category": category,
                    "turn": turn_name,
                    "prompt": prompt_text if pd.notna(prompt_text) else None,
                    "response": response_text if pd.notna(response_text) else None,
                })

    df = pd.DataFrame.from_records(records)

    # Enforce a useful category order for grouping / plotting later.
    df["category"] = pd.Categorical(
        df["category"], categories=list(CATEGORY_COLS), ordered=True
    )
    df["turn"] = pd.Categorical(df["turn"], categories=TURNS, ordered=True)

    return df.sort_values(["prompt_num", "category", "turn"]).reset_index(drop=True)


if __name__ == "__main__":
    csv_path = "/mnt/user-data/uploads/Chatbot_Prompt_Responses_Log_-_ChatGPT.csv"
    df = parse_chatbot_log(csv_path)

    print(f"Parsed {len(df)} rows "
          f"({df['prompt_num'].nunique()} prompts × "
          f"{df['category'].nunique()} categories × "
          f"{df['turn'].nunique()} turns)")
    print()
    print("Columns:", list(df.columns))
    print()
    print("Sample (first 6 rows, truncated text):")
    preview = df.head(6).copy()
    for col in ("prompt", "response"):
        preview[col] = preview[col].str.slice(0, 70).fillna("") + "…"
    print(preview.to_string(index=False))
    print()
    print("Row counts per category:")
    print(df["category"].value_counts().to_string())