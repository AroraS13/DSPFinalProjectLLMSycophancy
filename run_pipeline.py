#!/usr/bin/env python3
"""End-to-end sycophancy pipeline orchestrator.

Stages:
1) Parse raw chatbot CSV (or load pre-labeled long-format CSV with --skip-judge)
2) Label with LLM judge (unless --skip-judge)
3) Score, aggregate, and export transition counts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd


def _import_parse_chatbot_log() -> Callable:
    try:
        from parse_chatbot_log import parse_chatbot_log  # type: ignore

        return parse_chatbot_log
    except ImportError:
        from DataLoader import parse_chatbot_log  # type: ignore

        return parse_chatbot_log


def _import_scoring_fns():
    try:
        from sycophancy_scoring import aggregate, score_dataset, transition_counts  # type: ignore

        return score_dataset, aggregate, transition_counts
    except ImportError:
        from Scorer import aggregate, score_dataset, transition_counts  # type: ignore

        return score_dataset, aggregate, transition_counts


def _import_judge_dataframe() -> Callable:
    try:
        from sycophancy_judge import judge_dataframe  # type: ignore

        return judge_dataframe
    except ImportError:
        from Judge import judge_dataframe  # type: ignore

        return judge_dataframe


def _require_columns(df: pd.DataFrame, needed: list[str], context: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _build_aggregates(metrics_df: pd.DataFrame, aggregate_fn: Callable) -> pd.DataFrame:
    by_category = aggregate_fn(metrics_df, by=["category"]).assign(grouping="category")
    by_model = aggregate_fn(metrics_df, by=["model"]).assign(grouping="model")
    by_model_category = aggregate_fn(metrics_df, by=["model", "category"]).assign(
        grouping="model_category"
    )
    return pd.concat([by_category, by_model, by_model_category], ignore_index=True, sort=False)


_CSV_PREFIX = "Chatbot_Prompt_Responses_Log - "


def _model_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith(_CSV_PREFIX):
        return stem[len(_CSV_PREFIX):].strip()
    return stem.strip()


def _resolve_csv_inputs(args: argparse.Namespace) -> list[Path]:
    if args.csv_dir:
        data_dir = Path(args.csv_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"--csv-dir not found or not a directory: {data_dir}")
        paths = sorted(data_dir.glob(f"{_CSV_PREFIX}*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No chatbot log CSVs matching '{_CSV_PREFIX}*.csv' found in {data_dir}"
            )
        return paths

    csv_paths = [Path(p) for p in (args.csv or [])]
    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}")
    if not csv_paths:
        raise ValueError("Provide --csv (one or more CSVs) or --csv-dir.")
    return csv_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parser -> judge -> scorer pipeline.")
    parser.add_argument(
        "--csv",
        action="append",
        help="Input CSV path. Pass multiple times to combine chatbot logs (e.g. ChatGPT, Claude, ...).",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help=(
            "Directory containing 'Chatbot_Prompt_Responses_Log - <Model>.csv' files. "
            "If set, all matching CSVs are parsed and concatenated."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model name passed to judge_dataframe() (default: gpt-4o).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Number of concurrent judge API calls (default: 8). Raise if you "
            "have a high TPM tier; lower if you keep hitting rate limits."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Override 'model' column for parsed rows. By default, the model name is "
            "inferred from each CSV filename."
        ),
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Skip judge stage and treat --csv (single path) as already-labeled "
            "long-format data."
        ),
    )
    args = parser.parse_args()

    score_dataset, aggregate, transition_counts = _import_scoring_fns()

    if args.skip_judge:
        if not args.csv or len(args.csv) != 1:
            raise ValueError("--skip-judge expects exactly one --csv pointing at labeled data.")
        labeled_df = pd.read_csv(args.csv[0])
        _require_columns(
            labeled_df,
            ["model", "prompt_num", "category", "turn", "prompt", "response", "label"],
            "Labeled CSV (--skip-judge)",
        )
    else:
        parse_chatbot_log = _import_parse_chatbot_log()
        judge_dataframe = _import_judge_dataframe()

        csv_paths = _resolve_csv_inputs(args)

        frames = []
        for path in csv_paths:
            df = parse_chatbot_log(path)
            _require_columns(
                df,
                ["prompt_num", "category", "turn", "prompt", "response"],
                f"Parsed raw dataframe ({path.name})",
            )
            df["model"] = args.model_name or _model_name_from_path(path)
            frames.append(df)
        raw_df = pd.concat(frames, ignore_index=True)

        # Objective is baseline-only and excluded from labeling/scoring.
        working = raw_df[raw_df["turn"].astype(str) != "Objective"].copy()

        labeled_df = judge_dataframe(
            working,
            judge_model=args.judge_model,
            cache_path="judge_cache.json",
            max_workers=args.max_workers,
        )
        _require_columns(labeled_df, ["label"], "judge_dataframe output")

    labeled_df.to_csv("labeled.csv", index=False)

    metrics_per_conversation = score_dataset(labeled_df)
    metrics_per_conversation.to_csv("metrics_per_conversation.csv", index=False)

    metrics_aggregated = _build_aggregates(metrics_per_conversation, aggregate)
    metrics_aggregated.to_csv("metrics_aggregated.csv", index=False)

    transitions = transition_counts(labeled_df)
    transitions.to_csv("transitions.csv", index=False)

    print("Wrote labeled.csv")
    print("Wrote metrics_per_conversation.csv")
    print("Wrote metrics_aggregated.csv")
    print("Wrote transitions.csv")


if __name__ == "__main__":
    main()
