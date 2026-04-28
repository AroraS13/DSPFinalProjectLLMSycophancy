#!/usr/bin/env python3
"""Validate LLM judge agreement against hand labels (kappa gate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd


def _import_judge_bundle():
    try:
        from sycophancy_judge import compute_kappa, judge_dataframe, print_kappa_report  # type: ignore

        return judge_dataframe, compute_kappa, print_kappa_report
    except ImportError:
        from Judge import compute_kappa, judge_dataframe, print_kappa_report  # type: ignore

        return judge_dataframe, compute_kappa, print_kappa_report


def _extract_kappa(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("kappa", "cohen_kappa", "cohens_kappa"):
            if key in value:
                return float(value[key])
    if hasattr(value, "kappa"):
        return float(value.kappa)
    raise ValueError(
        "compute_kappa() returned an unsupported type; expected float/dict/object with kappa."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate judge with Cohen's kappa threshold.")
    parser.add_argument("--csv", required=True, help="CSV containing a human_label column.")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model name passed to judge_dataframe() (default: gpt-4o).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum acceptable kappa (default: 0.75).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "human_label" not in df.columns:
        raise ValueError("Input CSV must contain a 'human_label' column.")

    judge_dataframe, compute_kappa, print_kappa_report = _import_judge_bundle()
    judged = judge_dataframe(df.copy(), judge_model=args.judge_model, cache_path="judge_cache.json")
    if "label" not in judged.columns:
        raise ValueError("judge_dataframe() output must include a 'label' column.")

    # Prefer the DataFrame-style API used by Judge.py in this repo.
    # Fallback to older signature styles for compatibility.
    try:
        result = compute_kappa(judged, human_col="human_label", judge_col="label")
    except TypeError:
        result = compute_kappa(judged["human_label"], judged["label"])
    print_kappa_report(result)

    kappa = _extract_kappa(result)
    print(f"Kappa: {kappa:.4f} (threshold: {args.threshold:.2f})")
    if kappa < args.threshold:
        print("Kappa validation failed.")
        sys.exit(1)
    print("Kappa validation passed.")


if __name__ == "__main__":
    main()
