"""Submission formatting for the official Redrob challenge."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def to_submission_frame(ranked: pd.DataFrame, top_n: int = 100, require_exact: bool = True) -> pd.DataFrame:
    """Convert ranked challenge results to the exact official CSV schema."""

    if len(ranked) < top_n and require_exact:
        raise ValueError(f"Official submission requires {top_n} candidates; received {len(ranked)}.")

    output = ranked.head(top_n).copy()
    output = output.rename(
        columns={
            "Candidate ID": "candidate_id",
            "Rank": "rank",
            "Challenge Score": "score",
            "Reason": "reasoning",
        }
    )
    if "score" not in output.columns and "Overall Score" in output.columns:
        output["score"] = output["Overall Score"]
    if "candidate_id" not in output.columns and "candidate_id" in ranked.columns:
        output["candidate_id"] = ranked["candidate_id"]
    if "reasoning" not in output.columns and "Reason" in output.columns:
        output["reasoning"] = output["Reason"]

    output["rank"] = range(1, len(output) + 1)
    output["score"] = output["score"].astype(float).map(lambda value: f"{value:.4f}")
    output = output[SUBMISSION_COLUMNS]
    return output


def write_submission(ranked: pd.DataFrame, path: Path | str, top_n: int = 100, require_exact: bool = True) -> pd.DataFrame:
    """Write an official UTF-8 submission CSV."""

    submission = to_submission_frame(ranked, top_n=top_n, require_exact=require_exact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return submission
