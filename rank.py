"""CLI for producing an official Redrob challenge submission.

Example:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from services.challenge_adapter import load_official_candidates
from services.challenge_ranker import ChallengeRankingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank official Redrob challenge candidates.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl, candidates.jsonl.gz, or sample JSON.")
    parser.add_argument("--out", required=True, help="Output CSV path, e.g. submission.csv.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output. Official submission requires 100.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit for sampling candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    candidates = load_official_candidates(args.candidates, limit=args.limit)
    service = ChallengeRankingService()
    ranked = service.rank(candidates, top_n=args.top_n)
    require_exact = args.top_n == 100 and len(candidates) >= 100
    service.write_submission(ranked, args.out, top_n=args.top_n, require_exact=require_exact)
    elapsed = time.perf_counter() - started
    logger.info("Wrote {} ranked candidates to {} in {:.2f}s", len(ranked), args.out, elapsed)


if __name__ == "__main__":
    main()
