"""Candidate ranking orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from services.challenge_adapter import is_official_candidates_frame, load_official_candidates
from services.challenge_ranker import ChallengeRankingService
from services.embedding_service import EmbeddingService, build_index
from services.explainability import generate_reason
from services.preprocessing import clean_candidates, clean_jobs, job_from_text
from services.scoring import calculate_scores
from utils.config import Settings, settings


class RankingService:
    """Coordinates data loading, semantic search, scoring, and output generation."""

    def __init__(self, config: Settings = settings, embedding_service: EmbeddingService | None = None) -> None:
        self.config = config
        self.config.ensure_directories()
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=config.model_name,
            cache_path=config.candidate_cache_path,
        )

    def load_candidates(self, path: Path | str | None = None) -> pd.DataFrame:
        candidate_path = Path(path) if path else self.config.data_dir / "candidates.csv"
        logger.info("Loading candidates from {}", candidate_path)
        suffixes = "".join(candidate_path.suffixes).lower()
        if suffixes.endswith((".jsonl", ".jsonl.gz")) or candidate_path.suffix.lower() == ".json":
            return load_official_candidates(candidate_path)
        return clean_candidates(pd.read_csv(candidate_path))

    def load_jobs(self, path: Path | str | None = None) -> pd.DataFrame:
        job_path = Path(path) if path else self.config.data_dir / "jobs.csv"
        logger.info("Loading jobs from {}", job_path)
        return clean_jobs(pd.read_csv(job_path))

    def rank(
        self,
        job: pd.Series | str,
        candidates: pd.DataFrame,
        top_k: int | None = None,
        output_path: Path | None = None,
    ) -> pd.DataFrame:
        """Rank candidates for a normalized job row or raw job description."""

        if is_official_candidates_frame(candidates):
            logger.info("Detected official challenge candidates; using challenge ranker.")
            return self.rank_challenge(candidates, top_n=top_k or 100, output_path=output_path)

        normalized_job = job_from_text(job) if isinstance(job, str) else job
        cleaned_candidates = clean_candidates(candidates)
        if cleaned_candidates.empty:
            raise ValueError("Candidate dataset is empty after preprocessing.")

        ids = cleaned_candidates["candidate_id"].astype(str).tolist()
        profiles = cleaned_candidates["searchable_profile"].astype(str).tolist()
        candidate_embeddings = self.embedding_service.get_candidate_embeddings(ids, profiles)
        query_embedding = self.embedding_service.encode([str(normalized_job.get("searchable_job", ""))])
        index = build_index(candidate_embeddings)
        requested_top_k = min(top_k or self.config.top_k, len(cleaned_candidates))
        similarities, indices = index.search(query_embedding, requested_top_k)

        results: list[dict[str, object]] = []
        for similarity, candidate_index in zip(similarities[0], indices[0], strict=False):
            candidate = cleaned_candidates.iloc[int(candidate_index)]
            semantic_score = float(np.clip(similarity, 0, 1) * 100)
            scores, matched, missing = calculate_scores(
                semantic_score=semantic_score,
                candidate=candidate,
                job=normalized_job,
                weights=self.config.weights,
            )
            results.append({
                "Candidate Name": candidate.get("name", ""),
                "Overall Score": scores.overall,
                "Semantic Score": scores.semantic,
                "Skill Score": scores.skills,
                "Experience Score": scores.experience,
                "Reason": generate_reason(candidate, normalized_job, scores, matched, missing),
            })

        ranked = pd.DataFrame(results).sort_values("Overall Score", ascending=False).reset_index(drop=True)
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        final_columns = [
            "Rank",
            "Candidate Name",
            "Overall Score",
            "Semantic Score",
            "Skill Score",
            "Experience Score",
            "Reason",
        ]
        ranked = ranked[final_columns]
        self.save_rankings(ranked, output_path or self.config.ranked_output_path)
        return ranked

    def rank_from_files(
        self,
        job_path: Path | str | None = None,
        candidate_path: Path | str | None = None,
        job_id: str | None = None,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Rank candidates from CSV, JSON, JSONL, or JSONL.GZ files."""

        candidates = self.load_candidates(candidate_path)
        if is_official_candidates_frame(candidates):
            return self.rank_challenge(candidates, top_n=top_k or 100)

        jobs = self.load_jobs(job_path)
        job = jobs[jobs["job_id"].astype(str) == job_id].iloc[0] if job_id else jobs.iloc[0]
        return self.rank(job=job, candidates=candidates, top_k=top_k)

    def save_rankings(self, rankings: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(path, index=False)
        logger.info("Saved ranked candidates to {}", path)

    def rank_challenge(
        self,
        candidates: pd.DataFrame,
        top_n: int = 100,
        output_path: Path | None = None,
        require_exact_submission: bool = False,
    ) -> pd.DataFrame:
        """Rank official challenge candidates and write review plus submission CSVs."""

        challenge_service = ChallengeRankingService()
        ranked = challenge_service.rank(candidates, top_n=top_n)
        review_path = output_path or self.config.output_dir / "ranked_candidates.csv"
        self.save_rankings(ranked, review_path)

        submission_path = self.config.output_dir / "submission.csv"
        challenge_service.write_submission(
            ranked,
            str(submission_path),
            top_n=min(100, top_n),
            require_exact=require_exact_submission,
        )
        logger.info("Saved official challenge submission to {}", submission_path)
        return ranked
