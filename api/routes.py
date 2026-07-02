"""FastAPI routes for HireSense AI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from services.preprocessing import job_from_text
from services.ranking_service import RankingService
from services.submission import to_submission_frame
from utils.config import settings


router = APIRouter()
ranking_service = RankingService()


class RankRequest(BaseModel):
    """Request body for ranking bundled candidate data."""

    job_description: str = Field(..., min_length=10)
    required_skills: str = ""
    min_experience_years: float = Field(default=0, ge=0)
    top_k: int = Field(default=10, ge=1, le=100)


class ChallengeRankRequest(BaseModel):
    """Request body for ranking an uploaded official candidate file."""

    candidates_path: str = Field(..., description="Path to .json, .jsonl, or .jsonl.gz official candidates file")
    top_k: int = Field(default=100, ge=1, le=100)


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "HireSense AI Candidate Ranking API"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.post("/rank")
def rank_candidates(payload: RankRequest) -> dict[str, object]:
    """Rank the default candidate dataset against a supplied job description."""

    try:
        candidates = ranking_service.load_candidates()
        job = job_from_text(
            description=payload.job_description,
            required_skills=payload.required_skills,
            min_experience_years=payload.min_experience_years,
        )
        ranked = ranking_service.rank(job=job, candidates=candidates, top_k=payload.top_k)
        return {"results": ranked.to_dict(orient="records")}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/rank-challenge")
def rank_challenge(payload: ChallengeRankRequest) -> dict[str, object]:
    """Rank official Redrob challenge candidates and return both review/submission rows."""

    try:
        candidates = ranking_service.load_candidates(payload.candidates_path)
        ranked = ranking_service.rank_challenge(candidates, top_n=payload.top_k, require_exact_submission=False)
        submission = to_submission_frame(ranked, top_n=len(ranked), require_exact=False)
        return {
            "results": ranked.to_dict(orient="records"),
            "submission": submission.to_dict(orient="records"),
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload-job")
async def upload_job(file: UploadFile = File(...)) -> dict[str, object]:
    """Upload a jobs CSV file into the data directory."""

    contents = await file.read()
    target = settings.data_dir / "uploaded_jobs.csv"
    target.write_bytes(contents)
    jobs = ranking_service.load_jobs(target)
    return {"message": "Job file uploaded", "path": str(target), "rows": len(jobs)}


@router.post("/upload-candidates")
async def upload_candidates(file: UploadFile = File(...)) -> dict[str, object]:
    """Upload candidate CSV or official JSON/JSONL data into the data directory."""

    contents = await file.read()
    suffixes = "".join(Path(file.filename or "").suffixes).lower()
    if suffixes.endswith((".json", ".jsonl", ".jsonl.gz")):
        target = settings.data_dir / f"uploaded_candidates{suffixes}"
        target.write_bytes(contents)
        candidates = ranking_service.load_candidates(target)
        return {"message": "Official candidate file uploaded", "path": str(target), "rows": len(candidates)}

    target = settings.data_dir / "uploaded_candidates.csv"
    target.write_bytes(contents)
    candidates = pd.read_csv(StringIO(contents.decode("utf-8")))
    return {"message": "Candidate file uploaded", "path": str(target), "rows": len(candidates)}
