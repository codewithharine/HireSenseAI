"""Application configuration and path management."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class ScoreWeights(BaseModel):
    """Configurable score weights."""

    semantic: float = Field(default=0.50, ge=0)
    skills: float = Field(default=0.20, ge=0)
    experience: float = Field(default=0.15, ge=0)
    projects: float = Field(default=0.10, ge=0)
    education: float = Field(default=0.00, ge=0)
    certifications: float = Field(default=0.00, ge=0)
    behavior: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def validate_total_weight(self) -> "ScoreWeights":
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("At least one score weight must be positive.")
        return self

    @property
    def normalized(self) -> dict[str, float]:
        weights = self.model_dump()
        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    output_dir: Path = ROOT_DIR / "output"
    model_name: str = os.getenv("HIRESENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("HIRESENSE_TOP_K", "10"))
    min_experience_years: float = float(os.getenv("HIRESENSE_MIN_EXPERIENCE_YEARS", "0"))
    weights: ScoreWeights = Field(default_factory=lambda: ScoreWeights(
        semantic=float(os.getenv("HIRESENSE_SEMANTIC_WEIGHT", "0.50")),
        skills=float(os.getenv("HIRESENSE_SKILL_WEIGHT", "0.20")),
        experience=float(os.getenv("HIRESENSE_EXPERIENCE_WEIGHT", "0.15")),
        projects=float(os.getenv("HIRESENSE_PROJECT_WEIGHT", "0.10")),
        education=float(os.getenv("HIRESENSE_EDUCATION_WEIGHT", "0.00")),
        certifications=float(os.getenv("HIRESENSE_CERTIFICATION_WEIGHT", "0.00")),
        behavior=float(os.getenv("HIRESENSE_BEHAVIOR_WEIGHT", "0.05")),
    ))

    @property
    def candidate_cache_path(self) -> Path:
        return self.models_dir / "candidate_embeddings.npz"

    @property
    def ranked_output_path(self) -> Path:
        return self.output_dir / "ranked_candidates.csv"

    def ensure_directories(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
