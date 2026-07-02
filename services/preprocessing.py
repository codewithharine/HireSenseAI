"""Text normalization and feature extraction for ranking."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


CANDIDATE_TEXT_COLUMNS = [
    "skills",
    "experience",
    "projects",
    "education",
    "certifications",
    "behavior",
    "platform_activity",
]
JOB_TEXT_COLUMNS = ["description", "required_skills", "education", "certifications"]


def normalize_text(value: object) -> str:
    """Return lowercased, whitespace-normalized text for nullable input."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9+#.\s,;|/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_list(value: object) -> list[str]:
    """Split comma, pipe, slash, or semicolon separated values into clean tokens."""

    text = normalize_text(value)
    if not text:
        return []
    parts = re.split(r"[,;|/]", text)
    return sorted({part.strip() for part in parts if part.strip()})


def extract_years(value: object) -> float:
    """Extract years of experience from a number or text field."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", normalize_text(value))]
    return max(numbers) if numbers else 0.0


def merge_profile(row: pd.Series, columns: Iterable[str] = CANDIDATE_TEXT_COLUMNS) -> str:
    """Merge searchable candidate attributes into one semantic profile."""

    values = [normalize_text(row.get(column, "")) for column in columns]
    return " ".join(value for value in values if value)


def clean_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidates, remove duplicates, and create profile fields."""

    cleaned = df.copy()
    cleaned.columns = [normalize_text(column).replace(" ", "_") for column in cleaned.columns]
    if "candidate_id" not in cleaned.columns:
        cleaned["candidate_id"] = [f"CAND-{index + 1:04d}" for index in range(len(cleaned))]
    if "name" not in cleaned.columns:
        cleaned["name"] = cleaned["candidate_id"]

    for column in set(CANDIDATE_TEXT_COLUMNS + ["name", "candidate_id"]):
        if column not in cleaned.columns:
            cleaned[column] = ""
        cleaned[column] = cleaned[column].fillna("")

    if "experience_years" not in cleaned.columns:
        cleaned["experience_years"] = cleaned.get("experience", "").apply(extract_years)
    else:
        cleaned["experience_years"] = cleaned["experience_years"].apply(extract_years)

    cleaned["skill_list"] = cleaned["skills"].apply(split_list)
    cleaned["certification_list"] = cleaned["certifications"].apply(split_list)
    cleaned["searchable_profile"] = cleaned.apply(merge_profile, axis=1)
    cleaned = cleaned.drop_duplicates(subset=["candidate_id"], keep="first")
    return cleaned[cleaned["searchable_profile"].str.len() > 0].reset_index(drop=True)


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize job descriptions and create semantic query text."""

    cleaned = df.copy()
    cleaned.columns = [normalize_text(column).replace(" ", "_") for column in cleaned.columns]
    if "job_id" not in cleaned.columns:
        cleaned["job_id"] = [f"JOB-{index + 1:04d}" for index in range(len(cleaned))]
    if "title" not in cleaned.columns:
        cleaned["title"] = cleaned["job_id"]
    for column in set(JOB_TEXT_COLUMNS + ["job_id", "title"]):
        if column not in cleaned.columns:
            cleaned[column] = ""
        cleaned[column] = cleaned[column].fillna("")
    if "min_experience_years" not in cleaned.columns:
        cleaned["min_experience_years"] = 0.0
    cleaned["min_experience_years"] = cleaned["min_experience_years"].apply(extract_years)
    cleaned["required_skill_list"] = cleaned["required_skills"].apply(split_list)
    cleaned["certification_list"] = cleaned["certifications"].apply(split_list)
    cleaned["searchable_job"] = cleaned.apply(
        lambda row: " ".join(normalize_text(row.get(column, "")) for column in JOB_TEXT_COLUMNS).strip(),
        axis=1,
    )
    return cleaned.drop_duplicates(subset=["job_id"], keep="first").reset_index(drop=True)


def job_from_text(description: str, required_skills: str = "", min_experience_years: float = 0) -> pd.Series:
    """Build a normalized job row from free-form input."""

    df = pd.DataFrame([{
        "job_id": "uploaded-job",
        "title": "Uploaded Job",
        "description": description,
        "required_skills": required_skills,
        "min_experience_years": min_experience_years,
        "education": "",
        "certifications": "",
    }])
    return clean_jobs(df).iloc[0]

