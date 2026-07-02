"""Deterministic score components for candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from services.preprocessing import normalize_text, split_list
from utils.config import ScoreWeights


@dataclass(frozen=True)
class ScoreBreakdown:
    semantic: float
    skills: float
    experience: float
    projects: float
    education: float
    certifications: float
    behavior: float
    overall: float


def clamp_score(value: float) -> float:
    """Clamp a score to the 0-100 range."""

    return max(0.0, min(100.0, round(value, 2)))


def skill_match_score(required: list[str], candidate: list[str]) -> tuple[float, list[str], list[str]]:
    """Score required skill overlap and return matched/missing evidence."""

    required_set = {normalize_text(skill) for skill in required if normalize_text(skill)}
    candidate_set = {normalize_text(skill) for skill in candidate if normalize_text(skill)}
    if not required_set:
        return 100.0, [], []
    matched = sorted(required_set & candidate_set)
    missing = sorted(required_set - candidate_set)
    return clamp_score((len(matched) / len(required_set)) * 100), matched, missing


def experience_score(candidate_years: float, required_years: float) -> float:
    """Score experience against a required minimum."""

    if required_years <= 0:
        return 100.0 if candidate_years > 0 else 0.0
    return clamp_score((candidate_years / required_years) * 100)


def evidence_score(value: object) -> float:
    """Return full credit when a field has usable evidence."""

    return 100.0 if normalize_text(value) else 0.0


def certification_score(required: list[str], candidate: list[str]) -> float:
    """Score required certification overlap."""

    required_set = {normalize_text(item) for item in required if normalize_text(item)}
    candidate_set = {normalize_text(item) for item in candidate if normalize_text(item)}
    if not required_set:
        return 100.0 if candidate_set else 0.0
    return clamp_score((len(required_set & candidate_set) / len(required_set)) * 100)


def behavior_score(candidate: pd.Series) -> float:
    """Score behavioral and platform activity evidence when available."""

    evidence = " ".join([
        normalize_text(candidate.get("behavior", "")),
        normalize_text(candidate.get("platform_activity", "")),
    ]).strip()
    return 100.0 if evidence else 0.0


def calculate_scores(
    semantic_score: float,
    candidate: pd.Series,
    job: pd.Series,
    weights: ScoreWeights,
) -> tuple[ScoreBreakdown, list[str], list[str]]:
    """Calculate weighted overall score and return skill evidence."""

    skill_score, matched, missing = skill_match_score(
        list(job.get("required_skill_list", [])),
        list(candidate.get("skill_list", split_list(candidate.get("skills", "")))),
    )
    exp_score = experience_score(
        float(candidate.get("experience_years", 0) or 0),
        float(job.get("min_experience_years", 0) or 0),
    )
    project_score = evidence_score(candidate.get("projects", ""))
    edu_score = evidence_score(candidate.get("education", ""))
    cert_score = certification_score(
        list(job.get("certification_list", [])),
        list(candidate.get("certification_list", split_list(candidate.get("certifications", "")))),
    )
    beh_score = behavior_score(candidate)
    normalized = weights.normalized
    overall = (
        semantic_score * normalized["semantic"]
        + skill_score * normalized["skills"]
        + exp_score * normalized["experience"]
        + project_score * normalized["projects"]
        + edu_score * normalized["education"]
        + cert_score * normalized["certifications"]
        + beh_score * normalized["behavior"]
    )
    return ScoreBreakdown(
        semantic=clamp_score(semantic_score),
        skills=clamp_score(skill_score),
        experience=clamp_score(exp_score),
        projects=clamp_score(project_score),
        education=clamp_score(edu_score),
        certifications=clamp_score(cert_score),
        behavior=clamp_score(beh_score),
        overall=clamp_score(overall),
    ), matched, missing
