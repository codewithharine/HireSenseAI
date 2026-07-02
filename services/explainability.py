"""Evidence-grounded candidate explanations."""

from __future__ import annotations

import pandas as pd

from services.preprocessing import normalize_text
from services.scoring import ScoreBreakdown


def generate_reason(
    candidate: pd.Series,
    job: pd.Series,
    scores: ScoreBreakdown,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """Create a concise explanation using only available candidate/job data."""

    reasons: list[str] = []
    if scores.semantic >= 75:
        reasons.append("Strong semantic alignment with the job description")
    elif scores.semantic >= 50:
        reasons.append("Moderate semantic alignment with the job description")
    else:
        reasons.append("Limited semantic alignment with the job description")

    required_skills = list(job.get("required_skill_list", []))
    if required_skills:
        reasons.append(f"matches {len(matched_skills)} of {len(required_skills)} required skills")
    elif matched_skills:
        reasons.append(f"shows relevant skills including {', '.join(matched_skills[:5])}")

    years = float(candidate.get("experience_years", 0) or 0)
    if years:
        reasons.append(f"{years:g} years experience")

    if normalize_text(candidate.get("projects", "")):
        reasons.append("has project evidence in the profile")

    if normalize_text(candidate.get("certifications", "")):
        reasons.append("includes listed certifications")

    if normalize_text(candidate.get("behavior", "")) or normalize_text(candidate.get("platform_activity", "")):
        reasons.append("has behavioral or platform activity signals")

    if missing_skills:
        reasons.append(f"missing {', '.join(missing_skills[:5])}")

    return ". ".join(reasons) + "."
