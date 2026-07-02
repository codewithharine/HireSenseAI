"""Adapters for the official Redrob challenge candidate format."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from services.preprocessing import normalize_text


CONSULTING_COMPANIES = {
    "accenture",
    "capgemini",
    "cognizant",
    "hcl",
    "infosys",
    "mindtree",
    "tcs",
    "tech mahindra",
    "wipro",
}


def iter_candidate_records(path: Path | str) -> Iterator[dict]:
    """Yield official candidate records from JSON, JSONL, or gzipped JSONL."""

    candidate_path = Path(path)
    suffixes = "".join(candidate_path.suffixes).lower()
    if suffixes.endswith(".jsonl.gz"):
        with gzip.open(candidate_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    if candidate_path.suffix.lower() == ".jsonl":
        with candidate_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    if candidate_path.suffix.lower() == ".json":
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return
    raise ValueError(f"Unsupported official candidate format: {candidate_path.suffix}")


def _join(values: list[str]) -> str:
    return " ".join(value for value in values if value)


def _skill_names(skills: list[dict]) -> list[str]:
    return [str(skill.get("name", "")).strip() for skill in skills if skill.get("name")]


def _skill_profile(skills: list[dict]) -> str:
    parts: list[str] = []
    for skill in skills:
        name = skill.get("name", "")
        if not name:
            continue
        parts.append(
            f"{name} {skill.get('proficiency', '')} "
            f"{skill.get('duration_months', 0)} months {skill.get('endorsements', 0)} endorsements"
        )
    return " ".join(parts)


def _career_text(career_history: list[dict]) -> str:
    parts: list[str] = []
    for role in career_history:
        parts.append(
            _join([
                str(role.get("title", "")),
                str(role.get("company", "")),
                str(role.get("industry", "")),
                str(role.get("description", "")),
            ])
        )
    return _join(parts)


def _education_text(education: list[dict]) -> str:
    return _join([
        _join([
            str(item.get("institution", "")),
            str(item.get("degree", "")),
            str(item.get("field_of_study", "")),
            str(item.get("tier", "")),
        ])
        for item in education
    ])


def _certification_text(certifications: list[dict]) -> str:
    return _join([
        _join([str(item.get("name", "")), str(item.get("issuer", "")), str(item.get("year", ""))])
        for item in certifications
    ])


def _career_months(career_history: list[dict]) -> int:
    return int(sum(int(role.get("duration_months", 0) or 0) for role in career_history))


def _product_months(career_history: list[dict]) -> int:
    total = 0
    for role in career_history:
        company = normalize_text(role.get("company", ""))
        industry = normalize_text(role.get("industry", ""))
        if "it services" not in industry and company not in CONSULTING_COMPANIES:
            total += int(role.get("duration_months", 0) or 0)
    return total


def _consulting_only(career_history: list[dict], profile: dict) -> bool:
    companies = {normalize_text(role.get("company", "")) for role in career_history}
    industries = {normalize_text(role.get("industry", "")) for role in career_history}
    current_company = normalize_text(profile.get("current_company", ""))
    if not companies and not current_company:
        return False
    company_set = companies | {current_company}
    return bool(company_set) and company_set <= CONSULTING_COMPANIES or (
        bool(industries) and all("it services" in industry for industry in industries)
    )


def _honeypot_flags(record: dict) -> list[str]:
    flags: list[str] = []
    profile = record.get("profile", {})
    skills = record.get("skills", [])
    career_history = record.get("career_history", [])
    years = float(profile.get("years_of_experience", 0) or 0)
    career_years = _career_months(career_history) / 12

    expert_zero_duration = [
        skill.get("name", "")
        for skill in skills
        if normalize_text(skill.get("proficiency", "")) == "expert"
        and int(skill.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero_duration) >= 2:
        flags.append("multiple expert skills with zero usage duration")
    if career_years > years + 3:
        flags.append("career duration materially exceeds stated experience")
    if years < 2 and sum(1 for skill in skills if normalize_text(skill.get("proficiency", "")) == "expert") >= 8:
        flags.append("many expert skills for very low experience")
    return flags


def flatten_candidate(record: dict) -> dict[str, object]:
    """Flatten one official candidate record into the internal tabular shape."""

    profile = record.get("profile", {})
    career_history = record.get("career_history", [])
    education = record.get("education", [])
    skills = record.get("skills", [])
    certifications = record.get("certifications", [])
    redrob = record.get("redrob_signals", {})

    skill_names = _skill_names(skills)
    career_text = _career_text(career_history)
    education_text = _education_text(education)
    certification_text = _certification_text(certifications)
    searchable_profile = _join([
        str(profile.get("headline", "")),
        str(profile.get("summary", "")),
        str(profile.get("current_title", "")),
        str(profile.get("current_industry", "")),
        career_text,
        _skill_profile(skills),
        education_text,
        certification_text,
        " ".join(str(key) for key in redrob.get("skill_assessment_scores", {}).keys()),
    ])

    return {
        "candidate_id": record.get("candidate_id", ""),
        "name": profile.get("anonymized_name", record.get("candidate_id", "")),
        "current_title": profile.get("current_title", ""),
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "experience_years": float(profile.get("years_of_experience", 0) or 0),
        "career_months": _career_months(career_history),
        "product_months": _product_months(career_history),
        "consulting_only": _consulting_only(career_history, profile),
        "skills": ", ".join(skill_names),
        "skill_list": [normalize_text(skill) for skill in skill_names],
        "skill_records": skills,
        "career_text": career_text,
        "education": education_text,
        "education_records": education,
        "certifications": certification_text,
        "certification_records": certifications,
        "redrob_signals": redrob,
        "profile_completeness_score": redrob.get("profile_completeness_score", 0),
        "open_to_work_flag": bool(redrob.get("open_to_work_flag", False)),
        "last_active_date": redrob.get("last_active_date", ""),
        "recruiter_response_rate": float(redrob.get("recruiter_response_rate", 0) or 0),
        "avg_response_time_hours": float(redrob.get("avg_response_time_hours", 0) or 0),
        "notice_period_days": int(redrob.get("notice_period_days", 0) or 0),
        "preferred_work_mode": redrob.get("preferred_work_mode", ""),
        "willing_to_relocate": bool(redrob.get("willing_to_relocate", False)),
        "github_activity_score": float(redrob.get("github_activity_score", -1) or -1),
        "interview_completion_rate": float(redrob.get("interview_completion_rate", 0) or 0),
        "offer_acceptance_rate": float(redrob.get("offer_acceptance_rate", -1) or -1),
        "saved_by_recruiters_30d": int(redrob.get("saved_by_recruiters_30d", 0) or 0),
        "search_appearance_30d": int(redrob.get("search_appearance_30d", 0) or 0),
        "skill_assessment_scores": redrob.get("skill_assessment_scores", {}),
        "honeypot_flags": _honeypot_flags(record),
        "searchable_profile": searchable_profile,
        "source_format": "official_json",
    }


def load_official_candidates(path: Path | str, limit: int | None = None) -> pd.DataFrame:
    """Load and flatten official challenge candidates."""

    rows: list[dict[str, object]] = []
    for index, record in enumerate(iter_candidate_records(path)):
        if limit is not None and index >= limit:
            break
        rows.append(flatten_candidate(record))
    return pd.DataFrame(rows)


def is_official_candidates_frame(df: pd.DataFrame) -> bool:
    return "source_format" in df.columns and set(df["source_format"].dropna().unique()) == {"official_json"}
