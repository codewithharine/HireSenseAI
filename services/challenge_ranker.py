"""Official Redrob challenge ranker.

This module is intentionally deterministic and local-only. It does not call
hosted APIs or require downloading embedding models during the ranking step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from services.challenge_adapter import is_official_candidates_frame, load_official_candidates
from services.preprocessing import normalize_text
from services.submission import write_submission


AI_RETRIEVAL_ALIASES: dict[str, tuple[str, ...]] = {
    "embeddings": ("embedding", "embeddings", "sentence-transformers", "bge", "e5"),
    "retrieval": ("retrieval", "information retrieval", "semantic search", "rag"),
    "ranking": ("ranking", "ranker", "recommendation", "recommender", "learning-to-rank", "ltr"),
    "vector_search": ("vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch"),
    "evaluation": ("ndcg", "mrr", "map", "a/b", "ab test", "offline benchmark", "evaluation"),
    "python_ml": ("python", "scikit-learn", "pytorch", "tensorflow", "numpy", "pandas"),
    "llm": ("llm", "fine-tuning", "lora", "qlora", "peft", "transformer", "nlp"),
    "production": ("production", "deployed", "scale", "monitoring", "regression", "latency", "index refresh"),
}

POSITIVE_TITLE_TERMS = (
    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "senior machine learning engineer",
    "data scientist",
    "search engineer",
    "recommendation engineer",
    "backend engineer",
    "data engineer",
)

NEGATIVE_TITLE_TERMS = (
    "marketing manager",
    "hr manager",
    "accountant",
    "sales executive",
    "graphic designer",
    "civil engineer",
    "mechanical engineer",
    "customer support",
    "content writer",
    "operations manager",
)

INDIA_LOCATION_TERMS = ("india", "pune", "noida", "delhi", "gurugram", "hyderabad", "mumbai", "bengaluru", "bangalore")
PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.8, "expert": 1.0}


@dataclass(frozen=True)
class ChallengeScore:
    score: float
    career: float
    skills: float
    experience: float
    product: float
    behavior: float
    location: float
    risk_penalty: float
    reasoning: str


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _term_coverage(text: str, aliases: dict[str, tuple[str, ...]]) -> tuple[float, list[str]]:
    matched = [label for label, terms in aliases.items() if _contains_any(text, terms)]
    return len(matched) / len(aliases), matched


def _skill_strength(candidate: pd.Series) -> tuple[float, list[str]]:
    matched_scores: list[float] = []
    matched_labels: list[str] = []
    skill_records = candidate.get("skill_records", []) or []
    assessment_scores = candidate.get("skill_assessment_scores", {}) or {}

    for label, aliases in AI_RETRIEVAL_ALIASES.items():
        best = 0.0
        for skill in skill_records:
            skill_name = normalize_text(skill.get("name", ""))
            if not _contains_any(skill_name, aliases):
                continue
            proficiency = PROFICIENCY_WEIGHT.get(normalize_text(skill.get("proficiency", "")), 0.4)
            duration = _clamp(float(skill.get("duration_months", 0) or 0) / 48)
            endorsements = _clamp(float(skill.get("endorsements", 0) or 0) / 60)
            best = max(best, 0.50 * proficiency + 0.30 * duration + 0.20 * endorsements)

        for assessment_name, assessment_score in assessment_scores.items():
            if _contains_any(normalize_text(assessment_name), aliases):
                best = max(best, _clamp(float(assessment_score or 0) / 100))

        if best > 0:
            matched_scores.append(best)
            matched_labels.append(label)

    if not AI_RETRIEVAL_ALIASES:
        return 0.0, []
    return sum(matched_scores) / len(AI_RETRIEVAL_ALIASES), matched_labels


def _experience_fit(years: float) -> float:
    if 5 <= years <= 9:
        return 1.0
    if 4 <= years < 5:
        return 0.78
    if 9 < years <= 12:
        return 0.72
    if 3 <= years < 4:
        return 0.48
    if 12 < years <= 15:
        return 0.42
    return 0.18


def _product_fit(candidate: pd.Series, text: str) -> float:
    product_months = float(candidate.get("product_months", 0) or 0)
    career_months = max(float(candidate.get("career_months", 0) or 0), 1.0)
    product_ratio = product_months / career_months
    production_mentions = 1.0 if _contains_any(text, AI_RETRIEVAL_ALIASES["production"]) else 0.0
    consulting_penalty = 0.35 if bool(candidate.get("consulting_only", False)) else 0.0
    return _clamp(0.65 * product_ratio + 0.35 * production_mentions - consulting_penalty)


def _days_since(value: object, today: date | None = None) -> int | None:
    if not value:
        return None
    today = today or date.today()
    try:
        return (today - datetime.strptime(str(value), "%Y-%m-%d").date()).days
    except ValueError:
        return None


def _behavior_fit(candidate: pd.Series) -> float:
    response_rate = _clamp(float(candidate.get("recruiter_response_rate", 0) or 0))
    profile_complete = _clamp(float(candidate.get("profile_completeness_score", 0) or 0) / 100)
    interview_rate = _clamp(float(candidate.get("interview_completion_rate", 0) or 0))
    saved = _clamp(float(candidate.get("saved_by_recruiters_30d", 0) or 0) / 12)
    github_raw = float(candidate.get("github_activity_score", -1) or -1)
    github = 0.0 if github_raw < 0 else _clamp(github_raw / 100)
    notice = int(candidate.get("notice_period_days", 180) or 180)
    notice_fit = 1.0 if notice <= 30 else 0.65 if notice <= 60 else 0.30 if notice <= 90 else 0.10
    active_days = _days_since(candidate.get("last_active_date", ""))
    recency = 0.5 if active_days is None else 1.0 if active_days <= 30 else 0.55 if active_days <= 90 else 0.10
    open_to_work = 1.0 if bool(candidate.get("open_to_work_flag", False)) else 0.35

    return _clamp(
        0.22 * response_rate
        + 0.13 * profile_complete
        + 0.12 * interview_rate
        + 0.10 * saved
        + 0.12 * github
        + 0.12 * notice_fit
        + 0.12 * recency
        + 0.07 * open_to_work
    )


def _location_fit(candidate: pd.Series) -> float:
    text = normalize_text(
        " ".join([
            str(candidate.get("location", "")),
            str(candidate.get("country", "")),
            str(candidate.get("preferred_work_mode", "")),
        ])
    )
    if _contains_any(text, INDIA_LOCATION_TERMS):
        return 1.0
    if bool(candidate.get("willing_to_relocate", False)):
        return 0.72
    return 0.25


def _risk_penalty(candidate: pd.Series, text: str, title_text: str, matched_skill_labels: list[str]) -> float:
    penalty = 0.0
    if _contains_any(title_text, NEGATIVE_TITLE_TERMS) and len(matched_skill_labels) >= 5:
        penalty += 0.18
    if _contains_any(title_text, NEGATIVE_TITLE_TERMS) and not _contains_any(text, ("retrieval", "ranking", "ml", "machine learning")):
        penalty += 0.22
    if bool(candidate.get("consulting_only", False)):
        penalty += 0.08
    if candidate.get("honeypot_flags"):
        penalty += min(0.30, 0.12 * len(candidate.get("honeypot_flags", [])))
    if int(candidate.get("notice_period_days", 0) or 0) > 90:
        penalty += 0.05
    if float(candidate.get("recruiter_response_rate", 0) or 0) < 0.10:
        penalty += 0.08
    return _clamp(penalty, 0.0, 0.55)


def _make_reason(candidate: pd.Series, score: ChallengeScore | None, matched_terms: list[str], matched_skills: list[str]) -> str:
    title = str(candidate.get("current_title", "") or "Candidate")
    years = float(candidate.get("experience_years", 0) or 0)
    response = float(candidate.get("recruiter_response_rate", 0) or 0)
    product_years = float(candidate.get("product_months", 0) or 0) / 12
    positives: list[str] = [f"{title} with {years:.1f} yrs"]
    if matched_terms:
        positives.append(f"career evidence for {', '.join(matched_terms[:3]).replace('_', ' ')}")
    if matched_skills:
        positives.append(f"{len(matched_skills)} relevant AI/search skill groups")
    if product_years:
        positives.append(f"{product_years:.1f} yrs outside pure services")
    positives.append(f"response rate {response:.2f}")

    concerns: list[str] = []
    if bool(candidate.get("consulting_only", False)):
        concerns.append("mostly services-company background")
    if int(candidate.get("notice_period_days", 0) or 0) > 60:
        concerns.append(f"{int(candidate.get('notice_period_days'))}-day notice")
    if candidate.get("honeypot_flags"):
        concerns.append(str(candidate.get("honeypot_flags")[0]))
    if concerns:
        return f"{'; '.join(positives)}. Concern: {', '.join(concerns[:2])}."
    return f"{'; '.join(positives)}."


def score_candidate(candidate: pd.Series) -> ChallengeScore:
    """Score one official candidate for the released Senior AI Engineer JD."""

    text = normalize_text(
        " ".join([
            str(candidate.get("headline", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("career_text", "")),
            str(candidate.get("skills", "")),
            str(candidate.get("certifications", "")),
        ])
    )
    title_text = normalize_text(" ".join([str(candidate.get("current_title", "")), str(candidate.get("headline", ""))]))
    career_score, matched_terms = _term_coverage(text, AI_RETRIEVAL_ALIASES)
    title_score = 1.0 if _contains_any(title_text, POSITIVE_TITLE_TERMS) else 0.15
    negative_title = 0.0 if _contains_any(title_text, NEGATIVE_TITLE_TERMS) else 1.0
    career = _clamp(0.62 * career_score + 0.28 * title_score + 0.10 * negative_title)
    skills, matched_skills = _skill_strength(candidate)
    experience = _experience_fit(float(candidate.get("experience_years", 0) or 0))
    product = _product_fit(candidate, text)
    behavior = _behavior_fit(candidate)
    location = _location_fit(candidate)
    risk_penalty = _risk_penalty(candidate, text, title_text, matched_skills)

    raw_score = (
        0.30 * career
        + 0.23 * skills
        + 0.15 * product
        + 0.12 * experience
        + 0.13 * behavior
        + 0.07 * location
    )
    score = _clamp(raw_score - risk_penalty)
    preliminary = ChallengeScore(
        score=score,
        career=career,
        skills=skills,
        experience=experience,
        product=product,
        behavior=behavior,
        location=location,
        risk_penalty=risk_penalty,
        reasoning="",
    )
    return ChallengeScore(
        score=preliminary.score,
        career=career,
        skills=skills,
        experience=experience,
        product=product,
        behavior=behavior,
        location=location,
        risk_penalty=risk_penalty,
        reasoning=_make_reason(candidate, preliminary, matched_terms, matched_skills),
    )


class ChallengeRankingService:
    """Rank official challenge candidates and write compliant submissions."""

    def rank(self, candidates: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
        if not is_official_candidates_frame(candidates):
            raise ValueError("ChallengeRankingService expects official JSON/JSONL candidates.")

        rows: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            score = score_candidate(candidate)
            rows.append({
                "Candidate ID": candidate.get("candidate_id", ""),
                "Candidate Name": candidate.get("name", ""),
                "Challenge Score": round(score.score, 4),
                "Career Score": round(score.career, 4),
                "Skill Score": round(score.skills, 4),
                "Experience Score": round(score.experience, 4),
                "Product Score": round(score.product, 4),
                "Behavior Score": round(score.behavior, 4),
                "Location Score": round(score.location, 4),
                "Risk Penalty": round(score.risk_penalty, 4),
                "Reason": score.reasoning,
            })

        ranked = pd.DataFrame(rows)
        ranked = ranked.sort_values(["Challenge Score", "Candidate ID"], ascending=[False, True]).reset_index(drop=True)
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        return ranked.head(top_n)

    def rank_file(self, candidates_path: str, top_n: int = 100, limit: int | None = None) -> pd.DataFrame:
        candidates = load_official_candidates(candidates_path, limit=limit)
        return self.rank(candidates, top_n=top_n)

    def write_submission(self, ranked: pd.DataFrame, output_path: str, top_n: int = 100, require_exact: bool = True) -> pd.DataFrame:
        return write_submission(ranked, output_path, top_n=top_n, require_exact=require_exact)
