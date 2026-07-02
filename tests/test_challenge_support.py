import pandas as pd

from services.challenge_adapter import flatten_candidate, is_official_candidates_frame
from services.challenge_ranker import ChallengeRankingService
from services.submission import SUBMISSION_COLUMNS, to_submission_frame


def official_candidate(candidate_id: str = "CAND_0000001") -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Senior Machine Learning Engineer",
            "summary": "Built production semantic search, embeddings, retrieval, and ranking systems.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7,
            "current_title": "Senior Machine Learning Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Software Product",
        },
        "career_history": [{
            "company": "ProductCo",
            "title": "Senior Machine Learning Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 72,
            "is_current": True,
            "industry": "Software Product",
            "company_size": "201-500",
            "description": "Deployed embeddings retrieval, vector search, ranking evaluation with NDCG and A/B tests.",
        }],
        "education": [{
            "institution": "IIT Test",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2010,
            "end_year": 2014,
            "tier": "tier_1",
        }],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 80},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 25, "duration_months": 42},
        ],
        "redrob_signals": {
            "profile_completeness_score": 95,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 8,
            "skill_assessment_scores": {"Python": 92, "FAISS": 88},
            "connection_count": 100,
            "endorsements_received": 40,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 35, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 80,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 4,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def test_flatten_candidate_detects_official_frame():
    row = flatten_candidate(official_candidate())
    frame = pd.DataFrame([row])

    assert row["candidate_id"] == "CAND_0000001"
    assert "semantic search" in row["searchable_profile"].lower()
    assert is_official_candidates_frame(frame)


def test_challenge_ranker_outputs_submission_schema():
    rows = [flatten_candidate(official_candidate(f"CAND_000000{i}")) for i in range(1, 4)]
    ranked = ChallengeRankingService().rank(pd.DataFrame(rows), top_n=3)
    submission = to_submission_frame(ranked, top_n=3, require_exact=False)

    assert submission.columns.tolist() == SUBMISSION_COLUMNS
    assert submission["rank"].tolist() == [1, 2, 3]
    assert submission["candidate_id"].tolist() == ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
