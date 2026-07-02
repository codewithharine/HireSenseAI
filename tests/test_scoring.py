import pandas as pd

from services.preprocessing import job_from_text
from services.scoring import calculate_scores, skill_match_score
from utils.config import ScoreWeights


def test_skill_match_score_reports_missing_skills():
    score, matched, missing = skill_match_score(["python", "docker"], ["python"])

    assert score == 50
    assert matched == ["python"]
    assert missing == ["docker"]


def test_calculate_scores_combines_components():
    candidate = pd.Series({
        "skills": "Python, FastAPI",
        "skill_list": ["python", "fastapi"],
        "experience_years": 4,
        "projects": "Built APIs",
        "education": "Computer Science",
        "certifications": "",
        "certification_list": [],
        "behavior": "Mentored peers",
        "platform_activity": "",
    })
    job = job_from_text("Build Python APIs", "Python, FastAPI", 3)

    scores, matched, missing = calculate_scores(80, candidate, job, ScoreWeights())

    assert scores.overall > 70
    assert matched == ["fastapi", "python"]
    assert missing == []
