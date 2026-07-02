import pandas as pd

from services.preprocessing import clean_candidates, job_from_text, split_list


def test_split_list_normalizes_unique_values():
    assert split_list("Python, python; FastAPI / Docker") == ["docker", "fastapi", "python"]


def test_clean_candidates_builds_searchable_profile():
    df = pd.DataFrame([{
        "candidate_id": "1",
        "name": "Asha",
        "skills": "Python, FastAPI",
        "experience_years": "5 years",
        "projects": "Built APIs",
    }])

    cleaned = clean_candidates(df)

    assert cleaned.loc[0, "experience_years"] == 5
    assert "python" in cleaned.loc[0, "searchable_profile"]


def test_job_from_text_extracts_required_skills():
    job = job_from_text("Build Python APIs", "Python, FastAPI", 3)

    assert job["required_skill_list"] == ["fastapi", "python"]
    assert job["min_experience_years"] == 3
