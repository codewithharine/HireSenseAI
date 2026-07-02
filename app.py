"""Streamlit dashboard for HireSense AI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from services.challenge_adapter import is_official_candidates_frame
from services.preprocessing import job_from_text
from services.ranking_service import RankingService
from services.submission import to_submission_frame
from utils.config import settings


st.set_page_config(page_title="HireSense AI", layout="wide")


@st.cache_resource
def get_ranking_service() -> RankingService:
    """Cache the model-backed ranking service for the Streamlit session."""

    return RankingService()


def _load_uploaded_candidates(uploaded_file) -> pd.DataFrame:
    service = get_ranking_service()
    if uploaded_file is None:
        return pd.read_csv(settings.data_dir / "candidates.csv")

    filename = uploaded_file.name.lower()
    if filename.endswith((".json", ".jsonl", ".jsonl.gz")):
        suffix = "".join(Path(uploaded_file.name).suffixes) or ".jsonl"
        target = settings.output_dir / f"uploaded_candidates{suffix}"
        target.write_bytes(uploaded_file.getvalue())
        return service.load_candidates(target)
    return pd.read_csv(uploaded_file)


def main() -> None:
    st.title("HireSense AI")
    st.caption("Semantic candidate ranking with official Redrob challenge support")

    with st.sidebar:
        st.header("Ranking Settings")
        top_k = st.number_input("Top candidates", min_value=1, max_value=100, value=settings.top_k)
        challenge_mode = st.checkbox("Official challenge output", value=False)
        min_experience = st.number_input(
            "Minimum experience years",
            min_value=0.0,
            value=settings.min_experience_years,
            disabled=challenge_mode,
        )
        required_skills = st.text_area(
            "Required skills",
            placeholder="Python, FastAPI, FAISS, Docker",
            disabled=challenge_mode,
        )

    default_job_text = (
        "Senior AI Engineer founding-team role requiring production embeddings-based retrieval, "
        "vector search or hybrid search, ranking evaluation, strong Python, product engineering judgment, "
        "and recent Redrob platform availability signals."
    )
    job_description = st.text_area(
        "Job description",
        value=default_job_text if challenge_mode else "",
        height=180,
        disabled=challenge_mode,
        placeholder="Paste the role description, responsibilities, and requirements here.",
    )
    candidate_file = st.file_uploader("Upload candidate file", type=["csv", "json", "jsonl", "gz"])

    if st.button("Rank Candidates", type="primary", use_container_width=True):
        if not challenge_mode and not job_description.strip():
            st.error("Please provide a job description.")
            return

        try:
            service = get_ranking_service()
            candidates = _load_uploaded_candidates(candidate_file)

            if challenge_mode or is_official_candidates_frame(candidates):
                ranked = service.rank_challenge(candidates, top_n=int(top_k), require_exact_submission=False)
                st.success(f"Ranked {len(ranked)} official challenge candidates.")
                visible_columns = [
                    "Rank",
                    "Candidate ID",
                    "Candidate Name",
                    "Challenge Score",
                    "Career Score",
                    "Skill Score",
                    "Behavior Score",
                    "Risk Penalty",
                    "Reason",
                ]
                st.dataframe(ranked[visible_columns], use_container_width=True, hide_index=True)
                submission = to_submission_frame(ranked, top_n=len(ranked), require_exact=False)
                csv_buffer = StringIO()
                submission.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download official submission.csv",
                    data=csv_buffer.getvalue(),
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                return

            job = job_from_text(
                description=job_description,
                required_skills=required_skills,
                min_experience_years=min_experience,
            )
            ranked = service.rank(job=job, candidates=candidates, top_k=int(top_k))
            st.success(f"Ranked {len(ranked)} candidates.")

            visible_columns = [
                "Rank",
                "Candidate Name",
                "Overall Score",
                "Semantic Score",
                "Skill Score",
                "Experience Score",
                "Reason",
            ]
            st.dataframe(ranked[visible_columns], use_container_width=True, hide_index=True)

            csv_buffer = StringIO()
            ranked.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download ranked_candidates.csv",
                data=csv_buffer.getvalue(),
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
