# HireSense AI

HireSense AI is a production-oriented candidate ranking system that uses semantic embeddings and structured scoring to recommend the best candidates for a job description. It combines sentence-transformer similarity, FAISS nearest-neighbor search, skill overlap, experience fit, project evidence, certifications, education, and optional behavioral/platform signals into explainable ranking results.

## Architecture

The project is split into a small service layer and two interfaces:

- `services/preprocessing.py` normalizes job and candidate data, removes duplicates, handles missing values, and builds searchable candidate profiles.
- `services/embedding_service.py` loads `sentence-transformers/all-MiniLM-L6-v2`, caches candidate embeddings, and builds a FAISS vector index with a scikit-learn fallback.
- `services/scoring.py` calculates weighted score components.
- `services/explainability.py` generates grounded explanations from matched and missing evidence.
- `services/ranking_service.py` orchestrates data loading, semantic search, scoring, output writing, and ranking.
- `api/routes.py` exposes FastAPI endpoints.
- `app.py` provides the Streamlit dashboard.

## Features

- Load candidate and job datasets from CSV using pandas.
- Normalize text, fill missing values, remove duplicate candidates, and merge resume fields into one searchable profile.
- Generate semantic embeddings with `all-MiniLM-L6-v2`.
- Cache candidate embeddings in `models/candidate_embeddings.npz`.
- Search candidates efficiently with FAISS.
- Rank candidates using configurable score weights.
- Explain every score using only available candidate and job data.
- Export `output/ranked_candidates.csv` with rank, candidate name, scores, and reason.
- Run through Streamlit or FastAPI.

## Installation

```bash
cd HireSenseAI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
cd HireSenseAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Locally

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The first ranking request downloads and loads the embedding model. Later requests reuse cached candidate embeddings unless the candidate data changes.

## API Documentation

FastAPI also serves interactive documentation at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

Endpoints:

- `GET /` returns the API welcome message.
- `GET /health` returns service health.
- `POST /rank` ranks the bundled candidate dataset against a supplied job description.
- `POST /upload-job` uploads a jobs CSV to `data/uploaded_jobs.csv`.
- `POST /upload-candidates` uploads a candidates CSV to `data/uploaded_candidates.csv`.

Example `/rank` body:

```json
{
  "job_description": "Build semantic search and ranking systems with embeddings, FAISS, FastAPI, pandas, and model evaluation.",
  "required_skills": "Python, sentence-transformers, FAISS, FastAPI, pandas",
  "min_experience_years": 3,
  "top_k": 5
}
```

## Folder Structure

```text
HireSenseAI/
  app.py
  main.py
  requirements.txt
  README.md
  .env.example
  data/
    candidates.csv
    jobs.csv
  models/
  services/
    embedding_service.py
    ranking_service.py
    preprocessing.py
    scoring.py
    explainability.py
  api/
    routes.py
  utils/
    config.py
  output/
  tests/
  assets/
```

## Configuration

Copy `.env.example` to `.env` and adjust values as needed:

```bash
copy .env.example .env
```

The scoring weights are normalized automatically, so they do not need to sum to exactly `1.0`.

## Testing

```bash
pytest
```

## Future Improvements

- Add resume PDF/DOCX parsing.
- Persist uploaded datasets with version metadata.
- Add role-specific scoring profiles.
- Add a feedback loop for recruiter decisions.
- Add authentication and audit logs for production deployments.
- Add model monitoring for embedding drift and score distribution shifts.

## Official Redrob Challenge Support

The official challenge bundle differs from the original HireSense demo in several important ways. The project now supports both modes.

### Mismatches Found Against the Official Files

- Input format: the original project expected flat CSV files, while the official bundle provides nested `candidates.jsonl`, `sample_candidates.json`, and `candidate_schema.json` records.
- Candidate identifiers: the original demo used arbitrary IDs/names; the challenge requires `CAND_XXXXXXX` identifiers from the official dataset.
- Job source: the challenge ranks against one fixed Senior AI Engineer JD, not arbitrary CSV job rows.
- Output format: the original output used `Rank`, `Candidate Name`, `Overall Score`, and component scores; the official validator requires exactly `candidate_id,rank,score,reasoning`.
- Row count: the challenge submission must contain exactly 100 data rows plus one header row.
- Score ordering: challenge scores must be monotonically non-increasing, with deterministic tie handling by candidate ID.
- Runtime constraints: the official ranking step must run on CPU, within 16 GB RAM, within 5 minutes, and with no network calls.
- Behavioral signals: the official data contains 23 `redrob_signals`; the demo treated behavior as optional text.
- Honeypots/traps: the official docs warn about keyword stuffing, impossible profiles, behavioral twins, and honeypot candidates.

### Official Reproduction Command

Use the challenge CLI for official submissions:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

For a quick sample run:

```bash
python rank.py --candidates ./sample_candidates.json --out ./sample_submission_generated.csv --top-n 50
```

The CLI writes the exact official columns:

```text
candidate_id,rank,score,reasoning
```

The Streamlit UI also supports official files. Upload `.json`, `.jsonl`, or `.jsonl.gz` and enable `Official challenge output` to download `submission.csv`.

### Challenge Ranking Approach

The official path uses deterministic local scoring rather than hosted APIs. It evaluates:

- Applied ML/search career evidence: embeddings, retrieval, ranking, vector search, LLM/NLP, evaluation, and production deployment signals.
- Skill strength: proficiency, endorsements, duration, and Redrob assessment scores.
- Experience fit: strongest fit around the JD's 5-9 year senior-engineer band.
- Product-company fit: downweights pure services/consulting-only profiles in line with the JD.
- Redrob availability: response rate, profile completeness, recency, open-to-work flag, notice period, interview completion, GitHub activity, and recruiter saves.
- Location/logistics: India/Pune/Noida signals and relocation flexibility.
- Risk penalties: keyword-stuffed non-technical titles, low availability, long notice period, and suspicious impossible-profile indicators.

The original sentence-transformer/FAISS semantic ranking workflow is preserved for CSV demos. The challenge CLI is the recommended path for the official dataset because it avoids model downloads or network access during the ranking step.
