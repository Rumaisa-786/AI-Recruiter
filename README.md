# AI Candidate Ranking System

A rule-based candidate ranking system built for the Redrob Intelligent Candidate Discovery & Ranking Challenge. Ranks a 100,000-candidate pool against a single job description and outputs the top 100 fits with explainable reasoning, while explicitly filtering out unavailable candidates and honeypot profiles.

**The code that produces the submission CSV is `src/data_loader.py` and `src/runner.py` only.** The repo also contains an earlier 5-stage hybrid pipeline (`src/scoring/`, `src/ranking/`, `src/matching/`) kept for transparency about our design process; it is not called by `runner.py` and does not run at submission time. See "Two design iterations" below for why.

## Quick start

```bash
pip install -r requirements.txt
python -m src.runner
python validate_submission.py outputs/submission.csv
```

`python -m src.runner` reads `data/raw/candidates.jsonl`, scores all 100,000 candidates, and writes the ranked top 100 to `outputs/submission.csv`. The full run completes in under a minute on a standard CPU, well inside the 5-minute / 16GB / CPU-only constraint. No GPU and no hosted LLM calls are used at ranking time.

## Project structure

```
ai-recruiter/
├── data/
│   ├── raw/
│   │   └── candidates.jsonl       100k candidate pool (gitignored, not in repo)
│   ├── jobs/
│   │   └── job_description.docx   target role
│   └── candidate_schema.json
├── src/
│   ├── data_loader.py             scoring logic and honeypot detection
│   ├── runner.py                  orchestrates load -> score -> rank -> export
│   ├── understanding/
│   │   └── llm_extractor.py       Groq-based JD parsing (development only)
│   ├── matching/
│   │   ├── embedder.py            dense embeddings (exploratory, not used at ranking time)
│   │   └── sparse_retriever.py    BM25 (exploratory, not used at ranking time)
│   ├── scoring/
│   │   └── scorer.py              generic multi-signal scorer (early prototype)
│   ├── ranking/
│   │   └── pipeline.py            5-stage hybrid pipeline (early prototype)
│   └── explainability/
│       └── explainer.py           recruiter-facing report generation
├── outputs/
│   └── submission.csv             final ranked output
├── validate_submission.py         official format validator
├── requirements.txt
└── README.md
```

## Two design iterations

The system went through two design iterations. An early prototype (`src/scoring/`, `src/ranking/`, `src/matching/`) implemented a 5-stage hybrid pipeline with dense retrieval, BM25, cross-encoder re-ranking, and an LLM evaluation pass. That pipeline produces strong rankings but calls an LLM per candidate batch, which cannot fit the 100k-candidate, 5-minute, CPU-only, no-network constraint enforced at submission time. The submission-facing system in `src/data_loader.py` and `src/runner.py` distills the same reasoning into a fast deterministic scorer that runs entirely on precomputed fields, satisfying the compute budget while keeping the scoring logic auditable. We kept the prototype in the repo rather than deleting it because it documents the reasoning behind the simpler system's design choices.

## How candidates are scored

Each candidate gets a single composite score from seven signals, computed entirely from fields already present in `candidates.jsonl`:

| Signal | Weight | What it captures |
|---|---|---|
| AI/ML skill match | 40% | Skills list matched against a curated AI/ML vocabulary, blended with a semantic scan of career history text so candidates who did the work without listing it as a skill still score correctly |
| Title relevance | 25% | Hard gate against clearly unrelated titles (Sales, HR, Project Manager, Frontend, etc.); strong match for ML/AI-specific titles |
| Experience fit | 18% | Peaks at 6-8 years per the JD's stated 5-9 year range, tapering on both sides |
| Availability | 12% | Recruiter response rate, days since last active, open-to-work flag |
| Company quality | 3% | Penalizes careers spent entirely at consulting firms, rewards tier-1 product companies |
| Location | 2% | India-based preferred, Pune/Noida/Delhi NCR prioritized per the JD |

Two hard gates cap the final score regardless of the weighted sum: a disqualifying title caps the score at 0.35, and near-zero recruiter responsiveness (or six-plus months of inactivity) caps it at 0.45. A candidate can have a high skill count and still rank low if either gate fires, which is what the JD's "down-weight unavailable candidates" instruction is asking for.

## Honeypot detection

The dataset contains profiles with internally inconsistent claims by design. `detect_honeypot()` in `src/data_loader.py` flags a candidate when any of the following hold:

- a skill is marked "expert" proficiency with under 6 months of recorded usage
- claimed total years of experience exceeds the sum of career history durations by more than 3x
- a single role's duration exceeds total claimed experience
- a platform metric falls outside its valid range (e.g. response rate above 1.0)

Flagged candidates are forced to a score of 0.001, pushing them to the bottom of the ranking without needing to special-case them elsewhere in the pipeline. Thresholds were calibrated against the full 100k pool to land close to the dataset's documented honeypot count while minimizing false positives on legitimate profiles.

## Ties and determinism

Candidates are sorted by `(-score, candidate_id)`, so identical scores break deterministically by candidate ID rather than by insertion order, satisfying the submission spec's tie-breaking requirement.

