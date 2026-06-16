#!/usr/bin/env python3
"""
AI Candidate Ranking System
Main entry point — run this file to build index or rank candidates.

Usage:
  python main.py build <candidates_dir>
  python main.py rank <jd_file> [top_n]
  python main.py demo
"""

import json
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()


def build_index(candidates_dir: str, index_dir: str = "outputs/index"):
    """Step 1 — Process all candidate files and build FAISS + BM25 indexes."""
    from src.ingestion.resume_parser import parse_resume
    from src.understanding.llm_extractor import extract_candidate_profile
    from src.matching.embedder import (
        CandidateVectorStore, embed_texts,
        create_candidate_text_representation
    )
    from src.matching.sparse_retriever import BM25Retriever

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    candidates_path = Path(candidates_dir)

    # Support JSON, PDF, TXT, DOCX
    all_files = (
        list(candidates_path.glob("*.json")) +
        list(candidates_path.glob("*.pdf")) +
        list(candidates_path.glob("*.txt")) +
        list(candidates_path.glob("*.docx"))
    )

    if not all_files:
        logger.error(f"No candidate files found in {candidates_dir}")
        return

    logger.info(f"Found {len(all_files)} candidate files")

    all_candidate_data = {}
    all_texts = {}

    for f in tqdm(all_files, desc="Processing candidates"):
        cid = f.stem

        if f.suffix == ".json":
            with open(f, encoding="utf-8") as fp:
                raw = json.load(fp)

            # If already extracted (has career_signals), use directly
            if "career_signals" in raw:
                raw["candidate_id"] = cid
                all_candidate_data[cid] = raw
            else:
                # Extract with LLM
                resume_text = raw.get("raw_text", raw.get("text", str(raw)))
                logger.info(f"Extracting profile for {cid}...")
                extracted = extract_candidate_profile(resume_text)
                extracted["candidate_id"] = cid
                extracted["name"] = raw.get("name", cid)
                all_candidate_data[cid] = extracted
        else:
            # PDF/TXT/DOCX — parse then extract
            logger.info(f"Parsing resume: {f.name}")
            profile = parse_resume(str(f), cid)
            extracted = extract_candidate_profile(profile.raw_text)
            extracted["candidate_id"] = cid
            extracted["name"] = profile.name or cid
            all_candidate_data[cid] = extracted

        all_texts[cid] = create_candidate_text_representation(all_candidate_data[cid])

    logger.info("Creating embeddings...")
    candidates_list = list(all_candidate_data.values())
    texts_list = [all_texts[c["candidate_id"]] for c in candidates_list]

    embeddings = embed_texts(texts_list, batch_size=16)

    logger.info("Building FAISS vector store...")
    vector_store = CandidateVectorStore()
    vector_store.add_candidates(candidates_list, embeddings)
    vector_store.save(index_dir)

    logger.info("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build_index(candidates_list, texts_list)

    with open(f"{index_dir}/candidate_data.json", "w", encoding="utf-8") as f:
        json.dump(all_candidate_data, f, indent=2)

    with open(f"{index_dir}/candidate_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, indent=2)

    with open(f"{index_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    logger.info(f"Index built for {len(candidates_list)} candidates → saved to {index_dir}")


def rank_candidates(jd_path: str, index_dir: str = "outputs/index", top_n: int = 10):
    """Step 2 — Load index and rank candidates for a given job description."""
    from src.understanding.llm_extractor import extract_jd_profile
    from src.matching.embedder import CandidateVectorStore
    from src.ranking.pipeline import run_full_pipeline
    from src.explainability.explainer import (
        generate_recruiter_report, export_ranked_csv, print_summary
    )

    # Load JD
    jd_path = Path(jd_path)
    if jd_path.suffix == ".json":
        with open(jd_path, encoding="utf-8") as f:
            raw = json.load(f)
        jd_text = raw.get("text", raw.get("raw_text", str(raw)))
    else:
        jd_text = jd_path.read_text(encoding="utf-8")

    logger.info("Extracting JD profile with LLM...")
    jd_data = extract_jd_profile(jd_text)
    jd_data["raw_text"] = jd_text

    # Load indexes
    logger.info("Loading indexes...")
    vector_store = CandidateVectorStore()
    vector_store.load(index_dir)

    with open(f"{index_dir}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(f"{index_dir}/candidate_data.json", encoding="utf-8") as f:
        candidate_data_map = json.load(f)

    with open(f"{index_dir}/candidate_texts.json", encoding="utf-8") as f:
        candidate_texts = json.load(f)

    # Run pipeline
    ranked = run_full_pipeline(
        jd_data=jd_data,
        candidate_data_map=candidate_data_map,
        vector_store=vector_store,
        bm25=bm25,
        candidate_texts=candidate_texts,
        top_n=top_n
    )

    # Save outputs
    Path("outputs").mkdir(exist_ok=True)

    report = generate_recruiter_report(ranked, candidate_data_map, jd_data)

    with open("outputs/recruiter_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    export_ranked_csv(ranked, "outputs/ranked_candidates.csv")
    print_summary(ranked, jd_data)

    logger.info("Done! Outputs saved to outputs/")
    return ranked


def run_demo():
    """Quick demo with synthetic data — no real resumes needed."""
    import json
    from pathlib import Path

    logger.info("Running demo with synthetic candidates...")

    Path("data/candidates").mkdir(parents=True, exist_ok=True)
    Path("data/jobs").mkdir(parents=True, exist_ok=True)

    # Create 5 synthetic candidates
    candidates = [
        {
            "candidate_id": "alice_chen",
            "name": "Alice Chen",
            "skills": {"technical": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
                       "soft": ["Leadership", "Communication"],
                       "tools": ["Git", "Kubernetes"],
                       "domains": ["Backend", "Cloud"]},
            "experience": [{
                "company": "Google", "title": "Senior Software Engineer",
                "duration_months": 36, "start_year": 2021, "end_year": 2024,
                "is_current": True, "level": "senior",
                "responsibilities": ["Led backend API team"],
                "achievements": ["Reduced latency by 40%", "Scaled system to 10M users"],
                "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Kubernetes"],
                "team_size_managed": 5, "scope": "team"
            }],
            "education": [{"degree": "B.Tech", "field": "Computer Science",
                           "institution": "IIT Bombay", "tier": "tier1", "year": 2018}],
            "projects": [{"name": "OpenAPI Gateway",
                          "description": "High-performance API gateway",
                          "impact": "Used by 500+ developers",
                          "tech_stack": ["Python", "FastAPI"],
                          "complexity": "high", "is_open_source": True}],
            "career_signals": {
                "total_experience_years": 6.0, "career_level": "senior",
                "career_trajectory": "ascending", "promotion_velocity": "fast",
                "company_prestige_avg": "tier1", "job_switching_pattern": "stable",
                "leadership_signals": ["Led team of 5"], "learning_velocity": "high",
                "specialization": "specialist"
            },
            "achievements": [{
                "description": "Reduced API latency by 40%",
                "impact_type": "performance", "magnitude": "large", "is_quantified": True
            }],
            "behavioral_signals": {
                "open_source_contributions": True, "publications": False,
                "patents": False, "speaking_engagements": True,
                "mentorship": True, "entrepreneurial": False
            }
        },
        {
            "candidate_id": "bob_smith",
            "name": "Bob Smith",
            "skills": {"technical": ["Python", "Django", "MySQL"],
                       "soft": ["Teamwork"], "tools": ["Git"], "domains": ["Backend"]},
            "experience": [{
                "company": "Startup XYZ", "title": "Junior Developer",
                "duration_months": 18, "start_year": 2023, "end_year": 2024,
                "is_current": True, "level": "junior",
                "responsibilities": ["Built REST APIs"],
                "achievements": ["Delivered 3 features on time"],
                "tech_stack": ["Python", "Django", "MySQL"],
                "team_size_managed": 0, "scope": "individual"
            }],
            "education": [{"degree": "B.Sc", "field": "IT",
                           "institution": "Local University", "tier": "tier3", "year": 2022}],
            "projects": [],
            "career_signals": {
                "total_experience_years": 1.5, "career_level": "junior",
                "career_trajectory": "ascending", "promotion_velocity": "normal",
                "company_prestige_avg": "tier3", "job_switching_pattern": "stable",
                "leadership_signals": [], "learning_velocity": "medium",
                "specialization": "generalist"
            },
            "achievements": [],
            "behavioral_signals": {
                "open_source_contributions": False, "publications": False,
                "patents": False, "speaking_engagements": False,
                "mentorship": False, "entrepreneurial": False
            }
        },
        {
            "candidate_id": "priya_nair",
            "name": "Priya Nair",
            "skills": {"technical": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
                       "soft": ["Problem Solving", "Mentoring"],
                       "tools": ["Git", "CI/CD"], "domains": ["Backend", "DevOps"]},
            "experience": [
                {
                    "company": "Microsoft", "title": "Software Engineer II",
                    "duration_months": 24, "start_year": 2022, "end_year": 2024,
                    "is_current": True, "level": "mid",
                    "responsibilities": ["Built microservices"],
                    "achievements": ["Improved DB query speed by 60%"],
                    "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Redis"],
                    "team_size_managed": 0, "scope": "team"
                }
            ],
            "education": [{"degree": "M.Tech", "field": "Computer Science",
                           "institution": "IIT Delhi", "tier": "tier1", "year": 2020}],
            "projects": [{"name": "Cache Layer",
                          "description": "Distributed caching for PostgreSQL",
                          "impact": "60% query speed improvement",
                          "tech_stack": ["Python", "Redis", "PostgreSQL"],
                          "complexity": "high", "is_open_source": False}],
            "career_signals": {
                "total_experience_years": 4.0, "career_level": "mid",
                "career_trajectory": "ascending", "promotion_velocity": "normal",
                "company_prestige_avg": "tier1", "job_switching_pattern": "stable",
                "leadership_signals": ["Mentored 2 interns"],
                "learning_velocity": "high", "specialization": "specialist"
            },
            "achievements": [{
                "description": "Improved DB performance by 60%",
                "impact_type": "performance", "magnitude": "large", "is_quantified": True
            }],
            "behavioral_signals": {
                "open_source_contributions": False, "publications": True,
                "patents": False, "speaking_engagements": False,
                "mentorship": True, "entrepreneurial": False
            }
        }
    ]

    # Save synthetic candidates
    for c in candidates:
        cid = c["candidate_id"]
        with open(f"data/candidates/{cid}.json", "w") as f:
            json.dump(c, f, indent=2)

    # Sample JD
    jd = {"text": """Senior Python Developer

We are looking for a Senior Python Developer with 5+ years of experience.

Required Skills: Python, FastAPI, PostgreSQL, Docker
Preferred Skills: Redis, Kubernetes, AWS
Experience: 5+ years in backend development
Education: Bachelor's degree in Computer Science or related field

Responsibilities:
- Design and build high-performance REST APIs
- Lead a small team of engineers
- Optimize database queries and system performance
- Mentor junior developers

We value engineers who are proactive, have a track record of impact,
and enjoy working in a collaborative fast-paced environment."""}

    with open("data/jobs/senior_python_dev.json", "w") as f:
        json.dump(jd, f, indent=2)

    logger.info("Demo data created. Building index...")
    build_index("data/candidates")

    logger.info("Ranking candidates...")
    rank_candidates("data/jobs/senior_python_dev.json", top_n=3)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main.py demo                         — Run with sample data")
        print("  python main.py build <candidates_dir>       — Build index")
        print("  python main.py rank <jd_file> [top_n]      — Rank candidates")
        sys.exit(0)

    command = sys.argv[1]

    if command == "demo":
        run_demo()

    elif command == "build":
        if len(sys.argv) < 3:
            print("Usage: python main.py build <candidates_dir>")
            sys.exit(1)
        build_index(sys.argv[2])

    elif command == "rank":
        if len(sys.argv) < 3:
            print("Usage: python main.py rank <jd_file> [top_n]")
            sys.exit(1)
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        rank_candidates(sys.argv[2], top_n=top_n)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)