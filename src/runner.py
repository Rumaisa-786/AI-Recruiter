"""
Main runner — processes hackathon dataset and produces submission CSV.
Run: python src/runner.py
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm
from src.data_loader import score_candidate_for_jd, load_candidates_jsonl, load_candidates_json


def run(use_sample: bool = False, max_candidates: int = None):
    """
    Score all candidates and produce ranked submission CSV.
    
    Args:
        use_sample: Use sample_candidates.json instead of full candidates.jsonl
        max_candidates: Limit number of candidates (for testing)
    """
    print("=" * 60)
    print("AI CANDIDATE RANKING SYSTEM")
    print("Job: Senior AI Engineer — Redrob AI")
    print("=" * 60)

    # Choose data source
    if use_sample:
        data_path = "data/raw/sample_candidates.json"
        print(f"\nLoading sample candidates from {data_path}...")
        candidates = load_candidates_json(data_path)
        if max_candidates:
            candidates = candidates[:max_candidates]
        print(f"Loaded {len(candidates)} candidates")
    else:
        data_path = "data/raw/candidates.jsonl"
        print(f"\nStreaming full dataset from {data_path}...")
        candidates = list(tqdm(
            load_candidates_jsonl(data_path, max_candidates=max_candidates),
            desc="Loading"
        ))
        print(f"Loaded {len(candidates)} candidates")

    # Score all candidates
    print("\nScoring candidates...")
    scored = []
    for candidate in tqdm(candidates, desc="Scoring"):
        result = score_candidate_for_jd(candidate)
        scored.append(result)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    
# Only keep top 100 for submission
    scored = scored[:100]

    # Assign ranks 1-100
    for rank, candidate in enumerate(scored):
        candidate["rank"] = rank + 1

    # Save submission CSV
    Path("outputs").mkdir(exist_ok=True)
    output_path = "outputs/submission.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for c in scored:
            writer.writerow([
                c["candidate_id"],
                c["rank"],
                c["score"],
                c["reasoning"]
            ])

    print(f"\nSubmission saved to {output_path}")

    # Save detailed report
    report_path = "outputs/detailed_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(scored[:50], f, indent=2)
    print(f"Top 50 detailed report saved to {report_path}")

    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 CANDIDATES")
    print("=" * 60)
    for c in scored[:10]:
        b = c["_breakdown"]
        print(f"\n#{c['rank']} {c['candidate_id']}")
        print(f"   Score      : {c['score']}")
        print(f"   Reasoning  : {c['reasoning']}")
        print(f"   AI Skills  : {b['ai_skill_count']} ({', '.join(b['matched_ai_skills'][:3])}...)")
        print(f"   Experience : {b['years']} yrs")
        print(f"   Response   : {b['response_rate']}")
        print(f"   Location   : {b['location']}")
        print(f"   Notice     : {b['notice_days']} days")
        print(f"   Platform   : {b['platform_score']}")
    print(f"\nTotal candidates ranked: {len(scored)}")
    return scored


if __name__ == "__main__":
    import sys
    use_sample = "--sample" in sys.argv
    run(use_sample=use_sample)