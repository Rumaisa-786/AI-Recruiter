import json
import csv
from typing import List, Dict, Tuple
from src.scoring.scorer import CandidateScore


def generate_recruiter_report(
    ranked_candidates: List[Tuple[str, CandidateScore]],
    candidate_data_map: Dict,
    jd_data: Dict
) -> Dict:
    """Generate full human-readable recruiter report grounded in extracted data."""
    report = {
        "job_title": jd_data.get("title", "Unknown Role"),
        "seniority": jd_data.get("seniority", ""),
        "total_candidates_evaluated": len(ranked_candidates),
        "shortlist": []
    }

    for rank, (cid, score) in enumerate(ranked_candidates):
        cdata = candidate_data_map.get(cid, {})
        cs = cdata.get("career_signals", {})

        strengths = score.strengths or _extract_strengths(score, cdata, jd_data)
        gaps = score.gaps or _extract_gaps(score, cdata, jd_data)
        reasons = score.ranking_reasons or _generate_reasons(score, cdata, jd_data)

        candidate_report = {
            "rank": rank + 1,
            "candidate_id": cid,
            "name": cdata.get("name", f"Candidate {cid}"),

            "scores": {
                "overall": round(score.weighted_total, 1),
                "hard_fit": round(score.hard_fit_score, 1),
                "career_momentum": round(score.career_momentum_score, 1),
                "achievement_impact": round(score.achievement_impact_score, 1),
                "growth_potential": round(score.growth_potential_score, 1),
                "behavioral_signals": round(score.behavioral_signal_score, 1),
                "cultural_alignment": round(score.cultural_alignment_score, 1),
            },

            "signal_scores": {
                "semantic_similarity": round(score.semantic_similarity, 3),
                "cross_encoder": round(score.cross_encoder_score, 3),
                "llm_rank": score.llm_rank,
            },

            "recruiter_confidence": score.recruiter_confidence,

            "strengths": strengths,
            "gaps": gaps,
            "missing_skills": score.missing_skills[:5],
            "key_ranking_reason": reasons[0] if reasons else "",

            "profile_summary": {
                "career_level": cs.get("career_level", "unknown"),
                "experience_years": cs.get("total_experience_years", 0),
                "trajectory": cs.get("career_trajectory", "unknown"),
                "specialization": cs.get("specialization", "unknown"),
                "company_tier": cs.get("company_prestige_avg", "unknown"),
            },

            "interview_recommendation": _get_recommendation(score, rank),
        }

        report["shortlist"].append(candidate_report)

    return report


def _generate_reasons(score: CandidateScore, cdata: Dict, jd_data: Dict) -> List[str]:
    reasons = []
    cs = cdata.get("career_signals", {})

    if score.hard_fit_score >= 80:
        reasons.append(
            f"Covers {score.hard_fit_score:.0f}% of required skills — strong technical match"
        )
    elif score.hard_fit_score >= 60:
        reasons.append(
            f"Covers {score.hard_fit_score:.0f}% of required skills with transferable expertise"
        )
    else:
        reasons.append(
            f"Partial skill match at {score.hard_fit_score:.0f}% — review gaps before interviewing"
        )

    if cs.get("career_trajectory") == "ascending":
        reasons.append("Consistent upward trajectory with growing scope and responsibility")

    if score.achievement_impact_score >= 70:
        reasons.append("Strong track record of quantified, high-impact achievements")

    if score.behavioral_signal_score >= 60:
        reasons.append("Active technical community presence signals initiative and depth")

    return reasons


def _extract_strengths(score: CandidateScore, cdata: Dict, jd_data: Dict) -> List[str]:
    strengths = []
    cs = cdata.get("career_signals", {})
    bs = cdata.get("behavioral_signals", {})

    if score.hard_fit_score >= 75:
        strengths.append("Strong alignment with required technical skill set")

    if cs.get("career_trajectory") == "ascending":
        strengths.append("Clear upward career progression with increasing responsibilities")

    if cs.get("promotion_velocity") == "fast":
        strengths.append("Fast promotion velocity — above average growth speed")

    if score.achievement_impact_score >= 65:
        strengths.append("Demonstrated history of measurable, high-impact work")

    if cs.get("learning_velocity") == "high":
        strengths.append("High learning velocity — adapts quickly to new technologies")

    if bs.get("open_source_contributions"):
        strengths.append("Open source contributor — signals technical initiative")

    if bs.get("publications"):
        strengths.append("Published researcher — strong domain knowledge depth")

    if cs.get("company_prestige_avg") == "tier1":
        strengths.append("Experience at top-tier companies — high bar environment")

    return strengths[:4]


def _extract_gaps(score: CandidateScore, cdata: Dict, jd_data: Dict) -> List[str]:
    gaps = []
    cs = cdata.get("career_signals", {})

    if score.missing_skills:
        gaps.append(f"Missing required skills: {', '.join(score.missing_skills[:3])}")

    if cs.get("job_switching_pattern") == "frequent":
        gaps.append("Frequent job changes — assess commitment and retention risk")

    if score.growth_potential_score < 40:
        gaps.append("Limited signals of self-directed learning or upskilling")

    if cs.get("career_trajectory") == "descending":
        gaps.append("Declining career trajectory — investigate reasons")

    level_map = {"junior": 1, "mid": 2, "senior": 3, "staff": 4, "principal": 5, "exec": 6}
    candidate_level = level_map.get(cs.get("career_level", "mid"), 2)
    required_level = level_map.get(jd_data.get("seniority", "mid"), 2)

    if required_level - candidate_level >= 2:
        gaps.append("Significantly under-leveled for this role — may struggle with scope")
    elif candidate_level - required_level >= 2:
        gaps.append("Overqualified — assess motivation and long-term interest in role")

    return gaps[:3]


def _get_recommendation(score: CandidateScore, rank: int) -> str:
    if rank == 0 and score.weighted_total >= 70:
        return "STRONGLY RECOMMEND — Top candidate, prioritize interview"
    elif rank < 3 and score.weighted_total >= 65:
        return "RECOMMEND — Strong fit, schedule interview"
    elif score.weighted_total >= 50:
        return "CONSIDER — Decent fit, review gaps first"
    else:
        return "OPTIONAL — Only if top candidates are unavailable"


def export_ranked_csv(
    ranked_candidates: List[Tuple[str, CandidateScore]],
    output_path: str
):
    """Export final ranked output in CSV format for hackathon submission."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "candidate_id", "overall_score",
            "hard_fit", "career_momentum", "achievement_impact",
            "growth_potential", "recruiter_confidence",
            "missing_skills", "key_reason"
        ])

        for rank, (cid, score) in enumerate(ranked_candidates):
            writer.writerow([
                rank + 1,
                cid,
                round(score.weighted_total, 2),
                round(score.hard_fit_score, 2),
                round(score.career_momentum_score, 2),
                round(score.achievement_impact_score, 2),
                round(score.growth_potential_score, 2),
                round(score.recruiter_confidence, 2),
                "; ".join(score.missing_skills[:3]),
                score.ranking_reasons[0] if score.ranking_reasons else ""
            ])

    print(f"CSV saved to {output_path}")


def print_summary(ranked_candidates: List[Tuple[str, CandidateScore]], jd_data: Dict):
    """Print a clean terminal summary of results."""
    print("\n" + "=" * 65)
    print(f"TOP CANDIDATES FOR: {jd_data.get('title', 'Unknown Role').upper()}")
    print("=" * 65)

    for rank, (cid, score) in enumerate(ranked_candidates):
        print(f"\n#{rank + 1} — {cid}")
        print(f"   Overall Score      : {score.weighted_total:.1f}/100")
        print(f"   Recruiter Confidence: {score.recruiter_confidence:.1f}%")
        print(f"   Hard Fit           : {score.hard_fit_score:.1f}%")
        print(f"   Career Momentum    : {score.career_momentum_score:.1f}%")
        print(f"   Growth Potential   : {score.growth_potential_score:.1f}%")
        if score.missing_skills:
            print(f"   Missing Skills     : {', '.join(score.missing_skills[:3])}")
        if score.ranking_reasons:
            print(f"   Key Reason         : {score.ranking_reasons[0]}")
        print(f"   Recommendation     : {_get_recommendation(score, rank)}")

    print("\n" + "=" * 65)