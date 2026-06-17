# NOTE: Not used at submission time. See README "Two design iterations" —
# this is the generic multi-signal scorer from our first pipeline design.
# The live ranker is src/data_loader.py + src/runner.py.
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CandidateScore:
    candidate_id: str
    hard_fit_score: float = 0.0
    skill_gap_score: float = 0.0
    growth_potential_score: float = 0.0
    career_momentum_score: float = 0.0
    achievement_impact_score: float = 0.0
    cultural_alignment_score: float = 0.0
    behavioral_signal_score: float = 0.0
    weighted_total: float = 0.0
    recruiter_confidence: float = 0.0
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    ranking_reasons: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0
    bm25_score: float = 0.0
    cross_encoder_score: float = 0.0
    llm_rank: int = 0


SCORE_WEIGHTS = {
    "hard_fit_score": 0.28,
    "skill_gap_score": 0.15,
    "career_momentum_score": 0.15,
    "achievement_impact_score": 0.15,
    "growth_potential_score": 0.12,
    "behavioral_signal_score": 0.08,
    "cultural_alignment_score": 0.07,
}


def compute_hard_fit_score(candidate_data: Dict, jd_data: Dict) -> Tuple[float, List[str], List[str]]:
    """Check how many required skills the candidate has."""
    required = set(s.lower() for s in jd_data.get("required_skills", []))
    candidate_skills = set()

    skills = candidate_data.get("skills", {})
    for skill_list in skills.values():
        candidate_skills.update(s.lower() for s in skill_list)

    for exp in candidate_data.get("experience", []):
        candidate_skills.update(s.lower() for s in exp.get("tech_stack", []))

    for proj in candidate_data.get("projects", []):
        candidate_skills.update(s.lower() for s in proj.get("tech_stack", []))

    if not required:
        return 50.0, [], []

    matched = required & candidate_skills
    missing = required - candidate_skills

    # Fuzzy match — check substrings
    extra_matched = set()
    still_missing = set()
    for skill in missing:
        found = False
        for cs in candidate_skills:
            if skill in cs or cs in skill:
                extra_matched.add(skill)
                found = True
                break
        if not found:
            still_missing.add(skill)

    total_matched = len(matched) + len(extra_matched)
    score = (total_matched / len(required)) * 100
    return score, list(matched | extra_matched), list(still_missing)


def compute_career_momentum_score(candidate_data: Dict, jd_data: Dict) -> float:
    """Score based on career trajectory, promotion speed, and company quality."""
    score = 50.0
    cs = candidate_data.get("career_signals", {})

    trajectory_map = {"ascending": 25, "pivoting": 5, "flat": 0, "descending": -20}
    score += trajectory_map.get(cs.get("career_trajectory", "flat"), 0)

    velocity_map = {"fast": 15, "normal": 0, "slow": -10}
    score += velocity_map.get(cs.get("promotion_velocity", "normal"), 0)

    prestige_map = {"tier1": 15, "tier2": 8, "tier3": 0}
    score += prestige_map.get(cs.get("company_prestige_avg", "tier3"), 0)

    switching_map = {"stable": 5, "normal": 0, "frequent": -15}
    score += switching_map.get(cs.get("job_switching_pattern", "normal"), 0)

    experiences = candidate_data.get("experience", [])
    if experiences and experiences[0].get("is_current", False):
        score += 5

    return max(0.0, min(100.0, score))


def compute_growth_potential_score(candidate_data: Dict, jd_data: Dict) -> float:
    """Estimate how much the candidate can grow into and beyond this role."""
    score = 40.0
    cs = candidate_data.get("career_signals", {})
    bs = candidate_data.get("behavioral_signals", {})

    learning_map = {"high": 20, "medium": 10, "low": 0}
    score += learning_map.get(cs.get("learning_velocity", "medium"), 10)

    behavioral_boosts = {
        "open_source_contributions": 8,
        "publications": 12,
        "patents": 10,
        "speaking_engagements": 7,
        "mentorship": 6,
        "entrepreneurial": 8
    }
    for signal, boost in behavioral_boosts.items():
        if bs.get(signal, False):
            score += boost

    level_map = {"junior": 1, "mid": 2, "senior": 3, "staff": 4, "principal": 5, "exec": 6}
    candidate_level = level_map.get(cs.get("career_level", "mid"), 2)
    required_level = level_map.get(jd_data.get("seniority", "mid"), 2)
    diff = required_level - candidate_level

    if diff == 1:
        score += 15   # Stretching up — high potential
    elif diff == 0:
        score += 10   # Perfect level match
    elif diff == -1:
        score += 3    # Slightly overqualified
    elif diff >= 2:
        score -= 20   # Too junior
    elif diff <= -2:
        score -= 10   # Overqualified — flight risk

    return max(0.0, min(100.0, score))


def compute_achievement_impact_score(candidate_data: Dict) -> float:
    """Score based on quantified, high-impact achievements."""
    achievements = candidate_data.get("achievements", [])

    if not achievements:
        all_exp_achievements = []
        for exp in candidate_data.get("experience", []):
            all_exp_achievements.extend(exp.get("achievements", []))
        if not all_exp_achievements:
            return 25.0
        return 40.0  # Has achievements but not structured

    score = 0.0
    magnitude_map = {"exceptional": 25, "large": 18, "medium": 12, "small": 6}
    impact_map = {"revenue": 1.2, "scale": 1.1, "performance": 1.0, "cost": 1.0, "recognition": 0.9}

    for ach in achievements[:5]:
        base = magnitude_map.get(ach.get("magnitude", "small"), 6)
        multiplier = impact_map.get(ach.get("impact_type", "performance"), 1.0)
        if ach.get("is_quantified", False):
            multiplier *= 1.3
        score += base * multiplier

    return min(100.0, score)


def compute_behavioral_signal_score(candidate_data: Dict) -> float:
    """Score extra-curricular technical signals like OSS, patents, publications."""
    bs = candidate_data.get("behavioral_signals", {})
    score = 25.0

    weights = {
        "publications": 20,
        "patents": 15,
        "open_source_contributions": 15,
        "speaking_engagements": 12,
        "entrepreneurial": 10,
        "mentorship": 8
    }
    for signal, weight in weights.items():
        if bs.get(signal, False):
            score += weight

    return min(100.0, score)


def compute_weighted_total(scores: CandidateScore) -> float:
    return (
        scores.hard_fit_score * SCORE_WEIGHTS["hard_fit_score"] +
        (100 - scores.skill_gap_score) * SCORE_WEIGHTS["skill_gap_score"] +
        scores.career_momentum_score * SCORE_WEIGHTS["career_momentum_score"] +
        scores.achievement_impact_score * SCORE_WEIGHTS["achievement_impact_score"] +
        scores.growth_potential_score * SCORE_WEIGHTS["growth_potential_score"] +
        scores.behavioral_signal_score * SCORE_WEIGHTS["behavioral_signal_score"] +
        scores.cultural_alignment_score * SCORE_WEIGHTS["cultural_alignment_score"]
    )


def score_candidate(candidate_data: Dict, jd_data: Dict, candidate_id: str) -> CandidateScore:
    """Run all scoring dimensions for one candidate against one JD."""
    scores = CandidateScore(candidate_id=candidate_id)

    scores.hard_fit_score, matched, missing = compute_hard_fit_score(candidate_data, jd_data)
    scores.missing_skills = missing

    total_required = len(jd_data.get("required_skills", []))
    if total_required > 0:
        scores.skill_gap_score = (len(missing) / total_required) * 100
    else:
        scores.skill_gap_score = 0.0

    scores.career_momentum_score = compute_career_momentum_score(candidate_data, jd_data)
    scores.achievement_impact_score = compute_achievement_impact_score(candidate_data)
    scores.growth_potential_score = compute_growth_potential_score(candidate_data, jd_data)
    scores.behavioral_signal_score = compute_behavioral_signal_score(candidate_data)
    scores.cultural_alignment_score = 50.0  # Default; upgraded by LLM stage

    scores.weighted_total = compute_weighted_total(scores)
    return scores