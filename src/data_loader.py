"""
Custom data loader for India Runs Data & AI Challenge dataset.
Reads candidates.jsonl and sample_candidates.json directly.
"""

import json
from pathlib import Path
from typing import List, Dict, Generator
from datetime import datetime, date
from tqdm import tqdm


# STRICT AI/ML core skills — only real AI/ML skills count
AI_CORE_SKILLS = {
    # Embeddings & Vector Search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector database", "vector search", "semantic search",
    "hybrid search", "dense retrieval", "embeddings", "sentence-transformers",
    "bge", "e5",

    # LLMs & Fine-tuning
    "llm", "large language model", "fine-tuning llms", "lora", "qlora",
    "peft", "rlhf", "instruction tuning", "gpt", "llama", "mistral",
    "gemini", "bert", "transformers", "rag",
    "retrieval augmented generation", "langchain", "llamaindex",

    # Core ML
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "information retrieval",
    "recommendation systems", "learning to rank", "pytorch", "tensorflow",
    "hugging face", "xgboost", "lightgbm",

    # MLOps & Evaluation
    "mlflow", "weights & biases", "wandb", "bentoml", "onnx",
    "model evaluation", "a/b testing", "ndcg",

    # AI-specific tools in dataset
    "image classification", "speech recognition", "tts", "gans",
    "statistical modeling", "fine-tuning", "chromadb",
}

# These skill names should NOT count as AI skills
NON_AI_SKILLS = {
    "postgresql", "mysql", "mongodb", "kafka", "spark", "airflow",
    "docker", "kubernetes", "aws", "gcp", "azure", "flask", "django",
    "react", "angular", "vue", "tailwind", "photoshop", "figma",
    "data pipelines", "apache beam", "snowflake", "dbt", "sql",
    "java", "go", "rust", "swift", "kotlin", "c++",
}

# Titles that are clearly NOT AI Engineer roles — heavy penalty
NON_AI_TITLES = {
    "marketing", "accountant", "sales", "customer support",
    "graphic designer", "civil engineer", "mechanical engineer",
    "content writer", "hr manager", "operations manager",
    "project manager", "frontend engineer", "mobile developer",
    "java developer", "android developer", "ios developer",
    "devops", "qa engineer", "test engineer", "business analyst",
    "ui developer", "ux designer", "product manager",
}

# Good AI-related titles — positive signal
GOOD_AI_TITLES = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist",
    "recommendation", "search engineer", "ranking engineer",
    "deep learning", "computer vision", "backend engineer",
    "software engineer", "full stack", "data engineer",
    "analytics engineer", "platform engineer",
}

# Consulting firms — red flag if entire career
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "hexaware", "ltimindtree", "persistent systems",
    "niit", "l&t infotech",
}

# Tier 1 product companies — positive signal
TIER1_COMPANIES = {
    "google", "microsoft", "amazon", "meta", "apple", "netflix",
    "openai", "anthropic", "deepmind", "nvidia", "adobe", "salesforce",
    "flipkart", "meesho", "razorpay", "zepto", "swiggy", "zomato",
    "cred", "phonepe", "paytm", "groww", "navi", "browserstack",
    "atlassian", "uber", "linkedin", "stripe", "twilio",
}

# India locations preferred per JD
INDIA_TOP_LOCATIONS = {"pune", "noida", "hyderabad", "delhi", "ncr", "gurugram"}
INDIA_OK_LOCATIONS = {"mumbai", "bangalore", "bengaluru", "chennai", "india"}


def load_candidates_jsonl(path: str, max_candidates: int = None) -> Generator:
    with open(path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
                count += 1
                if max_candidates and count >= max_candidates:
                    break


def load_candidates_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_ai_skills(skills: List[Dict]) -> tuple:
    """
    Strictly count AI/ML skills only.
    Excludes infrastructure, frontend, and database skills.
    """
    proficiency_scores = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    matched = []
    proficiency_total = 0

    for skill in skills:
        skill_name = skill.get("name", "").lower().strip()

        # Skip if it's a known non-AI skill
        if skill_name in NON_AI_SKILLS:
            continue
        if any(non_ai in skill_name for non_ai in NON_AI_SKILLS):
            continue

        # Check if it matches AI skills
        is_ai = False
        for ai_skill in AI_CORE_SKILLS:
            if ai_skill in skill_name or skill_name in ai_skill:
                is_ai = True
                break

        if is_ai:
            matched.append(skill)
            proficiency_total += proficiency_scores.get(
                skill.get("proficiency", "beginner"), 1
            )

    avg_proficiency = proficiency_total / len(matched) if matched else 0
    return len(matched), matched, avg_proficiency


def get_title_score(current_title: str) -> float:
    """Score based on job title relevance to AI Engineering."""
    title_lower = current_title.lower()

    # Immediate disqualifiers — return very low score
    hard_disqualify = [
        "marketing", "accountant", "sales", "customer support",
        "graphic designer", "civil engineer", "mechanical engineer",
        "content writer", "hr", "operations manager",
        "project manager", "frontend", "mobile developer",
        "java developer", "android", "ios developer",
        "qa engineer", "test engineer", "business analyst",
        "ui developer", "ux", "product manager",
        ".net developer", "net developer", "devops",
        "cloud engineer", "network engineer", "security engineer",
    ]
    for bad in hard_disqualify:
        if bad in title_lower:
            return 0.1  # Near-zero — wrong role entirely

    # Strong positive matches
    strong_match = [
        "machine learning", "ml engineer", "ai engineer",
        "nlp engineer", "research engineer", "applied scientist",
        "recommendation", "search engineer", "ranking engineer",
        "deep learning engineer", "data scientist",
    ]
    for good in strong_match:
        if good in title_lower:
            return 1.0

    # Moderate matches — can do AI work
    moderate_match = [
        "backend engineer", "software engineer", "full stack",
        "data engineer", "platform engineer", "analytics engineer",
        "senior engineer", "staff engineer",
    ]
    for ok in moderate_match:
        if ok in title_lower:
            return 0.75

    return 0.5  # Unknown title — neutral


def get_experience_fit(years: float) -> float:
    """JD sweet spot is 5-9 years, peaks at 6-8."""
    if years < 2:
        return 0.1
    elif years < 4:
        return 0.4
    elif years < 5:
        return 0.7
    elif 5 <= years <= 9:
        if 6 <= years <= 8:
            return 1.0
        return 0.9
    elif years <= 11:
        return 0.7
    else:
        return 0.5


def get_availability_score(redrob: Dict) -> float:
    """
    Per JD: candidates inactive 6+ months with low response rate
    are not actually available. Penalize very hard.
    """
    score = 1.0

    # Open to work flag
    if not redrob.get("open_to_work_flag", False):
        score *= 0.5

    # Last active date
    last_active = redrob.get("last_active_date", "")
    if last_active:
        try:
            last_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (date.today() - last_date).days
            if days_inactive > 180:
                score *= 0.15  # Basically unavailable
            elif days_inactive > 90:
                score *= 0.4
            elif days_inactive > 30:
                score *= 0.75
        except:
            pass

    # Recruiter response rate — CRITICAL signal per JD
    response_rate = redrob.get("recruiter_response_rate", 0)
    if response_rate < 0.05:
        score *= 0.1   # Near zero — completely unresponsive
    elif response_rate < 0.15:
        score *= 0.3   # Very poor
    elif response_rate < 0.30:
        score *= 0.55  # Below average
    elif response_rate < 0.50:
        score *= 0.75  # Average
    elif response_rate < 0.70:
        score *= 0.9   # Good
    else:
        score *= 1.0   # Excellent

    return min(1.0, score)


def get_company_quality(career: List[Dict]) -> tuple:
    """Returns (quality_score, is_consulting_only)."""
    if not career:
        return 0.5, False

    all_consulting = True
    total_score = 0.0

    for job in career:
        company = job.get("company", "").lower()
        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        is_tier1 = any(t1 in company for t1 in TIER1_COMPANIES)

        if not is_consulting:
            all_consulting = False

        if is_tier1:
            total_score += 3.0
        elif is_consulting:
            total_score += 0.5
        else:
            total_score += 1.5

    avg = total_score / len(career)
    if all_consulting:
        avg *= 0.3  # Heavy penalty per JD
        return avg, True

    return min(1.0, avg / 3.0), False


def get_location_score(profile: Dict) -> float:
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    if country != "india" and "india" not in location:
        return 0.3  # Outside India

    if any(loc in location for loc in INDIA_TOP_LOCATIONS):
        return 1.0  # Perfect — Pune/Noida/Hyderabad/Delhi
    if any(loc in location for loc in INDIA_OK_LOCATIONS):
        return 0.85  # Good India city

    return 0.7  # India but smaller city


def get_notice_score(notice_days: int) -> float:
    if notice_days == 0:
        return 1.0
    elif notice_days <= 30:
        return 1.0  # JD says they can buy out 30 days
    elif notice_days <= 60:
        return 0.7
    elif notice_days <= 90:
        return 0.5
    else:
        return 0.25


def get_platform_score(redrob: Dict) -> float:
    """GitHub activity + profile completeness + interview follow-through."""
    github = max(0, redrob.get("github_activity_score", 0)) / 100
    completeness = redrob.get("profile_completeness_score", 0) / 100
    interview_rate = redrob.get("interview_completion_rate", 0)
    saved_by_recruiters = min(1.0, redrob.get("saved_by_recruiters_30d", 0) / 10)

    return (
        github * 0.35 +
        completeness * 0.25 +
        interview_rate * 0.25 +
        saved_by_recruiters * 0.15
    )
def get_career_semantic_score(career: List[Dict]) -> float:
    """
    Read actual job descriptions for AI/ML work evidence.
    This catches candidates who DID the work but didn't list it as a skill.
    """
    if not career:
        return 0.0

    production_ai_signals = [
        "recommendation system", "ranking system", "search system",
        "embedding", "vector", "retrieval", "semantic search",
        "machine learning model", "ml model", "deployed model",
        "fine-tun", "llm", "language model", "neural network",
        "feature pipeline", "training pipeline", "inference",
        "a/b test", "model evaluation", "ndcg", "precision",
        "candidate ranking", "reranking", "bert", "transformer",
        "pytorch", "tensorflow", "hugging face", "scikit",
        "data science", "model serving", "mlops",
    ]

    score = 0.0
    total_months = sum(j.get("duration_months", 0) for j in career)

    for job in career:
        desc = job.get("description", "").lower()
        months = job.get("duration_months", 0)
        weight = months / total_months if total_months > 0 else 0

        matches = sum(1 for signal in production_ai_signals if signal in desc)
        job_score = min(1.0, matches / 5.0)
        score += job_score * weight

    return min(1.0, score)

def detect_honeypot(candidate: Dict) -> bool:
    """
    Detect honeypot candidates with subtly impossible profiles.
    Tightened thresholds based on dataset calibration (~80 expected honeypots).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})

    # Check 1: Expert proficiency with near-zero duration — strongest signal
    for skill in skills:
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        if proficiency == "expert" and duration < 6:
            return True

    # Check 2: Total years of experience wildly exceeds career history
    # Tightened from 2.5x to 3x to reduce false positives
    years_claimed = profile.get("years_of_experience", 0)
    total_months_in_history = sum(j.get("duration_months", 0) for j in career)
    years_from_history = total_months_in_history / 12.0

    if years_claimed > 0 and years_from_history > 0:
        if years_claimed > years_from_history * 3.0:
            return True

    # Check 3: A single role's duration exceeds total claimed experience
    # (i.e., a job that's literally longer than their whole career)
    for job in career:
        job_years = job.get("duration_months", 0) / 12.0
        if job_years > years_claimed + 2:  # +2 buffer to avoid false positives
            return True

    # Check 4: Impossible/out-of-range platform values only
    if redrob.get("github_activity_score", 0) > 100:
        return True
    if redrob.get("recruiter_response_rate", 0) > 1.0:
        return True
    if redrob.get("interview_completion_rate", 0) > 1.0:
        return True

    # Removed endorsement_mismatch check — too many false positives (210 hits)
    # Endorsements can legitimately exceed 3x connections if endorsed
    # before connections grew, so this isn't a reliable honeypot signal

    return False

def score_candidate_for_jd(candidate: Dict) -> Dict:
    """Score a candidate against the Senior AI Engineer JD."""
    cid = candidate.get("candidate_id")
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})
    # Honeypot check — force to bottom if detected
    is_honeypot = detect_honeypot(candidate)

    # --- Component Scores ---

    # 1. AI Skills (40% weight)
    ai_count, matched_skills, avg_proficiency = count_ai_skills(skills)
    assessment_scores = redrob.get("skill_assessment_scores", {})
    ai_assessment_avg = (
        sum(assessment_scores.values()) / len(assessment_scores)
        if assessment_scores else 50
    )
    ai_skills_score = min(1.0, ai_count / 8.0)
    if avg_proficiency >= 3.0:
        ai_skills_score = min(1.0, ai_skills_score * 1.1)
    if ai_assessment_avg > 70:
        ai_skills_score = min(1.0, ai_skills_score * 1.1)
# 1b. Career semantic score — catches AI work not listed as skills
    career_semantic = get_career_semantic_score(career)
    ai_skills_score = min(1.0, ai_skills_score * 0.7 + career_semantic * 0.3)
    # 2. Title relevance (25% weight)
    current_title = profile.get("current_title", "")
    title_score = get_title_score(current_title)

    # 3. Experience fit (18% weight)
    years = profile.get("years_of_experience", 0)
    exp_score = get_experience_fit(years)

    # 4. Availability (12% weight)
    availability_score = get_availability_score(redrob)

    # 5. Company quality (3% weight)
    company_score, is_consulting_only = get_company_quality(career)

    # 6. Location (2% weight)
    location_score = get_location_score(profile)

    # 7. Notice period
    notice_score = get_notice_score(redrob.get("notice_period_days", 90))

    # 8. Platform engagement
    platform_score = get_platform_score(redrob)

    # --- Hard Gates (disqualifiers per JD) ---

    # Gate 1: Wrong title — hard cap at 0.35 max
    title_is_disqualified = title_score <= 0.1

    # Gate 2: Terrible availability — hard cap at 0.45 max
    availability_is_terrible = (
        redrob.get("recruiter_response_rate", 0) < 0.05 or
        availability_score < 0.15
    )

    # --- Final Weighted Score ---
    raw_score = (
        ai_skills_score    * 0.40 +
        title_score        * 0.25 +
        exp_score          * 0.18 +
        availability_score * 0.12 +
        company_score      * 0.03 +
        location_score     * 0.02
    )

    # Apply hard caps
    if title_is_disqualified and availability_is_terrible:
        final_score = min(0.25, raw_score)
    elif title_is_disqualified:
        final_score = min(0.35, raw_score)
    elif availability_is_terrible:
        final_score = min(0.45, raw_score)
    else:
        final_score = raw_score

    # Hard cap at 1.0
    final_score = min(1.0, round(final_score, 4))

    # Force honeypots to near-zero score
    if is_honeypot:
        final_score = 0.001

    # Reasoning string matching sample_submission.csv format
    if is_honeypot:
        reasoning = "Profile flagged as inconsistent (honeypot signal detected) — excluded from consideration."
    else:
        reasoning = (
            f"{current_title} with {years} yrs; "
            f"{ai_count} AI core skills; "
            f"response rate {redrob.get('recruiter_response_rate', 0):.2f}"
        )

    return {
        "candidate_id": cid,
        "score": final_score,
        "reasoning": reasoning,
        "_breakdown": {
            "ai_skills_score": round(ai_skills_score, 3),
            "ai_skill_count": ai_count,
            "avg_proficiency": round(avg_proficiency, 2),
            "title_score": round(title_score, 3),
            "current_title": current_title,
            "exp_score": round(exp_score, 3),
            "years": years,
            "availability_score": round(availability_score, 3),
            "response_rate": redrob.get("recruiter_response_rate", 0),
            "company_score": round(company_score, 3),
            "is_consulting_only": is_consulting_only,
            "location_score": round(location_score, 3),
            "location": profile.get("location", ""),
            "notice_score": round(notice_score, 3),
            "notice_days": redrob.get("notice_period_days", 0),
            "platform_score": round(platform_score, 3),
            "matched_ai_skills": [s.get("name") for s in matched_skills[:6]],
        }
    }