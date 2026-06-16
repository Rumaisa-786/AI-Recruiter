import json
import os
from groq import Groq
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Best free model on Groq — fast and accurate
MODEL = "llama-3.3-70b-versatile"

CANDIDATE_EXTRACTION_PROMPT = """You are an expert technical recruiter and talent analyst.

Analyze this resume and extract structured information.

Return ONLY valid JSON, no explanation, no markdown, no backticks. Just raw JSON.

Schema:
{{
  "skills": {{
    "technical": [],
    "soft": [],
    "tools": [],
    "domains": []
  }},
  "experience": [
    {{
      "company": "",
      "title": "",
      "duration_months": 0,
      "start_year": 0,
      "end_year": 0,
      "is_current": false,
      "level": "junior|mid|senior|staff|principal|exec",
      "responsibilities": [],
      "achievements": [],
      "tech_stack": [],
      "team_size_managed": 0,
      "scope": "individual|team|department|company|external"
    }}
  ],
  "education": [
    {{
      "degree": "",
      "field": "",
      "institution": "",
      "tier": "tier1|tier2|tier3",
      "year": 0,
      "gpa": null
    }}
  ],
  "projects": [
    {{
      "name": "",
      "description": "",
      "impact": "",
      "tech_stack": [],
      "complexity": "low|medium|high|research-grade",
      "is_open_source": false
    }}
  ],
  "career_signals": {{
    "total_experience_years": 0.0,
    "career_level": "junior|mid|senior|staff|principal|exec",
    "career_trajectory": "ascending|flat|pivoting|descending",
    "promotion_velocity": "fast|normal|slow",
    "company_prestige_avg": "tier1|tier2|tier3",
    "job_switching_pattern": "stable|normal|frequent",
    "leadership_signals": [],
    "learning_velocity": "high|medium|low",
    "specialization": "generalist|specialist|hybrid"
  }},
  "achievements": [
    {{
      "description": "",
      "impact_type": "revenue|cost|performance|scale|recognition",
      "magnitude": "small|medium|large|exceptional",
      "is_quantified": true
    }}
  ],
  "behavioral_signals": {{
    "open_source_contributions": false,
    "publications": false,
    "patents": false,
    "speaking_engagements": false,
    "mentorship": false,
    "entrepreneurial": false
  }}
}}

Resume:
{resume_text}"""


JD_EXTRACTION_PROMPT = """You are an expert recruiter analyzing a job description.

Extract ALL requirements including HIDDEN ones not explicitly stated.

Return ONLY valid JSON, no explanation, no markdown, no backticks. Just raw JSON.

Schema:
{{
  "title": "",
  "seniority": "junior|mid|senior|staff|principal|exec",
  "domain": "",
  "required_skills": [],
  "preferred_skills": [],
  "hidden_skills": [],
  "required_experience_years": 0,
  "required_education": [],
  "culture_indicators": [],
  "growth_expectations": [],
  "red_flags_for_candidates": [],
  "ideal_candidate_archetype": "",
  "success_metrics": [],
  "team_context": "",
  "tech_stack_full": []
}}

Job Description:
{jd_text}"""


LLM_RANKING_PROMPT = """You are a world-class technical recruiter.

Given this job and top candidates, re-rank them by true fit.

Return ONLY valid JSON, no explanation, no markdown, no backticks. Just raw JSON.

Schema:
{{
  "ranked_ids": [],
  "candidate_assessments": {{
    "CANDIDATE_ID": {{
      "strengths": [],
      "gaps": [],
      "recruiter_pitch": "",
      "false_positive_risk": false,
      "confidence": "high|medium|low"
    }}
  }}
}}

Job Summary:
{jd_summary}

Candidates:
{candidates_summary}"""


def _clean_json(text: str) -> str:
    """Strip any accidental markdown formatting from LLM output."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                json.loads(part)
                return part
            except:
                continue
    return text


def extract_candidate_profile(resume_text: str) -> Dict[str, Any]:
    """Send resume to Groq LLM and get back structured candidate data."""
    prompt = CANDIDATE_EXTRACTION_PROMPT.format(
        resume_text=resume_text[:6000]  # Groq context limit safety
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON extractor. Always return valid JSON only. No extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,      # Low temp = more consistent JSON output
        max_tokens=4096,
    )

    raw = response.choices[0].message.content
    cleaned = _clean_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {raw[:500]}")
        # Return safe empty structure so pipeline doesn't crash
        return _empty_candidate_profile()


def extract_jd_profile(jd_text: str) -> Dict[str, Any]:
    """Send job description to Groq LLM and get back structured JD data."""
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text[:4000])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON extractor. Always return valid JSON only. No extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    cleaned = _clean_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return _empty_jd_profile()


def rank_with_llm(jd_summary: str, candidates_summary: str) -> Dict[str, Any]:
    """Ask Groq LLM to holistically re-rank the top candidates."""
    prompt = LLM_RANKING_PROMPT.format(
        jd_summary=jd_summary,
        candidates_summary=candidates_summary
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON extractor. Always return valid JSON only. No extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content
    cleaned = _clean_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"ranked_ids": [], "candidate_assessments": {}}


def _empty_candidate_profile() -> Dict:
    """Safe fallback if LLM returns broken JSON."""
    return {
        "skills": {"technical": [], "soft": [], "tools": [], "domains": []},
        "experience": [],
        "education": [],
        "projects": [],
        "career_signals": {
            "total_experience_years": 0.0,
            "career_level": "mid",
            "career_trajectory": "flat",
            "promotion_velocity": "normal",
            "company_prestige_avg": "tier3",
            "job_switching_pattern": "normal",
            "leadership_signals": [],
            "learning_velocity": "medium",
            "specialization": "generalist"
        },
        "achievements": [],
        "behavioral_signals": {
            "open_source_contributions": False,
            "publications": False,
            "patents": False,
            "speaking_engagements": False,
            "mentorship": False,
            "entrepreneurial": False
        }
    }


def _empty_jd_profile() -> Dict:
    """Safe fallback if LLM returns broken JSON."""
    return {
        "title": "Unknown Role",
        "seniority": "mid",
        "domain": "software",
        "required_skills": [],
        "preferred_skills": [],
        "hidden_skills": [],
        "required_experience_years": 2,
        "required_education": [],
        "culture_indicators": [],
        "growth_expectations": [],
        "red_flags_for_candidates": [],
        "ideal_candidate_archetype": "",
        "success_metrics": [],
        "team_context": "",
        "tech_stack_full": []
    }