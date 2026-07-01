from src.config import AI_SKILL_ALIASES


def normalize_skill(skill: str):

    skill = skill.lower().strip()

    if skill in AI_SKILL_ALIASES:
        return AI_SKILL_ALIASES[skill]

    return skill
def extract_candidate_skills(skills):

    result = []

    for s in skills:

        name = s.get("name", "")

        if not name:
            continue

        result.append(normalize_skill(name))

    return sorted(set(result))
def match_skills(candidate_skills, jd_skills):

    matched = []

    missing = []

    for jd_skill in jd_skills:

        found = False

        for cand_skill in candidate_skills:

            if jd_skill in cand_skill or cand_skill in jd_skill:
                matched.append(jd_skill)
                found = True
                break

        if not found:
            missing.append(jd_skill)

    coverage = 0

    if jd_skills:
        coverage = len(matched) / len(jd_skills)

    return {

        "matched": matched,

        "missing": missing,

        "coverage": coverage

    }
