from pathlib import Path
import re
import docx

from src.config import AI_CORE_SKILLS


def load_job_description(path: str) -> str:
    """
    Read a .docx job description and return plain text.
    """

    doc = docx.Document(path)

    paragraphs = []

    for p in doc.paragraphs:
        if p.text.strip():
            paragraphs.append(p.text.strip())

    return "\n".join(paragraphs)
def extract_experience(text: str):
    """
    Finds experience like:
    5-9 years
    3+ years
    7 years
    """

    text = text.lower()

    patterns = [
        r'(\d+)\s*-\s*(\d+)\s*years',
        r'(\d+)\+\s*years',
        r'(\d+)\s*years'
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            if len(match.groups()) == 2:
                return (
                    int(match.group(1)),
                    int(match.group(2))
                )

            return (
                int(match.group(1)),
                int(match.group(1))
            )

    return None
def extract_skills(text: str):

    text = text.lower()

    found = []

    for skill in AI_CORE_SKILLS:

        if skill in text:
            found.append(skill)

    return sorted(found)
def extract_location(text: str):

    text = text.lower()

    cities = [
        "bangalore",
        "bengaluru",
        "hyderabad",
        "pune",
        "noida",
        "gurugram",
        "delhi",
        "mumbai",
        "india"
    ]

    result = []

    for city in cities:

        if city in text:
            result.append(city)

    return result
def parse_job_description(path: str):

    text = load_job_description(path)

    profile = {

        "raw_text": text,

        "skills": extract_skills(text),

        "experience": extract_experience(text),

        "locations": extract_location(text)

    }

    return profile