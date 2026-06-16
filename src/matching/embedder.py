from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple

BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"  # Lighter version, faster download
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lighter cross encoder

print("Loading embedding models...")
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
print("Models loaded.")


def create_candidate_text_representation(candidate_data: Dict) -> str:
    """Convert structured candidate JSON into rich text for embedding."""
    parts = []

    cs = candidate_data.get("career_signals", {})
    if cs:
        parts.append(f"Career Level: {cs.get('career_level', '')}")
        parts.append(f"Total Experience: {cs.get('total_experience_years', 0)} years")
        parts.append(f"Trajectory: {cs.get('career_trajectory', '')}")
        parts.append(f"Specialization: {cs.get('specialization', '')}")

    skills = candidate_data.get("skills", {})
    if skills:
        all_skills = (
            skills.get("technical", []) +
            skills.get("tools", []) +
            skills.get("domains", [])
        )
        parts.append(f"Technical Skills: {', '.join(all_skills)}")
        if skills.get("soft"):
            parts.append(f"Soft Skills: {', '.join(skills.get('soft', []))}")

    for exp in candidate_data.get("experience", [])[:5]:
        role = f"{exp.get('title', '')} at {exp.get('company', '')}"
        if exp.get("achievements"):
            role += ". " + ". ".join(exp["achievements"][:2])
        if exp.get("tech_stack"):
            role += f". Tech: {', '.join(exp['tech_stack'][:5])}"
        parts.append(role)

    for proj in candidate_data.get("projects", [])[:3]:
        parts.append(
            f"Project: {proj.get('name', '')} — {proj.get('description', '')} "
            f"({proj.get('complexity', '')} complexity)"
        )

    for edu in candidate_data.get("education", []):
        parts.append(
            f"Education: {edu.get('degree', '')} in {edu.get('field', '')} "
            f"from {edu.get('institution', '')}"
        )

    bs = candidate_data.get("behavioral_signals", {})
    signals = [k for k, v in bs.items() if v]
    if signals:
        parts.append(f"Behavioral Signals: {', '.join(signals)}")

    return "\n".join(parts)


def create_jd_text_representation(jd_data: Dict) -> str:
    """Convert structured JD JSON into rich text for embedding."""
    parts = [
        f"Role: {jd_data.get('title', '')}",
        f"Seniority: {jd_data.get('seniority', '')}",
        f"Domain: {jd_data.get('domain', '')}",
        f"Required Skills: {', '.join(jd_data.get('required_skills', []))}",
        f"Preferred Skills: {', '.join(jd_data.get('preferred_skills', []))}",
        f"Hidden Skills: {', '.join(jd_data.get('hidden_skills', []))}",
        f"Full Tech Stack: {', '.join(jd_data.get('tech_stack_full', []))}",
        f"Culture: {', '.join(jd_data.get('culture_indicators', []))}",
        f"Ideal Candidate: {jd_data.get('ideal_candidate_archetype', '')}",
        f"Experience Required: {jd_data.get('required_experience_years', 0)} years",
    ]
    return "\n".join(parts)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts into dense vectors."""
    embeddings = bi_encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


class CandidateVectorStore:
    def __init__(self, dim: int = 384):  # bge-small outputs 384 dims
        self.dim = dim
        self.index = None
        self.candidate_ids = []
        self.candidate_data = {}

    def add_candidates(self, candidates: List[Dict], embeddings: np.ndarray):
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        for c in candidates:
            self.candidate_ids.append(c["candidate_id"])
            self.candidate_data[c["candidate_id"]] = c

    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        if self.index is None:
            return []
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(top_k, len(self.candidate_ids))
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.candidate_ids[idx], float(score)))
        return results

    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump({
                "candidate_ids": self.candidate_ids,
                "candidate_data": self.candidate_data,
                "dim": self.dim
            }, f)
        print(f"Vector store saved to {path}")

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        self.candidate_ids = meta["candidate_ids"]
        self.candidate_data = meta["candidate_data"]
        self.dim = meta.get("dim", 384)
        print(f"Loaded {len(self.candidate_ids)} candidates from vector store")