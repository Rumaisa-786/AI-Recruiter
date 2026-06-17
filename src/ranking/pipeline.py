# NOTE: Not used at submission time. See README "Two design iterations" —
# this 5-stage hybrid pipeline (dense + BM25 + cross-encoder + LLM pass) was
# our first design but can't fit the 100k-candidate / 5-min / CPU-only /
# no-network budget enforced at submission. The live ranker is
# src/data_loader.py + src/runner.py.
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
import json

from src.matching.embedder import (
    CandidateVectorStore, embed_texts,
    create_candidate_text_representation,
    create_jd_text_representation
)
from src.matching.sparse_retriever import BM25Retriever
from src.scoring.scorer import score_candidate, CandidateScore
from src.understanding.llm_extractor import rank_with_llm

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def stage1_hybrid_retrieval(
    jd_embedding: np.ndarray,
    jd_text: str,
    vector_store: CandidateVectorStore,
    bm25: BM25Retriever,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """Combine dense FAISS + sparse BM25 results using Reciprocal Rank Fusion."""
    dense_results = vector_store.search(jd_embedding, top_k=top_k)
    sparse_results = bm25.search(jd_text, top_k=top_k)

    rrf_scores = {}
    k = 60

    for rank, (cid, _) in enumerate(dense_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)

    for rank, (cid, _) in enumerate(sparse_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def stage2_multidim_scoring(
    retrieved: List[Tuple[str, float]],
    candidate_data_map: Dict,
    jd_data: Dict
) -> List[Tuple[str, CandidateScore]]:
    """Score each retrieved candidate across 7 dimensions."""
    scored = []
    for cid, retrieval_score in retrieved:
        cdata = candidate_data_map.get(cid, {})
        score = score_candidate(cdata, jd_data, cid)
        score.semantic_similarity = retrieval_score
        scored.append((cid, score))

    return sorted(scored, key=lambda x: x[1].weighted_total, reverse=True)


def stage3_cross_encoder_rerank(
    candidates: List[Tuple[str, CandidateScore]],
    jd_text: str,
    candidate_texts: Dict[str, str],
    top_k: int = 30
) -> List[Tuple[str, CandidateScore]]:
    """Use cross-encoder to re-rank top candidates more accurately."""
    top = candidates[:top_k]
    rest = candidates[top_k:]

    pairs = [
        [jd_text[:512], candidate_texts.get(cid, "")[:512]]
        for cid, _ in top
    ]

    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)

    for i, (cid, score) in enumerate(top):
        score.cross_encoder_score = float(ce_scores[i])

    top.sort(
        key=lambda x: 0.5 * x[1].cross_encoder_score + 0.5 * (x[1].weighted_total / 100),
        reverse=True
    )

    return top + rest


def stage4_llm_rerank(
    candidates: List[Tuple[str, CandidateScore]],
    jd_data: Dict,
    candidate_data_map: Dict,
    top_k: int = 15
) -> List[Tuple[str, CandidateScore]]:
    """Ask Groq LLM to holistically re-rank the top candidates."""
    top = candidates[:top_k]
    rest = candidates[top_k:]

    jd_summary = f"""
Title: {jd_data.get('title')}
Seniority: {jd_data.get('seniority')}
Required Skills: {', '.join(jd_data.get('required_skills', [])[:10])}
Ideal Candidate: {jd_data.get('ideal_candidate_archetype', '')}
Experience Required: {jd_data.get('required_experience_years', 0)} years
"""

    candidates_summary = ""
    for i, (cid, score) in enumerate(top):
        cdata = candidate_data_map.get(cid, {})
        cs = cdata.get("career_signals", {})
        skills = cdata.get("skills", {}).get("technical", [])[:6]
        candidates_summary += f"""
Candidate {i+1} (ID: {cid}):
- Level: {cs.get('career_level', 'unknown')}
- Experience: {cs.get('total_experience_years', 0)} years
- Trajectory: {cs.get('career_trajectory', 'unknown')}
- Skills: {', '.join(skills)}
- Missing: {', '.join(score.missing_skills[:3])}
- Pre-Score: {score.weighted_total:.1f}/100
---"""

    llm_result = rank_with_llm(jd_summary, candidates_summary)

    cid_to_score = {cid: score for cid, score in top}
    ranked_ids = llm_result.get("ranked_ids", [cid for cid, _ in top])
    assessments = llm_result.get("candidate_assessments", {})

    reranked = []
    seen = set()
    for rank, cid in enumerate(ranked_ids):
        if cid in cid_to_score and cid not in seen:
            score = cid_to_score[cid]
            score.llm_rank = rank + 1
            assessment = assessments.get(cid, {})
            score.strengths = assessment.get("strengths", [])
            score.gaps = assessment.get("gaps", [])
            score.ranking_reasons = [assessment.get("recruiter_pitch", "")]
            reranked.append((cid, score))
            seen.add(cid)

    for cid, score in top:
        if cid not in seen:
            reranked.append((cid, score))

    return reranked + rest


def stage5_ensemble_ranking(
    candidates: List[Tuple[str, CandidateScore]]
) -> List[Tuple[str, CandidateScore]]:
    """Final ranking combining all signals with ensemble scoring."""
    if not candidates:
        return []

    n = len(candidates)
    weighted_totals = [s.weighted_total for _, s in candidates]
    cross_scores = [s.cross_encoder_score for _, s in candidates]
    llm_ranks = [s.llm_rank if s.llm_rank > 0 else n for _, s in candidates]

    def normalize(arr):
        mn, mx = min(arr), max(arr)
        if mx == mn:
            return [0.5] * len(arr)
        return [(x - mn) / (mx - mn) for x in arr]

    norm_w = normalize(weighted_totals)
    norm_c = normalize(cross_scores)
    norm_l = normalize([n - r for r in llm_ranks])

    ensemble = []
    for i, (cid, score) in enumerate(candidates):
        final = 0.35 * norm_w[i] + 0.30 * norm_c[i] + 0.35 * norm_l[i]
        score.recruiter_confidence = round(final * 100, 1)
        ensemble.append((cid, score, final))

    ensemble.sort(key=lambda x: x[2], reverse=True)
    return [(cid, score) for cid, score, _ in ensemble]


def run_full_pipeline(
    jd_data: Dict,
    candidate_data_map: Dict,
    vector_store: CandidateVectorStore,
    bm25: BM25Retriever,
    candidate_texts: Dict[str, str],
    top_n: int = 10
) -> List[Tuple[str, CandidateScore]]:
    """Run all 5 stages and return final ranked shortlist."""

    jd_text = create_jd_text_representation(jd_data)
    jd_embedding = embed_texts([jd_text])[0]

    print("\n[Stage 1] Hybrid Retrieval (FAISS + BM25)...")
    retrieved = stage1_hybrid_retrieval(jd_embedding, jd_text, vector_store, bm25)
    print(f"  → Retrieved {len(retrieved)} candidates")

    print("[Stage 2] Multi-dimensional Scoring...")
    scored = stage2_multidim_scoring(retrieved, candidate_data_map, jd_data)
    print(f"  → Scored {len(scored)} candidates")

    print("[Stage 3] Cross-Encoder Re-ranking...")
    reranked = stage3_cross_encoder_rerank(scored, jd_text, candidate_texts, top_k=30)
    print(f"  → Re-ranked top 30")

    print("[Stage 4] LLM Holistic Evaluation...")
    llm_ranked = stage4_llm_rerank(reranked, jd_data, candidate_data_map, top_k=15)
    print(f"  → LLM evaluated top 15")

    print("[Stage 5] Ensemble Final Ranking...")
    final = stage5_ensemble_ranking(llm_ranked)
    print(f"  → Final top {top_n} ready\n")

    return final[:top_n]