from rank_bm25 import BM25Okapi
import re
from typing import List, Tuple, Dict


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.candidate_ids = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                     'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def build_index(self, candidates: List[Dict], texts: List[str]):
        """Build BM25 index from candidate texts."""
        self.candidate_ids = [c["candidate_id"] for c in candidates]
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built for {len(self.candidate_ids)} candidates")

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Return top-k (candidate_id, score) pairs."""
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        return [(self.candidate_ids[i], float(scores[i])) for i in top_indices]