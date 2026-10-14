"""Recherche hybride + reranking (système S3).

1) Fusion RRF (Reciprocal Rank Fusion) : BM25 et dense produisent chacun un classement ;
   un chunk reçoit  score = somme sur les classements de 1 / (K + rang)  (K = 60, valeur usuelle).
   On ne compare jamais les scores bruts (échelles différentes), seulement les rangs : un chunk
   bien classé par les DEUX méthodes remonte.
2) Reranker cross-encoder : contrairement aux embeddings (question et chunk encodés séparément),
   le cross-encoder lit le couple (question, chunk) ensemble et produit un score de pertinence.
   Plus précis mais plus lent : on ne l'applique qu'aux N meilleurs candidats de la fusion.
"""
import gc

import torch
from sentence_transformers import CrossEncoder

RRF_K = 60
RERANKER = "BAAI/bge-reranker-v2-m3"
_ce = None


def rrf(rankings: list[list[int]], k: int = RRF_K, top: int = 30) -> list[int]:
    scores = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top]


def reranker() -> CrossEncoder:
    global _ce
    if _ce is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _ce = CrossEncoder(RERANKER, device=dev, max_length=1024,
                           model_kwargs={"torch_dtype": torch.float16} if dev == "cuda" else {})
    return _ce


def free_gpu() -> None:
    global _ce
    _ce = None
    gc.collect()
    torch.cuda.empty_cache()


def rerank(query: str, chunks: list[dict], candidates: list[int], k: int = 10, batch: int = 8) -> list[int]:
    pairs = [(query, chunks[i]["context"] + "\n" + chunks[i]["text"]) for i in candidates]
    sc = reranker().predict(pairs, batch_size=batch, show_progress_bar=False)
    order = sorted(range(len(candidates)), key=lambda j: -sc[j])
    return [candidates[j] for j in order[:k]]
