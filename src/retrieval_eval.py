"""Évaluation du retrieval : Recall@k et MRR, indépendamment de la façon dont le corpus est découpé.

Un chunk est « pertinent » pour une preuve si, dans le même document, il contient au moins
MIN_COVER de la preuve (part des triplets de mots de la preuve retrouvés dans le chunk).
Cette définition fonctionne pour toutes les tailles de chunks, y compris quand une preuve est
coupée entre deux chunks fixes (chacun la couvre alors en partie).

- Recall@k : part des preuves d'une question couvertes par les k premiers chunks
  (pour une question multi-passages à 2 preuves : 0, 0.5 ou 1), moyennée sur les questions.
- MRR : 1 / rang du premier chunk pertinent (0 s'il n'y en a aucun dans le top-K).
Les questions sans réponse n'ont pas de preuve : elles sont exclues du retrieval.
"""
import json
import re
from pathlib import Path

import numpy as np

from src.testset import norm

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "processed" / "chunks"
MIN_COVER = 0.6
KS = (1, 3, 5, 10)


def load_chunks(name: str) -> list[dict]:
    return [json.loads(l) for l in (CHUNK_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def _trigrams(text: str) -> set:
    w = re.findall(r"\w+", norm(text))
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)} or {tuple(w)}


class Gold:
    """Pré-calcule, pour une configuration de chunks, quels chunks couvrent chaque preuve."""

    def __init__(self, chunks: list[dict], questions: list[dict]):
        by_doc = {}
        for i, c in enumerate(chunks):
            by_doc.setdefault(c["doc"], []).append((i, _trigrams(c["text"])))
        self.rel = {}                                   # id question -> liste (une par preuve) d'ensembles d'indices
        for q in questions:
            per_ev = []
            for e in q["preuves"]:
                tg = _trigrams(e["quote"])
                per_ev.append({i for i, ctg in by_doc.get(e["doc"], [])
                               if len(tg & ctg) / len(tg) >= MIN_COVER})
            self.rel[q["id"]] = per_ev

    def unreachable(self) -> list[str]:
        return [q for q, evs in self.rel.items() if any(not s for s in evs)]


def evaluate(rank_fn, chunks: list[dict], questions: list[dict], ks=KS, kmax=10) -> dict:
    """rank_fn(question: str, k: int) -> liste d'indices de chunks (meilleur d'abord)."""
    qs = [q for q in questions if q["type"] != "sans_reponse"]
    gold = Gold(chunks, qs)
    per_q = []
    for q in qs:
        top = rank_fn(q["question"], kmax)
        evs = gold.rel[q["id"]]
        row = {"id": q["id"], "type": q["type"]}
        for k in ks:
            topk = set(top[:k])
            row[f"recall@{k}"] = np.mean([bool(s & topk) for s in evs])
        rr = 0.0
        for rank, i in enumerate(top, 1):
            if any(i in s for s in evs):
                rr = 1 / rank
                break
        row["mrr"] = rr
        per_q.append(row)
    return {"per_question": per_q, "n": len(per_q), "unreachable": gold.unreachable()}


def summarize(res: dict) -> dict:
    rows = res["per_question"]
    keys = [k for k in rows[0] if k not in ("id", "type")]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
