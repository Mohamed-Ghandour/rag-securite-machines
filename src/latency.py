"""Latence de la chaîne de recherche (hors appel au LLM), mesurée étape par étape.

Pour chaque question on chronomètre : BM25, encodage de la requête + FAISS (dense), fusion RRF et reranker
(30 candidats). La latence de bout en bout d'un système = recherche + génération (la génération est mesurée
lors de l'appel à l'API et enregistrée dans results/generations_*.csv).

Mesures faites sur le GPU local (RTX 3050 6 Go), après un échauffement : ordres de grandeur, pas des garanties.
Usage : python -m src.latency
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import retrieval_eval as R
from src.bm25 import BM25
from src.dense import DenseIndex, encode, free_gpu
from src.hybrid import rerank, rrf

ROOT = Path(__file__).resolve().parents[1]
CAND = 30


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    gen = pd.read_csv(ROOT / "results/generations_35.csv")
    ids = set(gen.id)
    qs = [q for q in map(json.loads, (ROOT / "data/testset/brouillon.jsonl").read_text(encoding="utf-8").splitlines())
          if q["id"] in ids]
    ch = R.load_chunks("structural_512")
    t = time.perf_counter()
    bm = BM25([c["context"] + "\n" + c["text"] for c in ch])
    t_bm25_index = time.perf_counter() - t
    dn = DenseIndex(ch, "structural_512")

    encode(["échauffement"])                                   # charge le modèle d'embeddings sur le GPU
    rerank("échauffement", ch, list(range(CAND)), k=5)          # charge le reranker

    rows = []
    for q in qs:
        text = q["question"]
        t0 = time.perf_counter(); b = [i for i, _ in bm.search(text, CAND)]; t1 = time.perf_counter()
        qe = encode([text]); d = dn.search_emb(qe, CAND)[0][0].tolist(); _sync(); t2 = time.perf_counter()
        f = rrf([b, d], top=CAND); t3 = time.perf_counter()
        rerank(text, ch, f, k=5); _sync(); t4 = time.perf_counter()
        rows.append({"bm25_ms": (t1 - t0) * 1e3, "dense_ms": (t2 - t1) * 1e3, "rrf_ms": (t3 - t2) * 1e3,
                     "rerank_ms": (t4 - t3) * 1e3})
    d = pd.DataFrame(rows)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
    free_gpu()

    ret = {"S0": 0.0, "S1_bm25": d.bm25_ms.median(), "S2_dense": d.dense_ms.median(),
           "S3_hybride_rerank": (d.bm25_ms + d.dense_ms + d.rrf_ms + d.rerank_ms).median()}
    llm_s = gen.groupby("systeme").latency_s.mean()
    out = pd.DataFrame({
        "latence recherche (ms, médiane)": {k: round(v, 1) for k, v in ret.items()},
        "latence génération API (s, moyenne)": llm_s.round(2),
    })
    out["latence de bout en bout (s)"] = (out["latence recherche (ms, médiane)"] / 1000 + out["latence génération API (s, moyenne)"]).round(2)
    out.index.name = "systeme"
    out.to_csv(ROOT / "results/latence_bout_en_bout.csv")
    stages = d.median().round(1).rename("médiane (ms)").to_frame()
    stages.to_csv(ROOT / "results/latence_etapes.csv")
    print(out.to_string()); print(); print(stages.to_string())
    print(f"\nindexation BM25 (2 220 chunks) : {t_bm25_index:.2f} s | VRAM maximale : {peak:.1f} Go | n = {len(qs)} questions")


if __name__ == "__main__":
    main()
