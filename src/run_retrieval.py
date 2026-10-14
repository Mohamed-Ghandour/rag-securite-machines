"""Compare S1 (BM25), S2 (dense), RRF et S3 (RRF + reranker) en retrieval ; sauvegarde les classements.

Sortie : results/retrieval_<config>.csv (une ligne par système) et
data/processed/rankings_<config>.json (top-10 par question et par système, réutilisé pour la génération).
Usage : python -m src.run_retrieval [--questions chemin.jsonl] [--config structural_512]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import retrieval_eval as R
from src.bm25 import BM25
from src.dense import DenseIndex, encode, free_gpu as free_dense
from src.hybrid import rrf, rerank, free_gpu as free_rerank

ROOT = Path(__file__).resolve().parents[1]
CAND = 30      # candidats envoyés au reranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data/testset/brouillon.jsonl"))
    ap.add_argument("--config", default="structural_512")
    a = ap.parse_args()
    qs = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines()]
    ch = R.load_chunks(a.config)
    bm = BM25([c["context"] + "\n" + c["text"] for c in ch])
    dn = DenseIndex(ch, a.config)
    Q = encode([q["question"] for q in qs])
    free_dense()

    rank = {s: {} for s in ("S1_bm25", "S2_dense", "RRF", "S3_hybride_rerank")}
    for i, q in enumerate(qs):
        b = [j for j, _ in bm.search(q["question"], CAND)]
        d = dn.search_emb(Q[i:i + 1], CAND)[0][0].tolist()
        f = rrf([b, d], top=CAND)
        rank["S1_bm25"][q["id"]] = b[:10]
        rank["S2_dense"][q["id"]] = d[:10]
        rank["RRF"][q["id"]] = f[:10]
        rank["S3_hybride_rerank"][q["id"]] = rerank(q["question"], ch, f, k=10)
    free_rerank()

    (ROOT / "data/processed").mkdir(exist_ok=True)
    (ROOT / f"data/processed/rankings_{a.config}.json").write_text(json.dumps(rank), encoding="utf-8")
    rows, per_q = [], []
    for name, rk in rank.items():
        res = R.evaluate(lambda q, k, rk=rk, qmap={q["question"]: q["id"] for q in qs}: rk[qmap[q]][:k], ch, qs)
        rows.append({"systeme": name, **R.summarize(res)})
        per_q += [{"systeme": name, **r} for r in res["per_question"]]
    (ROOT / "results").mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results/retrieval_{a.config}.csv", index=False)
    pd.DataFrame(per_q).to_csv(ROOT / f"results/retrieval_par_question_{a.config}.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
