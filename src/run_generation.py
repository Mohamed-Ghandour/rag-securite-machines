"""Génère et note les réponses pour S0..S3 sur un jeu de questions. Reprise automatique (cache LLM).

Usage : python -m src.run_generation --questions data/testset/questions.jsonl --n 60 --systems S0 S1_bm25 S2_dense S3_hybride_rerank
Sortie : results/generations.csv (une ligne par question et par système)
Les classements doivent exister : python -m src.run_retrieval --questions <même fichier>
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src import rag
from src import retrieval_eval as R

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data/testset/questions.jsonl"))
    ap.add_argument("--config", default="structural_512")
    ap.add_argument("--systems", nargs="+", default=["S0", "S1_bm25", "S2_dense", "S3_hybride_rerank"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=0, help="sous-échantillon stratifié par type (0 = tout)")
    ap.add_argument("--prompt", choices=["cite", "simple"], default="cite")
    ap.add_argument("--out", default=str(ROOT / "results/generations.csv"))
    a = ap.parse_args()

    qs = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines()]
    if a.n:
        rng, by = random.Random(42), {}
        for q in qs:
            by.setdefault(q["type"], []).append(q)
        qs = [q for v in by.values() for q in rng.sample(v, max(1, round(a.n * len(v) / len(qs))))]
    ch = R.load_chunks(a.config)
    rank = json.loads((ROOT / f"data/processed/rankings_{a.config}.json").read_text(encoding="utf-8"))
    prompt = rag.PROMPT_RAG_CITE if a.prompt == "cite" else rag.PROMPT_RAG_SIMPLE

    rows = []
    out = Path(a.out)
    try:
        run(a, qs, ch, rank, prompt, rows, out)
    except RuntimeError as e:                            # quota journalier atteint : on garde ce qui est fait
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"ARRÊT (quota) après {len(rows)} lignes sauvegardées : {str(e)[:160]}")
        raise SystemExit(2)


def run(a, qs, ch, rank, prompt, rows, out) -> None:
    for s in a.systems:
        for n, q in enumerate(qs, 1):
            chunks = None if s == "S0" else [ch[i] for i in rank[s][q["id"]][:a.k]]
            g = rag.answer(q["question"], chunks, prompt)
            ref = q["reponse"]
            j = rag.judge(q["question"], ref or "(aucune réponse dans le corpus : l'abstention est la bonne réponse)",
                          g["text"], chunks)
            abst = rag.is_abstention(g["text"])
            rows.append({"systeme": s, "id": q["id"], "type": q["type"], "question": q["question"],
                         "reference": ref, "reponse": g["text"], "abstention": abst,
                         "f1": rag.f1(g["text"], ref) if ref else None,
                         "correct_juge": (1.0 if abst else 0.0) if ref is None else j["correct"],
                         "fidele_juge": j["fidele"], "raison_juge": j["raison"],
                         "citation_ok": rag.citation_ok(g["text"], len(chunks)) if chunks else None,
                         "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"],
                         "latency_s": g["latency_s"], "judge_tokens": j["judge_tokens"]})
            if n % 10 == 0:
                print(f"{s}: {n}/{len(qs)}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("terminé ->", out)


if __name__ == "__main__":
    main()
