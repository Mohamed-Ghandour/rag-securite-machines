"""Analyse des résultats : métriques par système, bootstrap, erreurs, figures, annotation du juge (kappa).

Usage : python -m src.analyze report     [--gen results/generations_provisoire.csv]
        python -m src.analyze annotate   (crée results/annotations_a_remplir.csv, 30 réponses à noter à la main)
        python -m src.analyze kappa      (accord humain / juge, une fois le fichier rempli)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
SYSTEMS = ["S0", "S1_bm25", "S2_dense", "S3_hybride_rerank"]
LABEL = {"S0": "S0 sans retrieval", "S1_bm25": "S1 BM25", "S2_dense": "S2 dense", "S3_hybride_rerank": "S3 hybride+rerank"}
SEED, N_BOOT = 42, 5000
# Tarifs indicatifs en $ par million de tokens (à vérifier sur la page de tarification Groq avant de citer).
PRICE = {"in": 0.075, "out": 0.30}


def bootstrap_ci(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a, b, n=N_BOOT, seed=SEED):
    """IC bootstrap de la différence moyenne a - b sur les MÊMES questions (test apparié)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n, seed)


def fmt(m, lo, hi):
    return f"{m:.2f} [{lo:.2f} ; {hi:.2f}]"


def report(gen_path: Path) -> None:
    global SYSTEMS
    g = pd.read_csv(gen_path)
    g["correct_juge"] = pd.to_numeric(g["correct_juge"], errors="coerce")
    counts = g.groupby("systeme").size()
    SYSTEMS = [s for s in SYSTEMS if s in counts.index]
    for s in SYSTEMS:
        print(f"{LABEL[s]} : {counts[s]} questions" + ("  <-- INCOMPLET" if counts[s] < counts.max() else ""))
    excluded = [s for s in SYSTEMS if counts[s] < 0.5 * counts.max()]
    if excluded:
        print(f"Exclus de l'analyse (couverture < 50 %) : {[LABEL[s] for s in excluded]}")
        SYSTEMS = [s for s in SYSTEMS if s not in excluded]
    # comparaison équitable : uniquement les questions traitées par TOUS les systèmes présents
    common = set.intersection(*[set(g[g.systeme == s].id) for s in SYSTEMS])
    g = g[g.id.isin(common)]
    print(f"-> analyse sur les {len(common)} questions communes à tous les systèmes\n")
    ans, noans = g[g.type != "sans_reponse"], g[g.type == "sans_reponse"]
    ret = pd.read_csv(RES / "retrieval_par_question_structural_512.csv")
    rows = []
    for s in SYSTEMS:
        a, n = ans[ans.systeme == s], noans[noans.systeme == s]
        row = {"systeme": LABEL[s], "n_répondables": len(a), "n_sans_réponse": len(n)}
        row["correct (juge)"] = fmt(*bootstrap_ci(a.correct_juge.fillna(0)))
        row["F1"] = fmt(*bootstrap_ci(a.f1.fillna(0)))
        if s != "S0":
            row["fidélité (juge)"] = fmt(*bootstrap_ci(g[g.systeme == s].fidele_juge.dropna()))
        row["abstention correcte"] = fmt(*bootstrap_ci(n.abstention.astype(float))) if len(n) else "-"
        row["refus injustifié"] = f"{a.abstention.mean():.2f}"
        t = g[g.systeme == s]
        row["tokens/requête"] = int((t.prompt_tokens + t.completion_tokens).mean())
        row["latence (s)"] = round(t.latency_s.mean(), 2)
        row["coût / 1000 req ($)"] = round((t.prompt_tokens.mean() * PRICE["in"] + t.completion_tokens.mean() * PRICE["out"]) / 1000, 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RES / "metriques_par_systeme.csv", index=False)
    print(table.to_string(index=False))

    print("\nDifférences appariées (IC95 % bootstrap, correct juge, questions répondables) :")
    piv = ans.pivot(index="id", columns="systeme", values="correct_juge").fillna(0)
    diffs = []
    for a_, b_ in [("S3_hybride_rerank", "S1_bm25"), ("S3_hybride_rerank", "S2_dense"), ("S1_bm25", "S0"), ("S3_hybride_rerank", "S0")]:
        if a_ in piv and b_ in piv:
            m, lo, hi = paired_diff(piv[a_], piv[b_])
            sig = "significatif" if lo > 0 or hi < 0 else "non significatif"
            print(f"  {LABEL[a_]} - {LABEL[b_]} : {fmt(m, lo, hi)}  -> {sig}")
            diffs.append({"a": a_, "b": b_, "diff": m, "ic_bas": lo, "ic_haut": hi, "verdict": sig})
    pd.DataFrame(diffs).to_csv(RES / "differences_bootstrap.csv", index=False)

    # par type de question
    by_type = ans.groupby(["systeme", "type"]).correct_juge.mean().unstack().reindex(SYSTEMS).round(2)
    by_type.to_csv(RES / "correct_par_type.csv")
    print("\nCorrect (juge) par type de question :\n", by_type.to_string())

    # taxonomie d'erreurs (systèmes avec retrieval)
    errs = []
    for s in SYSTEMS[1:]:
        r5 = ret[ret.systeme == s].set_index("id")["recall@5"]
        for _, q in g[g.systeme == s].iterrows():
            if q.type == "sans_reponse":
                cat = "ok" if q.abstention else "hallucination (aurait dû s'abstenir)"
            elif q.correct_juge == 1:
                cat = "ok"
            elif q.abstention:
                cat = "refus injustifié" if r5.get(q.id, 0) > 0 else "passage non retrouvé (abstention)"
            elif r5.get(q.id, 0) == 0:
                cat = "passage non retrouvé"
            elif q.fidele_juge == 0:
                cat = "hallucination (info absente du contexte)"
            else:
                cat = "bon contexte, réponse incomplète/fausse"
            errs.append({"systeme": s, "categorie": cat})
    e = pd.DataFrame(errs).value_counts(["systeme", "categorie"]).unstack(0).fillna(0).astype(int)
    e.to_csv(RES / "taxonomie_erreurs.csv")
    print("\nTaxonomie d'erreurs (nombre de réponses) :\n", e.to_string())

    # figures
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cs = [bootstrap_ci(ans[ans.systeme == s].correct_juge.fillna(0)) for s in SYSTEMS]
    X = range(len(SYSTEMS))
    ax[0].bar(X, [c[0] for c in cs], yerr=[[c[0] - c[1] for c in cs], [c[2] - c[0] for c in cs]], capsize=4, color="#4c78a8")
    ax[0].set_xticks(X); ax[0].set_xticklabels([LABEL[s].replace(" ", "\n", 1) for s in SYSTEMS], fontsize=8)
    ax[0].set_ylim(0, 1); ax[0].set_title("Réponses correctes (juge)\nquestions avec réponse")
    ab = [bootstrap_ci(noans[noans.systeme == s].abstention.astype(float)) if len(noans[noans.systeme == s]) else (0, 0, 0) for s in SYSTEMS]
    ax[1].bar(X, [c[0] for c in ab], yerr=[[c[0] - c[1] for c in ab], [c[2] - c[0] for c in ab]], capsize=4, color="#e45756")
    ax[1].set_xticks(X); ax[1].set_xticklabels([LABEL[s].replace(" ", "\n", 1) for s in SYSTEMS], fontsize=8)
    ax[1].set_ylim(0, 1); ax[1].set_title("Abstention correcte\nquestions sans réponse")
    fig.suptitle("Résultats provisoires (jeu de test non relu à la main)" if "provisoire" in gen_path.name else "", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "resultats_generation.png", dpi=150)

    r = pd.read_csv(RES / "retrieval_structural_512.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = 0.2
    for k, col in enumerate(["recall@1", "recall@3", "recall@5", "mrr"]):
        ax.bar(np.arange(len(r)) + k * w, r[col], w, label=col)
    ax.set_xticks(np.arange(len(r)) + 1.5 * w); ax.set_xticklabels(r.systeme, fontsize=7, rotation=10)
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title("Retrieval (structurel 512)")
    fig.tight_layout(); fig.savefig(FIG / "retrieval.png", dpi=150)
    print("\nfigures :", [p.name for p in FIG.glob("*.png")])


def annotate(gen_path: Path, n: int = 30) -> None:
    g = pd.read_csv(gen_path)
    g = g[g.systeme.isin(SYSTEMS[1:])]
    s = g.sample(n=min(n, len(g)), random_state=SEED)
    out = pd.DataFrame({"id": s.id, "systeme": s.systeme, "question": s.question,
                        "reference": s.reference.fillna("(aucune réponse dans le corpus : s'abstenir est correct)"),
                        "reponse_du_systeme": s.reponse,
                        "ta_note": "", "commentaire": ""})
    # le juge n'est volontairement PAS montré : on évite d'influencer l'annotateur
    out.to_csv(RES / "annotations_a_remplir.csv", index=False, sep=";", encoding="utf-8-sig")
    s[["id", "systeme", "correct_juge"]].to_csv(RES / "_notes_juge_cachees.csv", index=False)
    print(f"{len(out)} réponses -> results/annotations_a_remplir.csv  (ta_note : 1 = correcte, 0.5 = partielle, 0 = fausse ou abstention à tort)")


def kappa(annot: str = "annotations_a_remplir.csv", col: str = "ta_note") -> None:
    """Accord entre les notes d'un annotateur (humain ou autre) et le juge LLM."""
    h = pd.read_csv(RES / annot, sep=";", encoding="utf-8-sig")
    j = pd.read_csv(RES / "_notes_juge_cachees.csv")
    d = h.merge(j, on=["id", "systeme"])
    d["ta_note"] = pd.to_numeric(d[col].astype(str).str.replace(",", "."), errors="coerce")
    d = d.dropna(subset=["ta_note", "correct_juge"])
    print(f"{len(d)} réponses annotées")
    print("accord brut :", round((d.ta_note == d.correct_juge).mean(), 3))
    h3, j3 = (d.ta_note * 2).astype(int), (d.correct_juge * 2).astype(int)     # 0 / 0.5 / 1  ->  classes 0 / 1 / 2
    print("kappa (3 niveaux 0/0.5/1) :", round(cohen_kappa_score(h3, j3), 3))
    print("kappa pondéré quadratique :", round(cohen_kappa_score(h3, j3, weights="quadratic"), 3))
    print("kappa binaire (1 vs pas 1) :", round(cohen_kappa_score(d.ta_note == 1, d.correct_juge == 1), 3))
    print(pd.crosstab(d.ta_note, d.correct_juge, rownames=["humain"], colnames=["juge"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "annotate", "kappa"])
    ap.add_argument("--gen", default=str(RES / "generations_provisoire.csv"))
    ap.add_argument("--annot", default="annotations_a_remplir.csv")
    ap.add_argument("--col", default="ta_note")
    a = ap.parse_args()
    {"report": lambda: report(Path(a.gen)), "annotate": lambda: annotate(Path(a.gen)),
     "kappa": lambda: kappa(a.annot, a.col)}[a.cmd]()


if __name__ == "__main__":
    main()
