"""Génération du jeu de test (questions de référence) avec un LLM, puis vérification automatique.

Chaque question vient d'un passage réel du corpus. Le LLM doit recopier MOT POUR MOT
l'extrait qui prouve la réponse (« preuve ») ; on vérifie que cet extrait existe bien
dans le document source et on en retrouve la page. Les questions dont la preuve n'est
pas retrouvée sont rejetées automatiquement. Le résultat est ensuite relu à la main
(fichier CSV) : le LLM propose, l'humain valide.

Types de questions : factuelle, numérique/tabulaire, multi-passages (deux sources à
combiner) et sans réponse dans le corpus (pour tester l'abstention).

Usage : python -m src.testset            (génération complète, reprend grâce au cache)
        python -m src.testset --quick    (2 questions par type, pour juger la qualité)
"""
import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import llm

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks" / "structural_512.jsonl"
TEXT_DIR = ROOT / "data" / "processed" / "text"
OUT_DIR = ROOT / "data" / "testset"

SEED = 42
GENERATORS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]     # 2e = repli si le quota journalier du 1er est atteint
# brouillons visés par type (on en génère plus que les ~150 finaux : la relecture en éliminera)
TARGETS = {"factuelle": 90, "numerique": 55, "multi": 45, "sans_reponse": 35}
STYLES = ["Quel / Quelle", "Comment", "Dans quel cas", "Que doit faire / Que doivent faire",
          "Qui est responsable de", "À quelle condition", "Quelle différence entre", "Que signifie", "Combien"]
BANNED = re.compile(r"\b(l'extrait|le passage|le texte|ce texte|ce document|le document|ci-dessus|cet extrait)\b", re.I)

SYSTEM = ("Tu es un expert en sécurité des machines industrielles. Tu construis un jeu de test pour évaluer "
          "un système de questions-réponses sur des documents réglementaires et techniques (INRS, "
          "règlement machines UE 2023/1230, directive 2006/42/CE). Tu réponds uniquement par un objet JSON.")

RULES = """Règles pour la question :
- Elle doit être AUTONOME : compréhensible sans voir l'extrait. N'écris jamais « l'extrait », « le texte », « le passage » ni « le document ».
- Elle doit contenir ce qui l'identifie sans ambiguïté (type de machine, équipement, article, norme, étape...) : une question comme « Quelle est la fréquence ? » est interdite.
- Elle doit porter sur un point précis, pas sur le sujet général. Style suggéré : {style}.
- Écris en français naturel, comme un préventeur, un concepteur ou un chef d'établissement.
- La réponse de référence ne contient QUE des faits écrits explicitement dans l'extrait : aucune explication, justification ni généralité ajoutée de ton propre chef.
- Si l'extrait ne permet pas une question utile et précise (case de formulaire, liste de références, en-tête, contenu trop vague), réponds {{"question": null}}."""


# ---------------------------------------------------------------- outils texte

def norm(t: str) -> str:
    """Forme de comparaison : apostrophes/espaces/casse neutralisés."""
    t = unicodedata.normalize("NFKC", t).replace("’", "'").replace("‘", "'").replace("«", '"').replace("»", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def find_page(doc: str, quote: str, default: int) -> int:
    """Page du document où se trouve la preuve (sinon page de début du chunk)."""
    q = norm(quote)
    d = json.loads((TEXT_DIR / f"{doc}.json").read_text(encoding="utf-8"))
    for p in d["pages"]:
        if q in norm(p["text"]):
            return p["page"]
    return default


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- sélection des passages sources

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:mm|cm|m|s|ms|kg|n|%|hz|db|v|a|mois|ans?|jours?|heures?|h|bar|pl[a-e]|sil)\b", re.I)


def load_pool() -> list[dict]:
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines()]
    pool = []
    for c in chunks:
        sec = c["section"].lower()
        letters = sum(ch.isalpha() for ch in c["text"]) / max(1, len(c["text"]))
        if (c["n_tokens"] < 150 or letters < 0.6 or "bibliograph" in sec or "sommaire" in sec
                or c["text"].count("☐") > 2                      # formulaires à cocher
                or re.search(r"isbn|dépôt légal|reproduction", c["text"], re.I)):
            continue
        rows = sum(" : " in l and " | " in l for l in c["text"].split("\n"))
        c["numeric"] = rows >= 3 or len(NUM_RE.findall(c["text"])) >= 4
        pool.append(c)
    return pool


def sample(pool: list[dict], n: int, rng: random.Random, used: set) -> list[dict]:
    """n chunks répartis sur les documents (poids ~ racine de la taille du document), sans répétition."""
    by_doc = {}
    for c in pool:
        if c["chunk_id"] not in used:
            by_doc.setdefault(c["doc"], []).append(c)
    for v in by_doc.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and by_doc:
        docs = list(by_doc)
        d = rng.choices(docs, weights=[math.sqrt(len(by_doc[k])) for k in docs])[0]
        c = by_doc[d].pop()
        if not by_doc[d]:
            del by_doc[d]
        out.append(c)
        used.add(c["chunk_id"])
    return out


def neighbours(pool: list[dict]):
    """Matrice de similarité TF-IDF entre chunks (pour trouver un second passage lié au premier)."""
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.5)
    X = vec.fit_transform([c["text"] for c in pool])
    return (X @ X.T).toarray()


# ---------------------------------------------------------------- prompts

def passage(c: dict) -> str:
    return f"[{c['context']}, page {c['page_start']}]\n{c['text']}"


def prompt_single(kind: str, c: dict, style: str) -> str:
    focus = ("La question doit demander une VALEUR PRÉCISE (nombre, seuil, durée, fréquence, distance, niveau, "
             "classe...) ou une information à lire dans un tableau ; la réponse contient cette valeur avec son unité."
             if kind == "numerique" else
             "La question doit avoir une réponse trouvable explicitement dans l'extrait.")
    return (f"Voici un extrait d'un document :\n\"\"\"\n{passage(c)}\n\"\"\"\n\n"
            f"Rédige UNE question dont la réponse se trouve dans cet extrait. {focus}\n\n"
            f"{RULES.format(style=style)}\n"
            "- « reponse » : 1 à 3 phrases, fondée uniquement sur l'extrait.\n"
            "- « preuve » : copie MOT POUR MOT (copier-coller exact, rien modifié) la ou les 1-2 phrases de l'extrait "
            "qui justifient la réponse.\n\n"
            'Réponds par : {"question": "...", "reponse": "...", "preuve": "..."}')


def prompt_multi(a: dict, b: dict, style: str) -> str:
    return (f"Voici deux extraits.\n\nEXTRAIT A :\n\"\"\"\n{passage(a)}\n\"\"\"\n\nEXTRAIT B :\n\"\"\"\n{passage(b)}\n\"\"\"\n\n"
            "Rédige UNE question qui nécessite de COMBINER une information de A et une information de B pour "
            "répondre (ni A seul ni B seul ne suffisent). Le lien entre A et B doit être naturel et porter sur le "
            "même sujet précis (même équipement, même obligation, même notion). Si ce lien n'existe pas, "
            'réponds {"question": null} : ne force pas un rapprochement artificiel.\n\n'
            f"{RULES.format(style=style)}\n"
            "- « reponse » : 2 à 4 phrases, fondée uniquement sur A et B.\n"
            "- « preuve_a » et « preuve_b » : copie MOT POUR MOT la phrase de A, puis celle de B, qui servent à répondre.\n\n"
            'Réponds par : {"question": "...", "reponse": "...", "preuve_a": "...", "preuve_b": "..."}')


def prompt_noanswer(c: dict, angle: str, style: str) -> str:
    return (f"Voici un extrait d'un document sur la sécurité des machines :\n\"\"\"\n{passage(c)}\n\"\"\"\n\n"
            "Rédige UNE question réaliste, du même domaine et du même sujet, que ce document NE PERMET PAS de résoudre : "
            f"{angle}. La question doit paraître légitime et précise (pas absurde, pas hors sujet), sans que la "
            "réponse figure dans l'extrait ni puisse se déduire de lui.\n\n"
            f"{RULES.format(style=style)}\n\n"
            'Réponds par : {"question": "...", "pourquoi_sans_reponse": "..."}')


ANGLES = ["elle demande une valeur chiffrée, un seuil ou une durée précis qui n'est pas mentionné",
          "elle porte sur un autre type de machine ou d'équipement que ceux décrits",
          "elle porte sur une règle propre à un secteur d'activité (agriculture, bâtiment, mines...) que l'extrait ne traite pas",
          "elle demande un exemple concret, un cas d'application ou une marque que l'extrait ne donne pas",
          "elle porte sur une exigence d'une norme ou d'un texte que l'extrait ne cite pas"]


# ---------------------------------------------------------------- génération

def call(prompt: str, state: dict) -> dict | None:
    """Appelle le générateur courant ; bascule sur le suivant si le quota journalier est atteint."""
    while state["g"] < len(GENERATORS):
        model = GENERATORS[state["g"]]
        try:
            r = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], model=model,
                         temperature=0.7, max_tokens=900, json_mode=True,
                         reasoning_effort="low" if "gpt-oss" in model else None, seed=state["seed"])
        except RuntimeError as e:
            print(f"  ! {model} indisponible ({str(e)[:90]}) -> générateur suivant")
            state["g"] += 1
            continue
        r["parsed"] = parse_json(r["text"])
        return r
    return None


def build_item(kind, r, chunks, quotes, extra=None):
    p = r["parsed"]
    if p and p.get("question") is None:
        return None, "ignoree_par_le_llm"
    if not p or not p.get("question") or not p.get("reponse" if kind != "sans_reponse" else "question"):
        return None, "json_invalide"
    q = p["question"].strip()
    if BANNED.search(norm(q)):
        return None, "question_non_autonome"
    if not (25 <= len(q) <= 300) or not q.endswith("?"):
        return None, "question_forme"
    ev = []
    for quote, c in zip(quotes, chunks):
        if kind == "sans_reponse":
            break
        quote = (quote or "").strip()
        if len(quote) < 20 or norm(quote) not in norm(c["text"]):
            return None, "preuve_introuvable"
        ev.append({"doc": c["doc"], "page": find_page(c["doc"], quote, c["page_start"]), "quote": quote,
                   "chunk_id_structural_512": c["chunk_id"]})
    item = {"type": kind, "question": q, "reponse": (p.get("reponse") or "").strip() or None,
            "preuves": ev, "generateur": r["model"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "source_chunks": [c["chunk_id"] for c in chunks]}
    if kind == "sans_reponse":
        item["pourquoi_sans_reponse"] = p.get("pourquoi_sans_reponse")
        item["reponse"] = None
    return item, "ok"


def generate(targets: dict) -> list[dict]:
    rng = random.Random(SEED)
    pool = load_pool()
    sim = neighbours(pool)
    idx = {c["chunk_id"]: i for i, c in enumerate(pool)}
    used, items, rejects = set(), [], Counter()
    state = {"g": 0, "seed": SEED}
    plain = [c for c in pool if not c["numeric"]]
    numeric = [c for c in pool if c["numeric"]]
    print(f"pool : {len(pool)} chunks ({len(numeric)} numériques/tabulaires, {len(plain)} textuels)")

    plan = []
    for c in sample(plain, targets["factuelle"], rng, used):
        plan.append(("factuelle", [c]))
    for c in sample(numeric, targets["numerique"], rng, used):
        plan.append(("numerique", [c]))
    for a in sample(pool, targets["multi"], rng, used):
        row = sim[idx[a["chunk_id"]]]
        cand = [j for j in np.argsort(-row)[:15]
                if 0.12 <= row[j] <= 0.65 and pool[j]["chunk_id"] != a["chunk_id"]
                and not (pool[j]["doc"] == a["doc"] and abs(j - idx[a["chunk_id"]]) <= 1)]
        if cand:
            plan.append(("multi", [a, pool[rng.choice(cand)]]))
    for c in sample(plain, targets["sans_reponse"], rng, used):
        plan.append(("sans_reponse", [c]))
    rng.shuffle(plan)

    for n, (kind, chunks) in enumerate(plan, 1):
        style = rng.choice(STYLES)
        if kind == "multi":
            prompt = prompt_multi(chunks[0], chunks[1], style)
        elif kind == "sans_reponse":
            prompt = prompt_noanswer(chunks[0], rng.choice(ANGLES), style)
        else:
            prompt = prompt_single(kind, chunks[0], style)
        r = call(prompt, state)
        if r is None:
            print("Plus aucun générateur disponible : on s'arrête, relancer plus tard (le cache est conservé).")
            break
        p = r["parsed"] or {}
        quotes = [p.get("preuve_a"), p.get("preuve_b")] if kind == "multi" else [p.get("preuve")]
        item, why = build_item(kind, r, chunks, quotes)
        rejects[why] += 1
        if item:
            items.append(item)
        if n % 10 == 0 or n == len(plan):
            print(f"  {n}/{len(plan)} appels | gardées {len(items)} | {dict(rejects)}")
    return items


def dedupe(items: list[dict], thr: float = 0.75) -> list[dict]:
    """Retire les questions quasi identiques (similarité TF-IDF > thr)."""
    if len(items) < 2:
        return items
    X = TfidfVectorizer(sublinear_tf=True).fit_transform([i["question"] for i in items])
    S = (X @ X.T).toarray()
    keep = []
    for i in range(len(items)):
        if all(S[i, j] <= thr for j in keep):
            keep.append(i)
    return [items[i] for i in keep]


REVIEW_COLS = ["id", "type", "question", "reponse_reference", "source", "preuve", "passages_proches",
               "verdict", "question_corrigee", "reponse_corrigee", "commentaire"]


def export_review() -> Path:
    """CSV de relecture. Pour les questions « sans réponse », on ajoute les 3 passages du corpus les plus
    proches (BM25) : c'est là qu'il faut vérifier que la réponse n'existe pas."""
    import csv
    from src.bm25 import BM25
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines()]
    bm = BM25([c["context"] + "\n" + c["text"] for c in chunks])
    items = [json.loads(l) for l in (OUT_DIR / "brouillon.jsonl").read_text(encoding="utf-8").splitlines()]
    order = {"factuelle": 0, "numerique": 1, "multi": 2, "sans_reponse": 3}
    items.sort(key=lambda i: (order[i["type"]], i["id"]))
    path = OUT_DIR / "questions_a_verifier.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS, delimiter=";")
        w.writeheader()
        for it in items:
            near = ""
            if it["type"] == "sans_reponse":
                near = "\n".join(f"[{chunks[i]['doc']} p.{chunks[i]['page_start']}] {chunks[i]['text'][:260]}"
                                 for i, _ in bm.search(it["question"], 3))
            w.writerow({"id": it["id"], "type": it["type"], "question": it["question"],
                        "reponse_reference": it["reponse"] or "(aucune : la question n'a pas de réponse dans le corpus)",
                        "source": " | ".join(f"{e['doc']} p.{e['page']}" for e in it["preuves"]) or "-",
                        "preuve": "\n".join(e["quote"] for e in it["preuves"]), "passages_proches": near,
                        "verdict": "", "question_corrigee": "", "reponse_corrigee": "", "commentaire": ""})
    return path


def finalize() -> Path:
    """Reprend le CSV relu : OK = gardée, CORRIGER = gardée avec la correction, KO (ou vide) = écartée."""
    import csv
    drafts = {i["id"]: i for i in map(json.loads, (OUT_DIR / "brouillon.jsonl").read_text(encoding="utf-8").splitlines())}
    final, skipped = [], Counter()
    with open(OUT_DIR / "questions_a_verifier.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            v = row["verdict"].strip().upper()
            it = drafts[row["id"]]
            if v not in ("OK", "CORRIGER"):
                skipped[v or "non relue"] += 1
                continue
            it = {k: it[k] for k in ("id", "type", "question", "reponse", "preuves", "generateur", "source_chunks")}
            tidy = str.maketrans({" ": " ", " ": " ", "‑": "-", "’": "'"})   # espaces/traits insécables des LLM
            it["question"] = it["question"].translate(tidy)
            it["reponse"] = it["reponse"].translate(tidy) if it["reponse"] else None
            if v == "CORRIGER":
                it["question"] = row["question_corrigee"].strip() or it["question"]
                if it["type"] != "sans_reponse":
                    it["reponse"] = row["reponse_corrigee"].strip() or it["reponse"]
                it["corrigee_a_la_main"] = True
            it["note"] = row["commentaire"].strip()
            final.append(it)
    path = OUT_DIR / "questions.jsonl"
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in final), encoding="utf-8")
    print(f"{len(final)} questions validées -> {path.name} | par type : {dict(Counter(i['type'] for i in final))}"
          f" | écartées : {dict(skipped)}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="generate", choices=["generate", "export", "finalize"])
    ap.add_argument("--quick", action="store_true", help="2 questions par type")
    args = ap.parse_args()
    if args.cmd == "export":
        print("CSV de relecture :", export_review())
        return
    if args.cmd == "finalize":
        finalize()
        return
    targets = {k: 2 for k in TARGETS} if args.quick else TARGETS
    items = dedupe(generate(targets))
    for k, it in enumerate(items, 1):
        it["id"] = f"Q{k:03d}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / ("brouillon_quick.jsonl" if args.quick else "brouillon.jsonl")
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    print(f"\n{len(items)} questions -> {path.name} | par type : {dict(Counter(i['type'] for i in items))}")


if __name__ == "__main__":
    main()
