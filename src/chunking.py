"""Découpage du corpus en chunks (étape 2 du pipeline RAG).

Deux stratégies comparées en ablation, appliquées sur la MÊME entrée nettoyée :
  - fixe       : fenêtre glissante de N tokens avec recouvrement, sans tenir compte
                 de la structure du document (baseline) ;
  - structurel : on ne traverse jamais un titre (article, section) ; les paragraphes
                 sont regroupés jusqu'à N tokens et les lignes de tableau ne sont
                 jamais coupées au milieu.

Chaque chunk porte un « contexte » (titre du document, et pour la stratégie
structurelle le fil d'Ariane des titres) ajouté devant le texte pour l'indexation.
Le contexte est compté dans le budget de N tokens : un chunk indexé ne dépasse
jamais N tokens, ce qui rend les tailles comparables.

Les tokens sont comptés avec le tokenizer de bge-m3, le modèle d'embeddings
envisagé : « 512 tokens » est donc ce que le modèle verra réellement.

Usage : python -m src.chunking
"""
import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "processed" / "text"
CHUNK_DIR = ROOT / "data" / "processed" / "chunks"
SOURCES_CSV = ROOT / "data" / "sources.csv"

TOKENIZER_NAME = "BAAI/bge-m3"
MIN_TOKENS = 80          # (structurel) une section plus courte est fusionnée avec la suivante
OVERLAP = 0.15           # (fixe) recouvrement entre fenêtres consécutives
BOILERPLATE_MIN_DOCS = 3  # un bloc identique dans >= 3 documents est du texte « maison » répété


@lru_cache(maxsize=1)
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def n_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------- chargement + nettoyage commun

def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def _is_toc_page(blocks: list[dict]) -> bool:
    """Page de sommaire : surtout des lignes « 1.2. Titre 14 » (numéro de page en fin de ligne)."""
    lines = [l for b in blocks for l in b["t"].split("\n") if l.strip()]
    if any(_norm(b["t"]) in ("sommaire", "table des matières") for b in blocks):
        return True
    toc = sum(bool(re.search(r"\S\s+\d{1,3}$", l)) and bool(re.match(r"\d", l)) for l in lines)
    return len(lines) >= 6 and toc >= 0.5 * len(lines)


def load_corpus() -> dict[str, dict]:
    """{doc: {"title": ..., "blocks": [{"page", "k", "t", "lvl"?}]}} nettoyé :
    sommaires retirés, blocs « maison » répétés dans plusieurs documents retirés."""
    titles = {}
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, encoding="utf-8") as f:
            titles = {Path(r["fichier"]).stem: r["titre"] for r in csv.DictReader(f)}

    docs, stats = {}, Counter()
    for f in sorted(TEXT_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        blocks = []
        for p in d["pages"]:
            if _is_toc_page(p["blocks"]):
                stats["pages_sommaire"] += 1
                continue
            blocks += [{**b, "page": p["page"]} for b in p["blocks"]]
        docs[d["doc"]] = {"title": titles.get(d["doc"], d["doc"]), "blocks": blocks}

    # texte « maison » (présentation de l'INRS, mentions légales...) répété dans plusieurs documents
    seen = defaultdict(set)
    for name, d in docs.items():
        for b in d["blocks"]:
            if b["k"] == "p" and len(b["t"]) >= 40:
                seen[_norm(b["t"])].add(name)
    boiler = {t for t, ds in seen.items() if len(ds) >= BOILERPLATE_MIN_DOCS}
    for d in docs.values():
        kept = [b for b in d["blocks"] if not (b["k"] == "p" and _norm(b["t"]) in boiler)]
        stats["blocs_maison"] += len(d["blocks"]) - len(kept)
        d["blocks"] = kept
    load_corpus.stats = dict(stats)
    return docs


# ---------------------------------------------------------------- stratégie 1 : fixe

def chunk_fixed(doc: str, d: dict, size: int, overlap: float = OVERLAP) -> list[dict]:
    """Fenêtres de `size` tokens (contexte inclus), recouvrement `overlap`, sans égard à la structure."""
    context = d["title"]
    budget = size - n_tokens(context) - 4      # marge : le texte découpé est re-tokenisé avec son contexte
    text, spans = "", []                       # spans : (début, fin, page) de chaque bloc dans `text`
    for b in d["blocks"]:
        start = len(text)
        text += b["t"] + "\n"
        spans.append((start, len(text), b["page"]))
    if not text.strip():
        return []

    enc = tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    step = max(1, int(budget * (1 - overlap)))
    chunks, i = [], 0
    while i < len(offs):
        j = min(i + budget, len(offs))
        c0, c1 = offs[i][0], offs[j - 1][1]
        chunks.append(_make(doc, d, context, text[c0:c1].strip(), _pages(spans, c0, c1), ""))
        if j == len(offs):
            break
        i += step
    return chunks


def _pages(spans, c0, c1):
    pages = [p for s, e, p in spans if e > c0 and s < c1]
    return (pages[0], pages[-1]) if pages else (spans[-1][2], spans[-1][2])


# ---------------------------------------------------------------- stratégie 2 : structurelle

def _units(block: dict, budget: int) -> list[str]:
    """Unités insécables d'un bloc : le paragraphe, ou chaque ligne d'un tableau.
    Une unité trop longue est coupée par phrases, puis en dernier recours par tokens."""
    raw = block["t"].split("\n") if block["k"] == "tbl" else [block["t"]]
    out = []
    for u in raw:
        if n_tokens(u) <= budget:
            out.append(u)
            continue
        for s in re.split(r"(?<=[.;:!?])\s+", u):
            if n_tokens(s) <= budget:
                out.append(s)
            else:                                     # phrase géante : découpe dure par tokens
                enc = tokenizer()(s, add_special_tokens=False, return_offsets_mapping=True)
                o = enc["offset_mapping"]
                out += [s[o[k][0]:o[min(k + budget, len(o)) - 1][1]] for k in range(0, len(o), budget)]
    return out


def chunk_structural(doc: str, d: dict, size: int) -> list[dict]:
    chunks, stack = [], {}              # stack : niveau -> texte du titre courant
    cur, cur_tok, cur_pages, cur_ctx = [], 0, [], ""

    def crumb():
        return " > ".join([d["title"]] + [stack[k] for k in sorted(stack)])

    def flush():
        nonlocal cur, cur_tok, cur_pages
        if cur:
            chunks.append(_make(doc, d, cur_ctx, "\n".join(cur), (cur_pages[0], cur_pages[-1]), cur_ctx))
        cur, cur_tok, cur_pages = [], 0, []

    for b in d["blocks"]:
        if b["k"] == "h":
            if cur_tok >= MIN_TOKENS:
                flush()                                   # on ne traverse pas un titre (sauf section minuscule)
            stack = {k: v for k, v in stack.items() if k < b["lvl"]}
            stack[b["lvl"]] = b["t"]
            if not cur:
                cur_ctx = crumb()
            continue
        if not cur:
            cur_ctx = crumb()
        budget = size - n_tokens(cur_ctx) - 1
        for u in _units(b, max(budget, 32)):
            t = n_tokens(u) + 1
            if cur and cur_tok + t > budget:
                flush()
                cur_ctx = crumb()
                budget = size - n_tokens(cur_ctx) - 1
            cur.append(u)
            cur_tok += t
            cur_pages.append(b["page"])
    flush()
    return chunks


# ---------------------------------------------------------------- sortie

def _make(doc, d, context, text, pages, section):
    return {"doc": doc, "page_start": pages[0], "page_end": pages[1],
            "section": section, "context": context, "text": text,
            "n_tokens": n_tokens(f"{context}\n{text}")}


def build(strategy: str, size: int) -> list[dict]:
    docs = load_corpus()
    fn = chunk_fixed if strategy == "fixed" else chunk_structural
    chunks = []
    for name, d in docs.items():
        for k, c in enumerate(fn(name, d, size)):
            chunks.append({"chunk_id": f"{name}#{k:04d}", **c})
    return chunks


def save(chunks: list[dict], strategy: str, size: int) -> Path:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    path = CHUNK_DIR / f"{strategy}_{size}.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    return path


def main() -> None:
    import statistics as st
    print(f"{'config':18} {'chunks':>7} {'tok moy':>8} {'tok méd':>8} {'tok max':>8} {'< 50 tok':>9}")
    for strategy in ("fixed", "structural"):
        for size in (256, 512, 1024):
            ch = build(strategy, size)
            save(ch, strategy, size)
            t = [c["n_tokens"] for c in ch]
            print(f"{strategy + '_' + str(size):18} {len(ch):7} {st.mean(t):8.0f} {st.median(t):8.0f} "
                  f"{max(t):8} {sum(x < 50 for x in t):9}")
    print("nettoyage commun :", load_corpus.stats)


if __name__ == "__main__":
    main()
