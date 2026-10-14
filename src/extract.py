"""Extraction du texte des PDF (étape 1 du pipeline RAG).

Approche hybride, choisie après comparaison sur des pages réelles du corpus :
  - texte courant : pymupdf (rapide, ordre de lecture correct, casse des sigles
    préservée) ;
  - tableaux      : pdfplumber (texte propre, cellules fusionnées bien gérées),
    réécrits ligne par ligne « colonne : valeur » pour que chaque ligne garde
    ses en-têtes une fois découpée en chunks ;
  - titres        : repérés par la taille de police (INRS) ou par la forme
    « Article N », « CHAPITRE », « ANNEXE » (textes EUR-Lex) ;
  - nettoyage     : en-têtes/pieds de page répétés, césures, puces Wingdings,
    numéros de paragraphe (« 1. ») recollés à leur texte.

Sortie : data/processed/text/<document>.json (cache), un objet par document :
{"doc": ..., "pages": [{"page": 1, "text": "...",
                        "blocks": [{"k": "p"|"h"|"tbl", "t": "...", "lvl": 2}, ...]}]}
(« lvl » n'existe que pour les titres, « k » vaut p = paragraphe, h = titre, tbl = tableau.)

Usage : python -m src.extract
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "text"

MARGIN = 0.08          # fraction de la hauteur de page considérée comme marge (en-tête/pied)
MIN_REPEAT = 0.20      # une ligne de marge présente sur >= 20 % des pages est un en-tête/pied
HEADING_RATIO = 1.3    # un titre INRS a une police >= 1,3 x celle du corps de texte
BULLET_FONTS = ("wingdings", "dingbats", "symbol")
LOWER = "a-zàâçéèêëîïôûùüœ"

# Caractères spéciaux désignés par leur code (pas de caractères invisibles dans le source)
SOFT_HYPHEN = chr(0x00AD)
HYPH = chr(0x2010)     # marqueur « coupure de ligne ambiguë », résolu par resolve_hyphens()
LIGATURES = {chr(0xFB01): "fi", chr(0xFB02): "fl", chr(0xFB00): "ff", chr(0xFB03): "ffi", chr(0xFB04): "ffl"}
LIGATURE_CLASS = "[" + "".join(LIGATURES) + "]"
# « � », caractères de contrôle, zone d'usage privé (codes de puces Wingdings, etc.)
JUNK_RE = re.compile(f"[{chr(0xFFFD)}{chr(0)}-{chr(8)}{chr(0x0B)}-{chr(0x1F)}{chr(0xE000)}-{chr(0xF8FF)}]")
NARROW_SPACES = {0xA0: " ", 0x2009: " ", 0x202F: " "}

# Titres des textes EUR-Lex : motif -> niveau hiérarchique
EURLEX_HEADINGS = [
    (re.compile(r"(CHAPITRE|ANNEXE|TITRE)\s+[IVXLC\d]+\w*"), 1),
    (re.compile(r"(SECTION|PARTIE)\s+[\w\d]+"), 2),
    (re.compile(r"Article\s+(\d+\w?|premier)"), 3),
]


# ---------------------------------------------------------------- nettoyage du texte

def clean_text(text: str) -> str:
    """Caractères parasites, espaces spéciales, ligatures."""
    text = JUNK_RE.sub("", text).translate(NARROW_SPACES)
    text = re.sub(f"({LIGATURE_CLASS}) (?=[{LOWER}])", r"\1", text)   # « vérifi cateur » -> « vérificateur »
    text = text.translate(str.maketrans(LIGATURES))
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _block_text(block: dict) -> str:
    """Texte d'un bloc pymupdf (dict) : puces Wingdings -> '•', lignes recollées."""
    lines = []
    for line in block["lines"]:
        parts = []
        for span in line["spans"]:
            txt = span["text"]
            if any(f in span["font"].lower() for f in BULLET_FONTS) and len(txt.strip()) == 1:
                txt = "•"
            parts.append(txt)
        lines.append("".join(parts).strip())
    text = "\n".join(l for l in lines if l)
    text = text.replace(SOFT_HYPHEN + "\n", "").replace(SOFT_HYPHEN, "")  # fin de ligne : on recolle
    # « per-\nmettre » : coupure ambiguë (césure ou vrai trait d'union ?) -> marqueur, résolu plus tard
    text = re.sub(r"(\w)-\n(?=[" + LOWER + "])", r"\1" + HYPH, text)
    return clean_text(re.sub(r"\s*\n\s*", " ", text))


def _in_table(bbox, tables) -> bool:
    """Vrai si le centre du bloc tombe dans l'un des tableaux de la page."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in tables)


# ---------------------------------------------------------------- tableaux

def _clean_cell(c) -> str:
    c = (c or "").replace(SOFT_HYPHEN + "\n", "").replace(SOFT_HYPHEN, "")
    c = re.sub(r"\(cid:\d+\)", chr(0x2610), c)   # glyphe de case à cocher non décodé par pdfplumber (☐)
    return clean_text(re.sub(r"\s+", " ", c))


def _render_table(rows: list[list]) -> str:
    """Tableau -> lignes « en-tête : valeur | en-tête : valeur » (lignes de section isolées)."""
    rows = [[_clean_cell(c) for c in r] for r in rows]
    header, body = rows[0], rows[1:]
    out = []
    for r in body:
        filled = [(h, v) for h, v in zip(header, r) if v]
        if not filled:
            continue
        if len(r) > 2 and len(filled) == 1 and r[0]:
            out.append(r[0])                               # ligne de section (cellule fusionnée)
        else:
            out.append(" | ".join(f"{h} : {v}" if h else v for h, v in filled))
    return "\n".join(out)


def _page_tables(plumber_page) -> list[tuple[tuple, str]]:
    """Vrais tableaux de la page : [(bbox, texte)]. Écarte les faux positifs
    (figures encadrées, schémas) via la proportion de cellules non vides."""
    found = []
    try:
        tables = plumber_page.find_tables()
    except Exception:
        return found
    for t in tables:
        rows = t.extract()
        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        cells = [c for r in rows for c in r]
        nonempty = sum(bool(_clean_cell(c)) for c in cells)
        if nonempty / len(cells) < 0.35 or nonempty < 4:
            continue
        text = _render_table(rows)
        if text:
            found.append((t.bbox, text))
    return found


# ---------------------------------------------------------------- titres

def _heading_level(text: str, size: float, body: float, eurlex: bool):
    """Niveau du titre (1 = le plus haut) ou None si le bloc n'est pas un titre."""
    if eurlex:
        for pat, lvl in EURLEX_HEADINGS:
            if pat.fullmatch(text):
                return lvl
        return None
    if size < max(HEADING_RATIO * body, body + 3) or not 3 <= len(text) <= 150:
        return None
    if re.fullmatch(r"(\w ?)+", text) and len(text) < 12:      # lettrines « C H A P I T R E »
        return None
    if text.startswith("•") or len(re.findall(r"[^\W\d_]", text)) < 3:   # puces, symboles
        return None
    m = re.match(r"(\d+(?:\.\d+)*)\.?\s", text)
    if m:
        return m.group(1).count(".") + 1                       # « 4.3.1. » -> niveau 3
    return 1 if size >= 1.8 * body else 2


def _mark_headings(items: list[dict], body: float, eurlex: bool) -> None:
    """Étiquette les blocs « h » et fusionne les titres coupés sur plusieurs blocs
    (« 4.4. » + « Écartements… » + « d'écrasement » ; « Article 31 » + son intitulé)."""
    prev = None
    for it in items:
        if it["k"] != "p":
            prev = None
            continue
        lvl = None if it["margin"] else _heading_level(it["t"], it["size"], body, eurlex)
        if lvl is not None:
            it["k"], it["lvl"] = "h", lvl
        if eurlex:
            # l'intitulé d'un article/chapitre suit son numéro, avec le même corps de police
            if prev and prev["k"] == "h" and it["k"] == "p" and len(it["t"]) <= 150 and not prev.get("titled"):
                prev["t"] += " - " + it["t"]
                prev["titled"] = True
                it["k"] = "drop"
                continue
        elif prev and prev["k"] == "h" and it["k"] == "h" and abs(prev["size"] - it["size"]) < 0.5:
            starts_numbered = re.match(r"\d+(\.\d+)*\.?\s", it["t"])
            if re.fullmatch(r"\d+(\.\d+)*\.?", prev["t"]) or not starts_numbered:
                prev["t"] += " " + it["t"]
                it["k"] = "drop"
                continue
        prev = it


# ---------------------------------------------------------------- document

def _signature(text: str) -> str:
    """Forme normalisée d'une ligne de marge (chiffres masqués) pour repérer les répétitions."""
    t = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t)).strip()


def extract_document(path: Path) -> list[dict]:
    """Pages du PDF sous forme de blocs étiquetés (les coupures ambiguës portent le marqueur HYPH)."""
    eurlex = path.name.startswith("EURLEX_")
    doc = pymupdf.open(path)
    pages, size_chars = [], Counter()

    with pdfplumber.open(path) as pl:
        for i, page in enumerate(doc):
            H = page.rect.height
            tables = _page_tables(pl.pages[i])
            items = []
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0 or _in_table(b["bbox"], [t[0] for t in tables]):
                    continue
                txt = _block_text(b)
                if not txt:
                    continue
                spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
                size = max((s["size"] for s in spans), default=0)
                is_margin = b["bbox"][3] < MARGIN * H or b["bbox"][1] > (1 - MARGIN) * H
                if not is_margin:
                    for s in spans:
                        size_chars[round(s["size"] * 2) / 2] += len(s["text"])
                items.append({"y": b["bbox"][1], "t": txt, "k": "p", "margin": is_margin, "size": size})
            for bbox, txt in tables:
                items.append({"y": bbox[1], "t": txt, "k": "tbl", "margin": False, "size": 0})
            pages.append(items)

    body = size_chars.most_common(1)[0][0] if size_chars else 10.0

    # signatures de marge -> en-têtes/pieds répétés
    sig_pages = Counter()
    for items in pages:
        for s in {_signature(it["t"]) for it in items if it["margin"]}:
            sig_pages[s] += 1
    repeated = {s for s, n in sig_pages.items() if n >= max(2, int(MIN_REPEAT * len(pages)))}

    result = []
    for i, items in enumerate(pages, start=1):
        # ordre : blocs de texte dans l'ordre pymupdf ; chaque tableau est inséré
        # avant le premier bloc de texte situé sous son bord supérieur
        seq = [it for it in items if it["k"] != "tbl"]
        for t in (it for it in items if it["k"] == "tbl"):
            seq.insert(next((k for k, it in enumerate(seq) if it["y"] > t["y"]), len(seq)), t)
        seq = [it for it in seq if not (it["margin"] and (
            _signature(it["t"]) in repeated or re.fullmatch(r"\d{1,3}\s*\w?", it["t"])))]

        # numéros de paragraphe « 1. » / « (a) » isolés -> recollés au bloc suivant
        merged, pending = [], ""
        for it in seq:
            if it["k"] == "p" and re.fullmatch(r"(\d+\.|\(\w{1,3}\))", it["t"]):
                pending = it["t"] + " "
                continue
            it["t"] = pending + it["t"]
            pending = ""
            # coupure de mot à cheval sur deux blocs : « at- » + « traction »
            if (merged and it["k"] == "p" and merged[-1]["k"] == "p"
                    and re.search(r"\w-$", merged[-1]["t"]) and re.match(f"[{LOWER}]", it["t"])):
                merged[-1]["t"] = merged[-1]["t"][:-1] + HYPH + it["t"]
            else:
                merged.append(it)

        _mark_headings(merged, body, eurlex)
        blocks = []
        for it in merged:
            if it["k"] == "drop":
                continue
            blocks.append({"k": it["k"], "t": it["t"], **({"lvl": it["lvl"]} if it["k"] == "h" else {})})
        result.append({"page": i, "blocks": blocks})
    return result


def resolve_hyphens(texts: list[str], vocab_texts: list[str]) -> list[str]:
    """Résout les coupures ambiguës « mot‐suite » avec un vocabulaire tiré du corpus :
    on recolle (« at‐traction » -> « attraction ») si la forme soudée existe ailleurs
    dans le corpus, sinon on garde le trait d'union (« main‐bras » -> « main-bras »)."""
    word = r"[^\W\d_]+"
    vocab = Counter(w.lower() for t in vocab_texts for w in re.findall(word, t.replace(HYPH, " ")))

    def repl(m):
        a, b = m.group(1), m.group(2)
        return a + b if vocab[(a + b).lower()] > 0 else f"{a}-{b}"

    out = [re.sub(f"({word}){HYPH}({word})", repl, t) for t in texts]
    return [t.replace(HYPH, "-") for t in out]       # marqueurs restants (chiffres, etc.)


def _page_text(blocks: list[dict]) -> str:
    return "\n".join(b["t"] for b in blocks)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fresh = {}                                           # documents extraits lors de cet appel
    for pdf in sorted(RAW.glob("*.pdf")):
        dest = OUT / f"{pdf.stem}.json"
        if dest.exists() and dest.stat().st_mtime >= pdf.stat().st_mtime:
            continue                                     # cache valide
        fresh[pdf.stem] = extract_document(pdf)
    if not fresh:
        return

    # vocabulaire du corpus complet (documents en cache + nouveaux) pour résoudre les césures
    vocab_texts = [b["t"] for pages in fresh.values() for p in pages for b in p["blocks"]]
    for f in OUT.glob("*.json"):
        if f.stem not in fresh:
            for p in json.loads(f.read_text(encoding="utf-8"))["pages"]:
                vocab_texts += [b["t"] for b in p["blocks"]]

    for name, pages in fresh.items():
        blocks = [b for p in pages for b in p["blocks"]]
        for b, t in zip(blocks, resolve_hyphens([b["t"] for b in blocks], vocab_texts)):
            b["t"] = t
        for p in pages:
            p["text"] = _page_text(p["blocks"])
        (OUT / f"{name}.json").write_text(json.dumps({"doc": name, "pages": pages}, ensure_ascii=False), encoding="utf-8")
        n_head = sum(b["k"] == "h" for b in blocks)
        print(f"{name:42} {len(pages):4} p. {sum(len(p['text']) for p in pages):8} car. {n_head:4} titres")


if __name__ == "__main__":
    main()
