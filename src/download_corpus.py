"""Téléchargement automatique du corpus (brochures INRS + textes EUR-Lex).

Les PDF vont dans data/raw/ (ignoré par git). La liste des sources
(URL, titre, nb de pages) est écrite dans data/sources.csv, qui lui est suivi
par git : c'est la « source du dataset » demandée au notebook.

Usage : python -m src.download_corpus
"""
import csv
import time
from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES_CSV = ROOT / "data" / "sources.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (projet-RAG-securite-machines)"}
INRS_URL = "https://www.inrs.fr/dam/inrs/CataloguePapier/ED/TI-ED-{n}.pdf"
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/FR/TXT/PDF/?uri=CELEX:{celex}"

# Brochures INRS candidates, repérées sur les pages « publications » INRS
# consacrées à la conception et à l'utilisation des machines.
INRS_REFS = [
    54, 113, 114, 123, 129, 134, 781, 828, 879, 912, 4450, 5100, 6016, 6038,
    6104, 6108, 6109, 6112, 6122, 6129, 6154, 6176, 6178, 6270, 6281, 6310,
    6339, 6389,
    # Ajoutées via le catalogue INRS « Machines » (ED 4707) et des recherches ciblées.
    126, 6231, 6289, 6323, 6330, 6386, 6446, 6499, 6531, 6540,
]

# Titres absents ou mal extraits des métadonnées PDF.
TITLE_OVERRIDES = {
    "INRS_ED6038.pdf": "Intervention sur un équipement de travail. Réflexions pour la sécurité",
    "EURLEX_directive_2006_42_CE_machines.pdf": "Directive 2006/42/CE du Parlement européen et du Conseil du 17 mai 2006 relative aux machines",
    "EURLEX_reglement_2023_1230_machines.pdf": "Règlement (UE) 2023/1230 du Parlement européen et du Conseil du 14 juin 2023 sur les machines",
}

# Brochures dont l'URL PDF ne suit pas le schéma TI-ED-{n}.pdf.
INRS_URL_OVERRIDES = {
    6499: "https://www.inrs.fr/dam/jcr:cd912671-9252-4bc0-a941-21efbc69ab58/ed6499.pdf",
    6330: "https://www.inrs.fr/dms/inrs/CataloguePapier/ED/TI-ED-6330/ed6330.pdf",
}

# (nom de fichier, CELEX)
EURLEX_DOCS = [
    ("EURLEX_reglement_2023_1230_machines", "32023R1230"),
    ("EURLEX_directive_2006_42_CE_machines", "32006L0042"),
]


def fetch_pdf(url: str, dest: Path) -> bool:
    """Télécharge url vers dest ; renvoie True seulement si c'est un vrai PDF."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        print(f"  ! erreur réseau : {e}")
        return False
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        print(f"  ! HTTP {r.status_code}, pas un PDF")
        return False
    dest.write_bytes(r.content)
    return True


def describe(path: Path) -> tuple[str, int]:
    """Titre (métadonnées, sinon début de la 1re page) et nombre de pages."""
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title", "").strip()
        if not title and doc.page_count:
            title = " ".join(doc[0].get_text().split())[:120]
        return title, doc.page_count


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(f"INRS_ED{n}", INRS_URL_OVERRIDES.get(n, INRS_URL.format(n=n))) for n in INRS_REFS]
    jobs += [(name, EURLEX_URL.format(celex=c)) for name, c in EURLEX_DOCS]

    rows, failed = [], []
    for name, url in jobs:
        dest = RAW / f"{name}.pdf"
        print(f"{name}")
        if dest.exists() or fetch_pdf(url, dest):
            title, pages = describe(dest)
            title = TITLE_OVERRIDES.get(dest.name, title)
            rows.append({"fichier": dest.name, "url": url, "titre": title, "pages": pages})
            print(f"  OK  {pages} p. | {title[:80]}")
        else:
            failed.append(name)
        time.sleep(1)  # politesse envers les serveurs

    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fichier", "url", "titre", "pages"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} documents OK, {len(failed)} échecs : {failed}")


if __name__ == "__main__":
    main()
