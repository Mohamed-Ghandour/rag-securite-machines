"""Recherche dense (système S2) : embeddings bge-m3 + index FAISS, avec cache disque.

Chaque chunk et chaque question devient un vecteur normalisé ; la similarité cosinus
(= produit scalaire sur vecteurs normalisés) mesure la proximité de sens, même sans
mot en commun. Index FAISS « plat » (IndexFlatIP) : recherche exacte, suffisante
pour quelques milliers de chunks.
"""
import gc
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "processed" / "embeddings"
MODEL_NAME = "BAAI/bge-m3"
BATCH = 16            # raisonnable pour 6 Go de VRAM avec des chunks <= 1024 tokens en fp16

_model = None


def model() -> SentenceTransformer:
    global _model
    if _model is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=dev)
        _model.max_seq_length = 1100
        if dev == "cuda":
            _model.half()
    return _model


def free_gpu() -> None:
    global _model
    _model = None
    gc.collect()
    torch.cuda.empty_cache()


def encode(texts: list[str], batch: int = BATCH) -> np.ndarray:
    # tri par longueur : moins de padding, donc plus rapide et moins de VRAM
    order = np.argsort([-len(t) for t in texts])
    emb = model().encode([texts[i] for i in order], batch_size=batch, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    out = np.empty_like(emb)
    out[order] = emb
    return out


class DenseIndex:
    def __init__(self, chunks: list[dict], name: str):
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        path = EMB_DIR / f"{MODEL_NAME.split('/')[-1]}_{name}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = encode([c["context"] + "\n" + c["text"] for c in chunks])
            np.save(path, self.emb)
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def search_emb(self, q_emb: np.ndarray, k: int = 10):
        scores, idx = self.index.search(q_emb, k)
        return idx, scores

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        idx, sc = self.search_emb(encode([query]), k)
        return list(zip(idx[0].tolist(), sc[0].tolist()))
