"""Recherche lexicale BM25, codée à la main (système S1 et brique du système hybride S3).

BM25 note un chunk d pour une requête q ainsi :
    score(q, d) = somme sur les mots t de q de
        idf(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )
    idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
- tf(t,d)  : nombre d'occurrences du mot t dans d (saturé par k1 : la 10e occurrence compte peu) ;
- idf(t)   : un mot rare est plus informatif qu'un mot courant ;
- |d|/avgdl: normalisation par la longueur (b = 0 la désactive, b = 1 la rend totale).
"""
import re
import unicodedata

import numpy as np
from scipy import sparse

STOPWORDS = set("""a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me meme
mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre
vous c d j l m n s t y est sont etre ete ont avoir peut doit sans sous si ainsi alors aussi car donc lors""".split())


def tokenize(text: str) -> list[str]:
    """Minuscules, accents supprimés (« machine »/« machiné » se rejoignent), mots vides retirés."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        self.tf = sparse.csc_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float32)
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean())
        df = np.diff(self.tf.indptr)                       # nb de documents contenant chaque mot
        n = len(docs)
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1).astype(np.float32)
        self.norm = k1 * (1 - b + b * self.dl / self.avgdl)  # terme de longueur, un par document

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.dl), dtype=np.float32)
        for t in set(tokenize(query)):
            j = self.vocab.get(t)
            if j is None:
                continue
            col = self.tf.getcol(j)
            idx, tf = col.indices, col.data
            s[idx] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.norm[idx])
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
