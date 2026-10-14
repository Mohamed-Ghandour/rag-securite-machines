"""Génération des réponses (S0 à S3) et évaluation : F1, abstention, juge LLM.

Modèles (Groq, open-weights) :
  - générateur : openai/gpt-oss-20b  (petit, rapide, quota propre)
  - juge       : qwen/qwen3.8-27b    (autre famille que le générateur, pour limiter le biais d'auto-préférence)
"""
import json
import re
import unicodedata
from collections import Counter

from src import llm
from src.testset import parse_json

GENERATOR = "openai/gpt-oss-20b"
JUDGE = "qwen/qwen3.8-27b"
ABSTAIN = "Information absente des documents."

PROMPT_RAG_SIMPLE = ("Tu es un assistant expert en sécurité des machines. Réponds à la question en français, "
                     "de façon concise, en t'aidant des extraits fournis.")
PROMPT_RAG_CITE = ("Tu es un assistant expert en sécurité des machines. Réponds en français, de façon concise (1 à 3 phrases), "
                   "UNIQUEMENT à partir des extraits fournis. Cite les extraits utilisés entre crochets, par exemple [1] ou [2]. "
                   f"Si les extraits ne contiennent pas la réponse, réponds exactement : « {ABSTAIN} » sans rien ajouter.")
PROMPT_CLOSED = ("Tu es un assistant expert en sécurité des machines industrielles. Réponds de mémoire, en français, de façon "
                 "concise (1 à 3 phrases). Si tu n'es pas certain de la réponse, réponds exactement : « Je ne sais pas. »")


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{n}] ({c['doc']}, p.{c['page_start']})\n{c['text']}" for n, c in enumerate(chunks, 1))


def answer(question: str, chunks: list[dict] | None, prompt: str = PROMPT_RAG_CITE, model: str = GENERATOR) -> dict:
    if chunks is None:                                   # S0 : sans documents
        msgs = [{"role": "system", "content": PROMPT_CLOSED}, {"role": "user", "content": question}]
    else:
        msgs = [{"role": "system", "content": prompt},
                {"role": "user", "content": f"Extraits :\n{format_context(chunks)}\n\nQuestion : {question}"}]
    r = llm.chat(msgs, model=model, temperature=0.0, max_tokens=600, reasoning_effort="low")
    return {k: r[k] for k in ("text", "prompt_tokens", "completion_tokens", "latency_s", "cached")}


# ---------------------------------------------------------------- métriques automatiques

def _tok(t: str) -> list[str]:
    t = unicodedata.normalize("NFKD", t.lower().replace("’", "'"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", " ", t))


def f1(pred: str, ref: str) -> float:
    p, r = _tok(pred), _tok(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(r)
    return 2 * pr * rc / (pr + rc)


def is_abstention(text: str) -> bool:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return bool(re.search(r"information absente|je ne sais pas|ne (?:permettent|contiennent|figure|mentionnent)", t))


def cited_ids(text: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)})


def citation_ok(text: str, n_chunks: int) -> bool | None:
    """Exactitude formelle des citations : toutes les références existent. None si aucune citation."""
    ids = cited_ids(text)
    return None if not ids else all(1 <= i <= n_chunks for i in ids)


# ---------------------------------------------------------------- juge LLM

JUDGE_SYSTEM = "Tu es un évaluateur rigoureux et impartial. Tu réponds uniquement par un objet JSON."


def judge(question: str, reference: str | None, pred: str, chunks: list[dict] | None) -> dict:
    """Notes du juge. correct : 1 / 0.5 / 0 par rapport à la référence ;
    fidele : 1 / 0 (toutes les affirmations sont soutenues par les extraits) ; None si pas d'extraits."""
    ctx = f"\nEXTRAITS FOURNIS AU SYSTÈME :\n{format_context(chunks)}\n" if chunks else ""
    ask_fid = ('- "fidele" : 1 si toutes les affirmations de la réponse sont soutenues par les extraits fournis, 0 si elle '
               "contient une information absente des extraits (hallucination). Une abstention est fidèle (1).\n"
               if chunks else "")
    p = (f"QUESTION : {question}\nRÉPONSE DE RÉFÉRENCE : {reference}\n{ctx}\nRÉPONSE À ÉVALUER : {pred}\n\n"
         'Évalue : \n- "correct" : 1 si la réponse est équivalente à la référence (mêmes faits clés, valeurs exactes), '
         "0.5 si partiellement correcte ou incomplète, 0 si fausse ou si elle s'abstient.\n"
         f"{ask_fid}"
         'Réponds par : {"correct": ..., "fidele": ..., "raison": "une phrase"}')
    r = llm.chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": p}],
                 model=JUDGE, temperature=0.0, max_tokens=500, json_mode=True)
    j = parse_json(r["text"]) or {}
    return {"correct": j.get("correct"), "fidele": j.get("fidele") if chunks else None,
            "raison": j.get("raison", ""), "judge_tokens": r["prompt_tokens"] + r["completion_tokens"]}
