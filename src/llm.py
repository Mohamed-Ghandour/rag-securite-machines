"""Client LLM (API Groq) avec cache disque, reprise sur limites de débit et mesure des coûts.

Le cache est la clé de la reproductibilité : une même requête (modèle, messages,
paramètres) n'est jamais payée deux fois, et relancer le notebook redonne les
mêmes réponses. Chaque appel renvoie aussi le nombre de tokens et la latence,
utilisés plus tard pour l'analyse des coûts.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "processed" / "llm_cache"
MAX_WAIT = 90          # au-delà (limite journalière atteinte), on s'arrête au lieu de dormir des heures

load_dotenv(ROOT / ".env")
_client = None


def client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY absente : la renseigner dans .env")
        _client = Groq(api_key=key, max_retries=0)      # les reprises sont gérées ici
    return _client


def _key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"m": model, "msg": messages, "p": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chat(messages: list[dict], model: str, temperature: float = 0.0, max_tokens: int = 1024,
         json_mode: bool = False, reasoning_effort: str | None = None, seed: int = 0) -> dict:
    """Un appel de chat. Renvoie {text, prompt_tokens, completion_tokens, latency_s, cached, model}."""
    params = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode,
              "reasoning_effort": reasoning_effort, "seed": seed}
    path = CACHE_DIR / f"{_key(model, messages, params)}.json"
    if path.exists():
        return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}

    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_completion_tokens=max_tokens, seed=seed)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    for attempt in range(8):
        t0 = time.time()
        try:
            r = client().chat.completions.create(**kwargs)
            break
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 2 ** attempt * 2))
            if wait > MAX_WAIT:
                raise RuntimeError(f"Limite de débit atteinte, attente demandée {wait:.0f}s : {e}") from e
            time.sleep(wait + 1)
        except (APIConnectionError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code < 500:
                raise
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("Échec après plusieurs tentatives")

    out = {"text": (r.choices[0].message.content or "").strip(),
           "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
           "latency_s": round(time.time() - t0, 3), "model": model}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {**out, "cached": False}
