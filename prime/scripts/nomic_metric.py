"""
Job 1 — RETRIEVAL / ABOUTNESS chart only (not agreement).

Families
--------
jina  (default)  jina-embeddings-v5-text-nano-retrieval
                 Prefixes: Query: / Document:
                 Backend: PRIME_JINA_BASE (llama-server --embedding), default :8765
                 LMS packages this GGUF as type=llm; /v1/embeddings on :1234
                 remaps to nomic or 400s — do not trust LMS for jina.

nomic            text-embedding-nomic-embed-text-v1.5
                 Prefixes: search_query: / search_document: / …
                 Backend: LM Studio DEFAULT_BASE :1234

Env
---
  PRIME_EMBED_FAMILY   jina | nomic          (default: jina)
  PRIME_JINA_BASE      http://127.0.0.1:8765
  PRIME_JINA_MODEL     jina-embeddings-v5-text-nano-retrieval
  PRIME_EMBED          override model key (family inferred from name when possible)
  PRIME_EMBED_FALLBACK 1 (default) — if jina unreachable, fall back to nomic

Not Job 2 (agreement). Cosine has no contradiction channel.
Measured (2026-08-05): jina C-floor ~0.10, A−C ~0.83; nomic C-floor ~0.47, A−C ~0.45.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from lms_layers import DEFAULT_BASE, l0_post

Task = Literal["search_query", "search_document", "clustering", "classification", "none"]
Family = Literal["jina", "nomic"]

NOMIC_MODEL = "text-embedding-nomic-embed-text-v1.5"
JINA_MODEL = os.environ.get(
    "PRIME_JINA_MODEL", "jina-embeddings-v5-text-nano-retrieval"
)
JINA_BASE = os.environ.get("PRIME_JINA_BASE", "http://127.0.0.1:8765").rstrip("/")

PREFIX_NOMIC = {
    "search_query": "search_query: ",
    "search_document": "search_document: ",
    "clustering": "clustering: ",
    "classification": "classification: ",
    "none": "",
}
# Official jina-embeddings-v5 retrieval / text-matching prefixes
PREFIX_JINA = {
    "search_query": "Query: ",
    "search_document": "Document: ",
    "clustering": "Document: ",
    "classification": "Document: ",
    "none": "",
}

# Content-hash cache
_CACHE: dict[str, list[float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_HITS = 0
_CACHE_MISS = 0
_CACHE_MAX = 2048


def cache_stats() -> dict[str, int]:
    return {"hits": _CACHE_HITS, "miss": _CACHE_MISS, "size": len(_CACHE)}


def resolve_family(model: str | None = None, family: str | None = None) -> Family:
    if family in ("jina", "nomic"):
        return family  # type: ignore[return-value]
    m = (model or os.environ.get("PRIME_EMBED") or "").lower()
    if "jina" in m:
        return "jina"
    if "nomic" in m:
        return "nomic"
    env = (os.environ.get("PRIME_EMBED_FAMILY") or "jina").lower().strip()
    return "nomic" if env == "nomic" else "jina"


def default_embed_model(family: str | None = None) -> str:
    env_model = os.environ.get("PRIME_EMBED")
    fam = resolve_family(env_model, family)
    if env_model and resolve_family(env_model) == fam:
        return env_model
    return JINA_MODEL if fam == "jina" else NOMIC_MODEL


def default_embed_base(family: str | None = None, base: str | None = None) -> str:
    """
    Jina ALWAYS hits the side embed server (:8765), never LMS :1234.

    Passing LM Studio base into embed() used to override JINA_BASE → silent
    nomic remap → query vectors in nomic space vs jina KB docs → cos≈0.05.
    """
    fam = resolve_family(family=family)
    if fam == "jina":
        # only honor explicit override if it looks like the jina port/base
        if base and ("8765" in base or "jina" in base.lower()):
            return base.rstrip("/")
        return JINA_BASE.rstrip("/")
    if base and "8765" not in base:
        return base.rstrip("/")
    return DEFAULT_BASE.rstrip("/")


def prefixes_for(family: Family) -> dict[str, str]:
    return PREFIX_JINA if family == "jina" else PREFIX_NOMIC


def apply_prefix(
    text: str,
    task: Task = "search_document",
    family: str | None = None,
) -> str:
    fam = resolve_family(family=family)
    table = prefixes_for(fam)
    t = (text or "").strip()
    if not t:
        return t
    for p in list(PREFIX_NOMIC.values()) + list(PREFIX_JINA.values()):
        if p and t.startswith(p):
            return t
    return table.get(task, "") + t


def l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return v
    return [x / n for x in v]


def matryoshka(v: list[float], dims: int | None) -> list[float]:
    if not dims or dims >= len(v):
        return v
    return l2_normalize(v[:dims])


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def center(v: list[float], mean: list[float] | None) -> list[float]:
    if not mean or len(mean) != len(v):
        return v
    return l2_normalize([x - m for x, m in zip(v, mean)])


def _cache_key(prefixed: str, model: str, dims: int | None) -> str:
    return hashlib.sha256(f"{model}|{dims}|{prefixed}".encode("utf-8")).hexdigest()


def _allow_fallback() -> bool:
    return os.environ.get("PRIME_EMBED_FALLBACK", "1").strip() not in ("0", "false", "no")


def _post_embeddings(
    base: str,
    model: str,
    input_payload: str | list[str],
    timeout: float,
) -> Any:
    return l0_post(
        "/v1/embeddings",
        {"model": model, "input": input_payload},
        base=base,
        timeout=timeout,
    )


def _detect_silent_remap(requested_family: Family, response_model: str) -> str | None:
    rm = (response_model or "").lower()
    if requested_family == "jina" and "nomic" in rm:
        return (
            f"LMS silent remap: requested jina, response model={response_model!r}. "
            "Start llama-server --embedding on PRIME_JINA_BASE (default :8765)."
        )
    if requested_family == "nomic" and "jina" in rm and "nomic" not in rm:
        return f"Unexpected remap to jina: response model={response_model!r}"
    return None


def _embed_once(
    prefixed: str,
    *,
    family: Family,
    model: str,
    base: str,
    dims: int | None,
    mean: list[float] | None,
    timeout: float,
    use_cache: bool,
) -> dict[str, Any]:
    global _CACHE_HITS, _CACHE_MISS

    ck = _cache_key(prefixed, model, dims) if use_cache and mean is None else None
    if ck:
        with _CACHE_LOCK:
            if ck in _CACHE:
                _CACHE_HITS += 1
                return {
                    "ok": True,
                    "job": "retrieval_aboutness",
                    "not_agreement": True,
                    "task": None,
                    "family": family,
                    "prefix_applied": True,
                    "model": model,
                    "base": base,
                    "dim": len(_CACHE[ck]),
                    "embedding": list(_CACHE[ck]),
                    "cached": True,
                    "note": "ABOUTNESS only (cache hit).",
                }

    r = _post_embeddings(base, model, prefixed, timeout)
    if not r.ok:
        return {
            "ok": False,
            "job": "retrieval_aboutness",
            "family": family,
            "model": model,
            "base": base,
            "error": r.error,
            "latency_ms": round(r.latency_ms, 1),
        }

    resp_model = str(r.data.get("model") or "")
    remap = _detect_silent_remap(family, resp_model)
    if remap:
        return {
            "ok": False,
            "job": "retrieval_aboutness",
            "family": family,
            "model": model,
            "base": base,
            "error": remap,
            "response_model": resp_model,
            "latency_ms": round(r.latency_ms, 1),
        }

    vec = (r.data.get("data") or [{}])[0].get("embedding") or []
    if not vec:
        return {
            "ok": False,
            "job": "retrieval_aboutness",
            "family": family,
            "error": "empty_embedding",
            "latency_ms": round(r.latency_ms, 1),
        }
    if mean:
        vec = center(vec, mean)
    else:
        vec = l2_normalize(vec)
    full_dim = len(vec)
    if dims:
        vec = matryoshka(vec, dims)
    if ck:
        with _CACHE_LOCK:
            _CACHE_MISS += 1
            if len(_CACHE) >= _CACHE_MAX:
                for k in list(_CACHE.keys())[:256]:
                    del _CACHE[k]
            _CACHE[ck] = list(vec)
    return {
        "ok": True,
        "job": "retrieval_aboutness",
        "not_agreement": True,
        "family": family,
        "model": model,
        "base": base,
        "response_model": resp_model or model,
        "dim": len(vec),
        "full_dim": full_dim,
        "matryoshka_dims": dims,
        "centered": mean is not None,
        "embedding": vec,
        "cached": False,
        "latency_ms": round(r.latency_ms, 1),
        "note": "ABOUTNESS only. Do not use as claim agreement / OPEN glue.",
    }


def embed(
    text: str,
    task: Task = "search_document",
    model: str | None = None,
    base: str | None = None,
    dims: int | None = None,
    mean: list[float] | None = None,
    timeout: float = 60,
    use_cache: bool = True,
    family: str | None = None,
    ensure_service: bool = True,
) -> dict[str, Any]:
    """Embed with family-correct task prefix. Returns ABOUTNESS vector, not agreement.

    Seamless jina: if family=jina, auto-ensure side server on :8765 before POST.
    """
    fam = resolve_family(model, family)
    model_key = model or default_embed_model(fam)
    # If caller passed nomic model under jina family env, trust model name
    fam = resolve_family(model_key, family)
    prefixed = apply_prefix(text, task, family=fam)
    if not prefixed.strip():
        return {
            "ok": False,
            "job": "retrieval_aboutness",
            "error": "empty_text",
            "task": task,
            "family": fam,
        }

    jina_ensure: dict[str, Any] | None = None
    if fam == "jina" and ensure_service:
        try:
            from jina_service import ensure_jina

            jina_ensure = ensure_jina()
        except Exception as e:
            jina_ensure = {"ok": False, "error": f"ensure_jina exception: {e}"}

    embed_base = default_embed_base(fam, base)
    result = _embed_once(
        prefixed,
        family=fam,
        model=model_key,
        base=embed_base,
        dims=dims,
        mean=mean,
        timeout=timeout,
        use_cache=use_cache,
    )
    result["task"] = task
    result["prefix_applied"] = bool(prefixes_for(fam).get(task))
    result["prefixed_preview"] = prefixed[:80]
    if jina_ensure is not None:
        result["jina_service"] = {
            "ok": jina_ensure.get("ok"),
            "status": jina_ensure.get("status"),
            "started": jina_ensure.get("started"),
            "base": jina_ensure.get("base"),
        }

    # If ensure claimed ready but first POST failed, one retry after brief wait
    if (
        not result.get("ok")
        and fam == "jina"
        and jina_ensure
        and jina_ensure.get("ok")
    ):
        time.sleep(0.4)
        result = _embed_once(
            prefixed,
            family=fam,
            model=model_key,
            base=embed_base,
            dims=dims,
            mean=mean,
            timeout=timeout,
            use_cache=use_cache,
        )
        result["task"] = task
        result["prefix_applied"] = True
        result["retry_after_ensure"] = True

    # jina still down → optional nomic fallback (explicit warning)
    if (
        not result.get("ok")
        and fam == "jina"
        and _allow_fallback()
        and "nomic" not in (model_key or "").lower()
    ):
        fb = _embed_once(
            apply_prefix(text, task, family="nomic"),
            family="nomic",
            model=NOMIC_MODEL,
            base=(base or DEFAULT_BASE).rstrip("/"),
            dims=dims,
            mean=mean,
            timeout=timeout,
            use_cache=use_cache,
        )
        fb["task"] = task
        fb["prefix_applied"] = True
        fb["family_requested"] = "jina"
        fb["fallback_from"] = result.get("error")
        fb["jina_service"] = result.get("jina_service") or jina_ensure
        fb["warning"] = (
            "Jina aboutness unavailable after ensure; fell back to nomic "
            "(compressed floor). Check prime/state/jina_embed.log; "
            "python prime/scripts/jina_service.py ensure"
        )
        if fb.get("ok"):
            return fb
        result["fallback_error"] = fb.get("error")
    return result


def embed_batch(
    texts: list[str],
    task: Task = "search_document",
    model: str | None = None,
    base: str | None = None,
    dims: int | None = None,
    timeout: float = 120,
    family: str | None = None,
    ensure_service: bool = True,
) -> dict[str, Any]:
    """Batch embed — one HTTP call for many strings (CPU overhead win)."""
    global _CACHE_HITS, _CACHE_MISS
    if not texts:
        return {"ok": True, "embeddings": [], "n": 0}

    fam = resolve_family(model, family)
    model_key = model or default_embed_model(fam)
    fam = resolve_family(model_key, family)
    if fam == "jina" and ensure_service:
        try:
            from jina_service import ensure_jina

            ensure_jina()
        except Exception:
            pass
    embed_base = default_embed_base(fam, base)
    prefixed = [apply_prefix(t, task, family=fam) for t in texts]
    out: list[list[float] | None] = [None] * len(prefixed)
    need_idx: list[int] = []
    need_text: list[str] = []
    for i, p in enumerate(prefixed):
        ck = _cache_key(p, model_key, dims)
        with _CACHE_LOCK:
            if ck in _CACHE:
                _CACHE_HITS += 1
                out[i] = list(_CACHE[ck])
            else:
                need_idx.append(i)
                need_text.append(p)

    warning = None
    if need_text:
        r = _post_embeddings(embed_base, model_key, need_text, timeout)
        if not r.ok or _detect_silent_remap(fam, str((r.data or {}).get("model") or "")):
            # sequential fallback via embed() (handles jina→nomic)
            for j, orig_i in enumerate(need_idx):
                one = embed(
                    texts[orig_i],
                    task=task,
                    model=model_key,
                    base=base,
                    dims=dims,
                    timeout=timeout,
                    family=fam,
                    use_cache=True,
                )
                if one.get("ok"):
                    out[orig_i] = one["embedding"]
                    if one.get("warning"):
                        warning = one["warning"]
                else:
                    return {
                        "ok": False,
                        "error": one.get("error") or (r.error if not r.ok else "batch_fail"),
                        "embeddings": out,
                        "family": fam,
                    }
        else:
            data = r.data.get("data") or []
            by_i = {
                int(d.get("index", j)): d.get("embedding") or []
                for j, d in enumerate(data)
            }
            for j, orig_i in enumerate(need_idx):
                vec = by_i.get(j) or (data[j].get("embedding") if j < len(data) else []) or []
                vec = l2_normalize(vec)
                if dims:
                    vec = matryoshka(vec, dims)
                out[orig_i] = vec
                ck = _cache_key(need_text[j], model_key, dims)
                with _CACHE_LOCK:
                    _CACHE_MISS += 1
                    _CACHE[ck] = list(vec)

    return {
        "ok": all(v is not None for v in out),
        "job": "retrieval_aboutness_batch",
        "family": fam,
        "model": model_key,
        "base": embed_base,
        "n": len(out),
        "n_fetched": len(need_text),
        "n_cached": len(texts) - len(need_text),
        "embeddings": out,
        "cache": cache_stats(),
        "warning": warning,
    }


def aboutness(
    a: str,
    b: str,
    a_task: Task = "search_query",
    b_task: Task = "search_document",
    mean: list[float] | None = None,
    dims: int | None = None,
    base: str | None = None,
    model: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Cosine aboutness between two texts (retrieval metric)."""
    ea = embed(
        a, task=a_task, mean=mean, dims=dims, base=base, model=model, family=family
    )
    eb = embed(
        b, task=b_task, mean=mean, dims=dims, base=base, model=model, family=family
    )
    if not ea.get("ok") or not eb.get("ok"):
        return {
            "ok": False,
            "job": "retrieval_aboutness",
            "error": ea.get("error") or eb.get("error"),
            "family": ea.get("family") or eb.get("family"),
        }
    cos = cosine(ea["embedding"], eb["embedding"])
    return {
        "ok": True,
        "job": "retrieval_aboutness",
        "not_agreement": True,
        "cosine": round(cos, 4),
        "dim": ea.get("dim"),
        "centered": mean is not None,
        "a_task": a_task,
        "b_task": b_task,
        "family": ea.get("family"),
        "model": ea.get("model"),
        "base": ea.get("base"),
        "warning": ea.get("warning") or eb.get("warning"),
        "interpretation": (
            "high = same topic / aboutness; "
            "NOT entailment (contradictions can score high)"
        ),
    }


def corpus_mean_from_index(index_path: Path | str) -> list[float] | None:
    """Mean vector over embedded chunks for anisotropy correction."""
    path = Path(index_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    vecs = [c.get("embed") for c in (data.get("chunks") or []) if c.get("embed")]
    if not vecs:
        return None
    d = len(vecs[0])
    acc = [0.0] * d
    n = 0
    for v in vecs:
        if len(v) != d:
            continue
        for i, x in enumerate(v):
            acc[i] += x
        n += 1
    if n < 2:
        return None
    return [x / n for x in acc]


def null_calibrate(
    pairs: list[tuple[str, str]],
    mean: list[float] | None = None,
    base: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Empirical null: aboutness cosines for given pairs (related vs unrelated)."""
    scores = []
    for a, b in pairs:
        r = aboutness(a, b, mean=mean, base=base, family=family)
        if r.get("ok"):
            scores.append(r["cosine"])
    if not scores:
        return {"ok": False, "error": "no scores"}
    scores_s = sorted(scores)
    return {
        "ok": True,
        "n": len(scores),
        "min": round(scores_s[0], 4),
        "max": round(scores_s[-1], 4),
        "mean": round(sum(scores) / len(scores), 4),
        "median": round(scores_s[len(scores) // 2], 4),
        "family": resolve_family(family=family),
        "note": "Use unrelated null to set aboutness floors; never as agreement threshold",
    }


# Back-compat alias used by older imports expecting PREFIX
PREFIX = PREFIX_NOMIC
# Default model for callers that imported DEFAULT_EMBED from elsewhere for aboutness
DEFAULT_ABOUTNESS_MODEL = default_embed_model
