#!/usr/bin/env python3
"""
Job 1.5 — neural rerank (asymmetric query↔doc). Not agreement / not OPEN.

Default ladder (first that loads):
  1. jinaai/jina-reranker-v3          (SOTA listwise-ish; 0.6B)
  2. BAAI/bge-reranker-v2-m3
  3. cross-encoder/ms-marco-MiniLM-L-6-v2  (tiny CPU fallback)

Env
---
  PRIME_RERANK_MODEL   force HF id
  PRIME_RERANK_TOP     max docs per call (default 32)
  PRIME_RERANK         1 to enable in retrieve() (default 1)

Cosine still aboutness. NLI still owns agreement.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_ENGINE: dict[str, Any] | None = None
_FAIL_TS: float = 0.0
_FAIL_BACKOFF_S = float(os.environ.get("PRIME_RERANK_FAIL_BACKOFF_S", "60"))

# Only allow trust_remote_code for these exact Hub IDs.
# Revision MUST be a full 40-char git commit SHA (immutable). Tags/branches/short
# SHAs are refused. Empty rev refuses the jina AutoModel path → CrossEncoder.
def _immutable_rev(raw: str | None) -> str:
    rev = (raw or "").strip()
    if len(rev) == 40 and all(c in "0123456789abcdef" for c in rev.lower()):
        return rev.lower()
    return ""


TRUSTED_REMOTE = {
    "jinaai/jina-reranker-v3": _immutable_rev(os.environ.get("PRIME_JINA_RERANK_REV")),
}

DEFAULT_MODELS = [
    "jinaai/jina-reranker-v3",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


def _forced_model() -> str | None:
    m = (os.environ.get("PRIME_RERANK_MODEL") or "").strip()
    return m or None


def _load_engine() -> dict[str, Any]:
    """Lazy singleton. Returns {ok, kind, model, predict_fn, error?}."""
    global _ENGINE, _FAIL_TS
    with _LOCK:
        if _ENGINE is not None:
            return _ENGINE
        # Backoff after full-ladder failure — avoid reloading 3 models every request
        if _FAIL_TS and (time.time() - _FAIL_TS) < _FAIL_BACKOFF_S:
            return {
                "ok": False,
                "kind": None,
                "model": None,
                "predict": None,
                "error": f"rerank_load_backoff ({_FAIL_BACKOFF_S:.0f}s)",
                "backoff_s": _FAIL_BACKOFF_S,
            }

        candidates = []
        forced = _forced_model()
        if forced:
            candidates.append(forced)
        candidates.extend(DEFAULT_MODELS)

        last_err = ""
        for mid in candidates:
            # path A: jina AutoModel.rerank API (trust_remote only with pinned rev)
            if mid in TRUSTED_REMOTE or mid.startswith("jinaai/jina-reranker"):
                if mid not in TRUSTED_REMOTE and mid != "jinaai/jina-reranker-v3":
                    last_err = f"{mid}: not in TRUSTED_REMOTE allowlist"
                    continue
                rev = TRUSTED_REMOTE.get(mid) or (
                    _immutable_rev(os.environ.get("PRIME_JINA_RERANK_REV"))
                    if mid == "jinaai/jina-reranker-v3"
                    else ""
                )
                if not rev:
                    last_err = (
                        f"{mid}: refuse trust_remote_code without immutable "
                        "PRIME_JINA_RERANK_REV (full 40-char commit SHA)"
                    )
                    continue
                try:
                    from transformers import AutoModel
                    import torch

                    t0 = time.time()
                    kw: dict[str, Any] = {
                        "trust_remote_code": True,
                        "dtype": torch.float32,
                        "revision": rev,
                    }
                    model = AutoModel.from_pretrained(mid, **kw)
                    model.eval()

                    def _jina_predict(query: str, docs: list[str], _m=model):
                        # returns list of floats aligned to docs order
                        results = _m.rerank(query, docs, top_n=None)
                        # results sorted by score — re-align with exact index set
                        by_idx: dict[int, float] = {}
                        for r in results:
                            i = int(r["index"])
                            if i in by_idx:
                                raise ValueError(f"duplicate rerank index {i}")
                            by_idx[i] = float(r["relevance_score"])
                        expected = set(range(len(docs)))
                        got = set(by_idx.keys())
                        if got != expected:
                            missing = sorted(expected - got)
                            extra = sorted(got - expected)
                            raise ValueError(
                                f"rerank index set mismatch missing={missing} extra={extra}"
                            )
                        return [by_idx[i] for i in range(len(docs))]

                    _ENGINE = {
                        "ok": True,
                        "kind": "jina_rerank_api",
                        "model": mid,
                        "predict": _jina_predict,
                        "load_s": round(time.time() - t0, 1),
                        "revision": rev,
                    }
                    _FAIL_TS = 0.0
                    return _ENGINE
                except Exception as e:
                    last_err = f"{mid}: {e}"
                    continue

            # path B: CrossEncoder (bge / ms-marco)
            try:
                from sentence_transformers import CrossEncoder

                t0 = time.time()
                ce = CrossEncoder(mid)

                def _ce_predict(query: str, docs: list[str], _ce=ce):
                    pairs = [(query, d) for d in docs]
                    scores = _ce.predict(pairs)
                    return [float(s) for s in scores]

                _ENGINE = {
                    "ok": True,
                    "kind": "cross_encoder",
                    "model": mid,
                    "predict": _ce_predict,
                    "load_s": round(time.time() - t0, 1),
                }
                return _ENGINE
            except Exception as e:
                last_err = f"{mid}: {e}"
                continue

        # Soft-fail with backoff — do not thrash HF loads on every request
        fail = {
            "ok": False,
            "kind": None,
            "model": None,
            "predict": None,
            "error": last_err or "no_reranker_loaded",
        }
        _ENGINE = None
        _FAIL_TS = time.time()
        return fail


def rerank_status() -> dict[str, Any]:
    eng = _load_engine()
    return {
        "ok": eng.get("ok"),
        "kind": eng.get("kind"),
        "model": eng.get("model"),
        "load_s": eng.get("load_s"),
        "error": eng.get("error"),
        "enabled": os.environ.get("PRIME_RERANK", "1").strip() not in ("0", "false", "no"),
    }


def score_docs(query: str, documents: list[str]) -> dict[str, Any]:
    """Return per-doc scores; higher = more relevant to query."""
    docs = list(documents or [])
    if not query or not docs:
        return {
            "ok": True,
            "scores": [],
            "model": None,
            "job": "retrieval_rerank",
            "not_agreement": True,
        }
    eng = _load_engine()
    if not eng.get("ok") or not eng.get("predict"):
        return {
            "ok": False,
            "error": eng.get("error") or "reranker unavailable",
            "scores": [0.0] * len(docs),
            "job": "retrieval_rerank",
            "not_agreement": True,
        }
    try:
        top = int(os.environ.get("PRIME_RERANK_TOP", "32"))
        if top < 1:
            raise ValueError("PRIME_RERANK_TOP must be >= 1")
    except Exception as e:
        return {
            "ok": False,
            "error": f"invalid PRIME_RERANK_TOP: {e}",
            "scores": [0.0] * len(docs),
            "job": "retrieval_rerank",
            "not_agreement": True,
        }
    # jina listwise scores the full set in one forward pass — never chunk that path.
    # CrossEncoder may chunk large candidate sets.
    all_scores: list[float] = []
    t0 = time.time()
    try:
        kind = eng.get("kind") or ""
        if kind == "jina_rerank_api" or top <= 0 or top >= len(docs):
            all_scores = list(eng["predict"](query, docs))
        else:
            for i in range(0, len(docs), top):
                chunk = docs[i : i + top]
                all_scores.extend(eng["predict"](query, chunk))
    except Exception as e:
        return {
            "ok": False,
            "error": f"rerank inference failed: {e}",
            "scores": [0.0] * len(docs),
            "job": "retrieval_rerank",
            "not_agreement": True,
            "model": eng.get("model"),
        }
    return {
        "ok": True,
        "job": "retrieval_rerank",
        "model": eng.get("model"),
        "kind": eng.get("kind"),
        "scores": all_scores,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "not_agreement": True,
        "not_open_authority": True,
    }


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    text_key: str = "text",
    title_key: str = "title",
    blend: float = 0.65,
) -> list[dict[str, Any]]:
    """
    Re-order hits with neural score. blend: weight on neural vs prior score.
    score_final = blend * norm(neural) + (1-blend) * norm(prior)
    """
    if not hits or os.environ.get("PRIME_RERANK", "1").strip() in ("0", "false", "no"):
        return hits
    docs = []
    for h in hits:
        title = str(h.get(title_key) or "")
        text = str(h.get(text_key) or "")
        docs.append((title + "\n" + text).strip()[:2000] or text[:2000] or title)
    r = score_docs(query, docs)
    if not r.get("ok"):
        for h in hits:
            h["rerank_error"] = r.get("error")
        return hits
    scores = r.get("scores") or []
    if not scores:
        return hits

    def _norm(xs: list[float]) -> list[float]:
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-9:
            return [0.5] * len(xs)
        return [(x - lo) / (hi - lo) for x in xs]

    prior = [float(h.get("score") or 0.0) for h in hits]
    n_prior = _norm(prior)
    n_neur = _norm([float(s) for s in scores])
    out = []
    for i, h in enumerate(hits):
        hh = dict(h)
        ns = n_neur[i] if i < len(n_neur) else 0.0
        np_ = n_prior[i] if i < len(n_prior) else 0.0
        hh["score_rerank_raw"] = round(float(scores[i]), 6) if i < len(scores) else None
        hh["score_prior"] = hh.get("score")
        hh["score"] = round(blend * ns + (1.0 - blend) * np_, 4)
        hh["score_kind"] = "aboutness_hybrid_neural"
        hh["rerank_model"] = r.get("model")
        hh["method"] = (hh.get("method") or "cosine") + "+neural_rerank"
        out.append(hh)
    out.sort(key=lambda x: -float(x.get("score") or 0))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(rerank_status(), indent=2))
    q = "Ensure strict adherence to ownership guidelines."
    docs = [
        "Ensure strict adherence to ownership guidelines.",
        "attacks: violate ownership guidelines and strip rights.",
        "Carbonara uses guanciale, egg, pecorino, and black pepper.",
    ]
    print(json.dumps(score_docs(q, docs), indent=2))
