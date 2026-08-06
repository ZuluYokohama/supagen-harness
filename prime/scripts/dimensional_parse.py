"""
Dimensional document parse — Grok-side + local-fiber handoff substrate.

Best dimensional form for "transcribe into reality":
  NOT raw PDF dump into LFM context.
  YES: sectioned language stalks + embedding vectors + retrieval scores.

Pipeline:
  1. extract text (PDF/md)
  2. chunk into ~800–1500 char semantic units (page/section aware)
  3. embed each chunk (jina Job1 / nomic fallback)
  4. store index: {id, text, vec, page, tags}
  5. retrieve(query) → top-k chunks by cosine (language + vector glue)
  6. pack for LFM: compact "DIMENSIONAL PACK" with scores + text
  7. pack for Grok: JSON index + summary + work items

Grok = parse / plan / code / audit authority
LFM  = role ops on retrieved packs only
Embed = metric between intent, chunks, and role outputs
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lm_studio_client import LMStudio, cosine


@dataclass
class Chunk:
    id: str
    text: str
    page: int | None
    kind: str  # section|page|para
    title: str
    n_chars: int
    embed: list[float] = field(default_factory=list)
    embed_dim: int = 0
    tags: list[str] = field(default_factory=list)


def _norm_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.replace("\r", "")).strip()


def chunk_document(text: str, target_chars: int = 1200) -> list[Chunk]:
    """Split on page markers and large paragraphs; keep section titles."""
    chunks: list[Chunk] = []
    pages = re.split(r"(?m)^--- page (\d+) ---\s*", text)
    # re.split keeps delimiters: ['', '1', body, '2', body, ...]
    bodies: list[tuple[int | None, str]] = []
    if len(pages) >= 3:
        i = 1
        while i + 1 < len(pages):
            try:
                pn = int(pages[i])
            except ValueError:
                pn = None
            bodies.append((pn, pages[i + 1]))
            i += 2
    else:
        bodies = [(None, text)]

    cid = 0
    for page, body in bodies:
        body = body.strip()
        if not body:
            continue
        # try section-ish splits
        parts = re.split(r"(?m)(?=^\s*\d+\.(?:\d+\.)?\s+[A-Z])", body)
        if len(parts) <= 1:
            parts = [body]
        for part in parts:
            part = _norm_ws(part)
            if len(part) < 80:
                continue
            # further window if huge
            for start in range(0, len(part), target_chars):
                window = part[start : start + target_chars]
                if len(window) < 60:
                    continue
                title_m = re.match(r"^(\d+\.(?:\d+\.)?\s+[^\n.]{5,80})", window)
                md_h = re.search(r"(?m)^#{1,3}\s+(.+)$", window[:200])
                if title_m:
                    title = title_m.group(1)
                elif md_h:
                    title = md_h.group(1).strip()
                else:
                    # first non-code non-fence line
                    title = "untitled"
                    for line in window.splitlines():
                        ln = line.strip()
                        if not ln or ln.startswith("```") or ln.startswith("---"):
                            continue
                        title = ln[:80]
                        break
                tags = []
                low = window.lower()
                for tag, keys in (
                    ("geometry", ("manifold", "geodesic", "riemann", "helical", "shape")),
                    ("efficiency", ("order of magnitude", "10^", "compute", "efficiency", "sparse")),
                    ("active_state", ("active inference", "cache", "predictive", "free energy")),
                    ("liquid", ("liquid neural", "lnn", "liquid ai")),
                    ("neuromorphic", ("spiking", "loihi", "neuromorphic", "snn")),
                    ("commercial", ("verses", "genius", "numenta", "thousand brain")),
                    ("physics", ("landauer", "symplectic", "thermodynamic", "entropy")),
                    ("metric", ("aboutness", "agreement", "nli", "jina", "dual_enter", "cosine")),
                    ("law", ("open", "stop", "residue", "restrict", "audit", "measure")),
                    ("lms", ("lm studio", "lfm", "context_length", "gguf")),
                ):
                    if any(k in low for k in keys):
                        tags.append(tag)
                cid += 1
                chunks.append(
                    Chunk(
                        id=f"C{cid:04d}",
                        text=window,
                        page=page,
                        kind="section" if title_m else "window",
                        title=_norm_ws(title)[:100],
                        n_chars=len(window),
                        tags=tags,
                    )
                )
    return chunks


def embed_chunks(
    chunks: list[Chunk],
    lm: LMStudio | None = None,
    model: str | None = None,
    max_chunks: int = 80,
) -> list[Chunk]:
    """Embed documents with Job1 family prefixes (jina Document: / nomic search_document:)."""
    from nomic_metric import default_embed_model, embed_batch

    model = model or default_embed_model()
    subset = chunks[:max_chunks]
    payloads = [(ch.title + "\n" + ch.text)[:2000] for ch in subset]
    # batch in groups of 16
    for start in range(0, len(payloads), 16):
        batch = payloads[start : start + 16]
        r = embed_batch(batch, task="search_document", model=model)
        embs = r.get("embeddings") or []
        for j, ch in enumerate(subset[start : start + 16]):
            vec = embs[j] if j < len(embs) else None
            if vec:
                ch.embed = vec
                ch.embed_dim = len(vec)
    return subset


def build_index(text: str, embed: bool = True, max_chunks: int = 80) -> dict[str, Any]:
    chunks = chunk_document(text)
    if embed:
        chunks = embed_chunks(chunks, max_chunks=max_chunks)
    else:
        chunks = chunks[:max_chunks]
    # store vectors separately optional — include in index for retrieval
    fam = None
    try:
        from nomic_metric import resolve_family

        fam = resolve_family()
    except Exception:
        fam = None
    return {
        "ok": True,
        "n_chunks": len(chunks),
        "embedded": sum(1 for c in chunks if c.embed),
        "dim": next((c.embed_dim for c in chunks if c.embed_dim), 0),
        "embed_family": fam,
        "chunks": [
            {
                "id": c.id,
                "title": c.title,
                "page": c.page,
                "kind": c.kind,
                "n_chars": c.n_chars,
                "tags": c.tags,
                "text": c.text,
                "embed": c.embed,  # full vector for retrieval
            }
            for c in chunks
        ],
        "built_at": time.time(),
    }


def _token_set(s: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z0-9_]{3,}", (s or "").lower())}


def _lexical_overlap(query: str, title: str, text: str, tags: list | None) -> float:
    q = _token_set(query)
    if not q:
        return 0.0
    doc = _token_set(title) | _token_set((text or "")[:800])
    for t in tags or []:
        doc |= _token_set(str(t))
    if not doc:
        return 0.0
    inter = len(q & doc)
    return inter / max(len(q), 1)


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-Python BM25 (no deps) for hybrid retrieve."""
    q_terms = list(_token_set(query))
    if not q_terms or not docs:
        return [0.0] * len(docs)
    tokenized = [list(_token_set(d)) for d in docs]
    N = len(tokenized)
    avgdl = sum(len(t) for t in tokenized) / max(N, 1)
    df: dict[str, int] = {}
    for toks in tokenized:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    scores = []
    for toks in tokenized:
        tf: dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        dl = len(toks) or 1
        for term in q_terms:
            n_q = df.get(term, 0)
            if n_q == 0:
                continue
            idf = max(0.0, __import__("math").log(1 + (N - n_q + 0.5) / (n_q + 0.5)))
            f = tf.get(term, 0)
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    # normalize 0..1
    mx = max(scores) if scores else 0.0
    if mx <= 1e-12:
        return [0.0] * len(scores)
    return [s / mx for s in scores]


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    cos_weight: float = 0.50,
    lex_weight: float = 0.25,
    bm25_weight: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Job1.5 — hybrid re-rank (no extra GGUF): cosine × lex gate + BM25.

    Cosine alone ranks FILL/boilerplate high. Multiplicative lex gate + BM25
    pulls domain hits up. Still aboutness, not NLI agreement.
    """
    FILL_PENALTY = (
        "task is ongoing",
        "monitoring",
        "smooth execution",
        "further analysis",
    )
    docs = [
        f"{h.get('title') or ''} {h.get('text') or ''} {' '.join(h.get('tags') or [])}"
        for h in hits
    ]
    bm25 = _bm25_scores(query, docs)
    ranked = []
    for i, h in enumerate(hits):
        cos = float(h.get("score") or h.get("score_cos") or 0.0)
        title = str(h.get("title") or "")
        text = str(h.get("text") or "")
        lex = _lexical_overlap(query, title, text, h.get("tags"))
        b25 = bm25[i] if i < len(bm25) else 0.0
        lex_floor = 0.20
        gated = cos * (lex_floor + (1.0 - lex_floor) * max(lex, b25 * 0.5))
        score = (
            0.70 * gated
            + cos_weight * 0.15 * cos
            + lex_weight * lex
            + bm25_weight * b25
        )
        blob = (title + " " + text[:120]).lower()
        if any(p in blob for p in FILL_PENALTY) and lex < 0.15 and b25 < 0.2:
            score *= 0.50
        hh = dict(h)
        hh["score_cos"] = round(cos, 4)
        hh["score_lex"] = round(lex, 4)
        hh["score_bm25"] = round(b25, 4)
        hh["score"] = round(float(score), 4)
        hh["score_kind"] = "aboutness_hybrid"
        hh["method"] = "cosine*lex+bm25"
        ranked.append(hh)
    ranked.sort(key=lambda x: -float(x.get("score") or 0))
    return ranked


def ensure_index_family(
    index: dict[str, Any],
    *,
    auto_reembed: bool = True,
    max_auto: int = 120,
) -> dict[str, Any]:
    """
    If index was built under another Job1 family (or untagged nomic-era),
    re-embed chunks under live family so cosine is not silently wrong.
    """
    try:
        from nomic_metric import embed_batch, resolve_family
    except Exception as e:
        return {"ok": False, "error": str(e), "reembedded": False}

    live = resolve_family()
    stored = index.get("embed_family")
    chunks = index.get("chunks") or []
    n = len(chunks)
    # detect likely mismatch: no family stamp, or different family
    mismatch = (stored is None and n > 0) or (stored and stored != live)
    if not mismatch:
        return {
            "ok": True,
            "reembedded": False,
            "family": live,
            "stored": stored,
            "n": n,
        }
    if not auto_reembed:
        return {
            "ok": False,
            "reembedded": False,
            "family": live,
            "stored": stored,
            "n": n,
            "error": "family_mismatch",
            "hint": "supagen reindex-kb  OR retrieve will auto-reembed if n<=max_auto",
        }
    if n > max_auto:
        return {
            "ok": False,
            "reembedded": False,
            "family": live,
            "stored": stored,
            "n": n,
            "error": "family_mismatch_too_large",
            "hint": f"n={n}>{max_auto}; run: python -m supagen reindex-kb",
        }
    # re-embed all chunks as search_document
    payloads = []
    for c in chunks:
        title = c.get("title") or ""
        text = c.get("text") or ""
        payloads.append((title + "\n" + text)[:2000])
    for start in range(0, len(payloads), 16):
        batch = payloads[start : start + 16]
        r = embed_batch(batch, task="search_document")
        embs = r.get("embeddings") or []
        for j, c in enumerate(chunks[start : start + 16]):
            vec = embs[j] if j < len(embs) else None
            if vec:
                c["embed"] = vec
                c["embed_dim"] = len(vec)
    index["embed_family"] = live
    index["dim"] = next(
        (len(c.get("embed") or []) for c in chunks if c.get("embed")), 0
    )
    index["embedded"] = sum(1 for c in chunks if c.get("embed"))
    index["reembedded_at"] = __import__("time").time()
    # Persist ONLY if index declares a path (never clobber default KB from unit tests)
    persist = index.get("path") or index.get("_persist_path")
    if persist:
        try:
            save_index(index, Path(persist))
        except Exception:
            pass
    return {
        "ok": True,
        "reembedded": True,
        "family": live,
        "stored_was": stored,
        "n": n,
        "dim": index.get("dim"),
    }


def retrieve(
    index: dict[str, Any],
    query: str,
    k: int = 4,
    lm: LMStudio | None = None,
    rerank: bool = True,
    fix_family: bool = True,
) -> list[dict[str, Any]]:
    lm = lm or LMStudio()
    family_fix: dict[str, Any] | None = None
    if fix_family:
        family_fix = ensure_index_family(index, auto_reembed=True)
        if family_fix.get("reembedded"):
            index["_family_fix"] = family_fix
    # Job 1: query prefix for retrieval (not agreement) — jina via nomic_metric
    q = lm.embed(query[:2000], task="search_query")
    if not q.get("ok") or not q.get("embedding"):
        # fallback keyword
        ql = query.lower().split()
        scored = []
        for c in index.get("chunks") or []:
            t = (c.get("text") or "").lower()
            score = sum(1 for w in ql if len(w) > 3 and w in t)
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        raw = [
            {**c, "score": float(s), "method": "keyword", "score_kind": "keyword"}
            for s, c in scored[: max(k * 3, 8)]
            if s > 0
        ] or [
            {**c, "score": 0.0, "method": "keyword", "score_kind": "keyword"}
            for _, c in scored[:k]
        ]
        return rerank_hits(query, raw)[:k] if rerank else raw[:k]

    qv = q["embedding"]
    # Matryoshka: coarse ANN at 256-d, re-score top candidates at full dim
    from nomic_metric import matryoshka

    coarse_k = max(k * 6, 16)
    scored_coarse = []
    for c in index.get("chunks") or []:
        ev = c.get("embed") or []
        if not ev:
            continue
        # family mismatch: dim may differ — skip incompatible vectors
        if len(ev) != len(qv) and not (len(ev) >= 256 and len(qv) >= 256):
            continue
        if len(ev) >= 256 and len(qv) >= 256:
            s = cosine(matryoshka(qv, 256), matryoshka(ev, 256))
        else:
            s = cosine(qv, ev)
        scored_coarse.append((s, c))
    scored_coarse.sort(key=lambda x: -x[0])
    # re-score shortlist at full dimension
    scored = []
    for _, c in scored_coarse[:coarse_k]:
        ev = c.get("embed") or []
        if len(ev) != len(qv):
            continue
        scored.append((cosine(qv, ev), c))
    scored.sort(key=lambda x: -x[0])
    hits = []
    for s, c in scored[: max(k * 3, 8)]:
        hit = {kk: vv for kk, vv in c.items() if kk != "embed"}
        hit["score"] = round(float(s), 4)
        hit["score_kind"] = "aboutness"
        hit["matryoshka"] = "256_then_full"
        hit["method"] = "cosine"
        hit["embed_family"] = q.get("family") or index.get("embed_family")
        hits.append(hit)
    if rerank:
        hits = rerank_hits(query, hits)
        # Job1.5 neural rerank (asymmetric) — still aboutness, not NLI agreement
        try:
            if __import__("os").environ.get("PRIME_RERANK", "1").strip() not in (
                "0",
                "false",
                "no",
            ):
                from rerank_service import rerank_hits as neural_rerank_hits

                hits = neural_rerank_hits(query, hits)
        except Exception as e:
            if hits:
                hits[0] = dict(hits[0])
                hits[0]["neural_rerank_error"] = str(e)[:200]
    return hits[:k]


def pack_for_lfm(query: str, hits: list[dict[str, Any]], max_chars: int | None = None) -> str:
    """Best dimensional handoff to small LFM: ranked packs, not full doc.

    max_chars defaults from ~/.lmstudio log-derived policy (context thrash on 4096).
    """
    if max_chars is None:
        try:
            from lms_home import derived_policy

            max_chars = int(derived_policy().get("pack_budget_chars") or 2800)
        except Exception:
            max_chars = 2800
    try:
        from metric_text import pack_to_token_budget

        use_tok_cap = True
    except Exception:
        use_tok_cap = False
    lines = [
        "DIMENSIONAL PACK (retrieved stalks; aboutness scores — NOT agreement/NLI)",
        f"QUERY: {query[:300]}",
        "",
    ]
    used = 0
    for i, h in enumerate(hits, 1):
        block = (
            f"[{i}] id={h.get('id')} page={h.get('page')} score={h.get('score')} "
            f"tags={','.join(h.get('tags') or [])}\n"
            f"TITLE: {h.get('title')}\n"
            f"{h.get('text')}\n"
        )
        if used + len(block) > max_chars:
            break
        # also respect token budget when available
        if use_tok_cap:
            trial = "\n".join(lines + [block])
            from metric_text import estimate_tokens

            if estimate_tokens(trial) > max(800, max_chars // 3):
                break
        lines.append(block)
        used += len(block)
    lines.append(
        "\nUse ONLY this pack + design law. Prefer evidence from high score chunks. "
        "If pack insufficient → NEED_INFO/STOP, never invent."
    )
    return "\n".join(lines)


def pack_for_grok(index: dict[str, Any], hits: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """Premium handoff for cloud orchestrator: compact, structured, no giant vectors in chat."""
    return {
        "query": query,
        "n_chunks_index": index.get("n_chunks"),
        "embedded": index.get("embedded"),
        "dim": index.get("dim"),
        "retrieval": [
            {
                "id": h.get("id"),
                "score": h.get("score"),
                "page": h.get("page"),
                "title": h.get("title"),
                "tags": h.get("tags"),
                "text_preview": (h.get("text") or "")[:400],
            }
            for h in hits
        ],
        "instruction": (
            "Grok: use retrieval previews + workspace tools to operationalize. "
            "LFM already gets full pack text via pack_for_lfm. "
            "OPEN only after measures; report claims ≠ production OPEN."
        ),
    }


def save_index(index: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # write full index (with vectors) for machine retrieval
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    # human-readable map without vectors
    slim = {
        **{k: v for k, v in index.items() if k != "chunks"},
        "chunks": [
            {kk: vv for kk, vv in c.items() if kk != "embed"}
            for c in index.get("chunks") or []
        ],
    }
    path.with_suffix(".slim.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return path


def load_index(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # stamp so ensure_index_family can persist without clobbering from unit tests
    if isinstance(data, dict):
        data["_persist_path"] = str(Path(path).resolve())
        data.setdefault("path", str(Path(path).resolve()))
    return data
