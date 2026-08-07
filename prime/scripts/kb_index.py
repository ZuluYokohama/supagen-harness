"""
KB dimensional index — multi-file corpus as one navigable manifold.

"Time travel" operationalized:
  past writings (PDF/md/txt) → Job1 aboutness stalks (jina default) → retrieve into present enter
  → LFM/Grok act under law → future OPEN only when measured

Not full 50k-file thrash: gated scan (extensions, max files, max chunks, size cap).
Design law: restrict → measure → audit → OPEN|STOP. Residue never forced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from dimensional_parse import (
    build_index,
    cosine,
    load_index,
    pack_for_grok,
    pack_for_lfm,
    retrieve,
    save_index,
)
from lm_studio_client import LMStudio


# Knowledge stalks — not machine state dumps (.json indices thrash embed time)
TEXT_EXT = {".txt", ".md", ".markdown", ".rst"}
CODE_EXT = {".py", ".toml", ".yaml", ".yml"}
DOC_EXT = {".pdf"} | TEXT_EXT  # default: human docs only

def _default_roots() -> list[Path]:
    try:
        from workspace import prime_root

        pr = prime_root()
    except Exception:
        pr = Path(__file__).resolve().parents[1]
    return [
        pr / "docs",
        pr / "state" / "deep" / "FINAL_BRIEF.md",
        pr / "state" / "deep" / "source.txt",
    ]


# Default roots — resolved at call time via _default_roots() for portable clones
DEFAULT_ROOTS = _default_roots()

# Skip machine artifacts even if under a knowledge root
SKIP_NAMES = {
    "dimensional_index.json",
    "dimensional_index.slim.json",
    "job.json",
    "session.json",
    "last_grok_pack.json",
    "last_retrieval.json",
    "manifold_index.json",
    "manifold_index.slim.json",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader

        r = PdfReader(str(path))
        parts = []
        for i, page in enumerate(r.pages):
            t = page.extract_text() or ""
            parts.append(f"--- page {i+1} ---\n{t}")
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace")


def scan_files(
    roots: list[Path],
    exts: set[str] | None = None,
    max_files: int = 40,
    max_bytes: int = 2_000_000,
    exclude_dirs: set[str] | None = None,
) -> list[Path]:
    exts = exts or DOC_EXT
    exclude_dirs = exclude_dirs or {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "bin", "extensions", "models", ".internal", "temp-downloads",
    }
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            if (
                root.suffix.lower() in exts
                and root.name not in SKIP_NAMES
                and root.stat().st_size <= max_bytes
            ):
                found.append(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in exclude_dirs for part in p.parts):
                continue
            if p.name in SKIP_NAMES:
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            found.append(p)
            if len(found) >= max_files:
                return found
    return found


def build_kb_index(
    roots: list[Path],
    out_path: Path,
    embed: bool = True,
    max_files: int = 24,
    max_chunks_per_file: int = 12,
    max_total_chunks: int = 96,
    query_probe: str = "",
    target_dim: int | None = None,
    quantize: str | None = None,
) -> dict[str, Any]:
    """Scan roots → per-file chunk/embed → merged index with source provenance.

    target_dim: optional Matryoshka-style prefix truncate (e.g. 512, 256) when
    the embed family supports ordered dimensions (jina-v5). Callers must record
    dim on the index and match query vectors to the same width.
    quantize: None | "sq8" — scalar int8 storage for embeds (aboutness only).
    """
    files = scan_files(roots, max_files=max_files)
    all_chunks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    dim = 0
    t0 = time.time()
    td = int(target_dim) if target_dim else None
    if td is not None and td < 32:
        td = None
    qmode = (quantize or "").strip().lower() or None
    if qmode and qmode not in ("sq8",):
        qmode = None

    print(
        f"kb_index: {len(files)} files → embed={embed} max_total={max_total_chunks}"
        f" target_dim={td} quantize={qmode}",
        flush=True,
    )
    for i, fp in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {fp.name} …", flush=True)
        try:
            text = load_text(fp)
        except Exception as e:
            sources.append({"path": str(fp), "ok": False, "error": str(e)[:200]})
            continue
        if len(text.strip()) < 80:
            sources.append({"path": str(fp), "ok": False, "error": "too_short"})
            continue
        # namespace chunk ids by file hash prefix
        idx = build_index(text, embed=embed, max_chunks=max_chunks_per_file)
        fhash = _sha(fp)[:12]
        n = 0
        for c in idx.get("chunks") or []:
            if len(all_chunks) >= max_total_chunks:
                break
            c = dict(c)
            c["id"] = f"{fhash}_{c.get('id')}"
            c["source"] = str(fp)
            c["source_sha12"] = fhash
            emb = c.get("embed")
            if emb and isinstance(emb, (list, tuple)):
                vec = list(emb)
                if td is not None and len(vec) > td:
                    vec = vec[:td]
                if qmode == "sq8" and vec:
                    # scalar min-max → int8; store scale for dequant on retrieve
                    lo = min(float(x) for x in vec)
                    hi = max(float(x) for x in vec)
                    scale = (hi - lo) / 255.0 if hi > lo else 1.0
                    c["embed"] = [int(round((float(x) - lo) / scale)) for x in vec]
                    c["embed_sq8"] = {"lo": lo, "scale": scale}
                    c["embed_dtype"] = "sq8"
                else:
                    c["embed"] = vec
                dim = max(dim, len(c["embed"]))
            all_chunks.append(c)
            n += 1
        sources.append({
            "path": str(fp),
            "ok": True,
            "n_chunks": n,
            "chars": len(text),
            "sha12": fhash,
        })
        print(f"    +{n} chunks (total {len(all_chunks)})", flush=True)
        if len(all_chunks) >= max_total_chunks:
            break

    index = {
        "ok": True,
        "kind": "kb_manifold",
        "n_chunks": len(all_chunks),
        "embedded": sum(1 for c in all_chunks if c.get("embed")),
        "dim": dim,
        "target_dim": td,
        "quantize": qmode,
        "chunks": all_chunks,
        "sources": sources,
        "roots": [str(r) for r in roots],
        "built_at": time.time(),
        "build_s": round(time.time() - t0, 2),
        "embed_family": None,
        "thesis": (
            "Past corpus projected into Job1 aboutness space (jina preferred). "
            "Retrieve into present. OPEN only under law. Cosine ≠ agreement."
        ),
        "law": "restrict→measure→audit→OPEN|STOP; residue never forced",
    }
    try:
        from nomic_metric import resolve_family

        index["embed_family"] = resolve_family()
    except Exception:
        pass
    save_index(index, out_path)

    probe: dict[str, Any] = {}
    if query_probe and index["embedded"]:
        hits = retrieve(index, query_probe, k=4)
        probe = {
            "query": query_probe,
            "hits": [
                {
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "source": h.get("source"),
                    "title": (h.get("title") or "")[:80],
                }
                for h in hits
            ],
            "lfm_pack_preview": pack_for_lfm(query_probe, hits)[:600],
        }

    return {
        "ok": True,
        "path": str(out_path),
        "slim_path": str(out_path.with_suffix(".slim.json")),
        "n_files": len([s for s in sources if s.get("ok")]),
        "n_chunks": index["n_chunks"],
        "embedded": index["embedded"],
        "dim": dim,
        "build_s": index["build_s"],
        "sources": sources,
        "probe": probe,
    }


def _dequant_index_embeds(index: dict[str, Any]) -> dict[str, Any]:
    """Materialize float embeds when index stored sq8 (aboutness-only)."""
    if index.get("quantize") != "sq8":
        return index
    out = dict(index)
    chunks = []
    for c in index.get("chunks") or []:
        c = dict(c)
        emb = c.get("embed")
        meta = c.get("embed_sq8") or {}
        if emb and meta and c.get("embed_dtype") == "sq8":
            lo = float(meta.get("lo") or 0.0)
            scale = float(meta.get("scale") or 1.0) or 1.0
            c["embed"] = [lo + int(x) * scale for x in emb]
            c["embed_dtype"] = "float32_dequant"
        chunks.append(c)
    out["chunks"] = chunks
    return out


def query_kb(
    index_path: Path,
    query: str,
    k: int = 5,
) -> dict[str, Any]:
    index = _dequant_index_embeds(load_index(index_path))
    # Match Matryoshka width on query path when index truncated
    hits = retrieve(index, query, k=k)
    td = index.get("target_dim") or index.get("dim")
    if td and hits:
        for h in hits:
            emb = h.get("embed")
            if emb and len(emb) > int(td):
                h["embed"] = emb[: int(td)]
    return {
        "ok": True,
        "query": query,
        "n_chunks_index": index.get("n_chunks"),
        "embedded": index.get("embedded"),
        "dim": index.get("dim"),
        "target_dim": index.get("target_dim"),
        "quantize": index.get("quantize"),
        "hits": hits,
        "lfm_pack": pack_for_lfm(query, hits),
        "grok_pack": pack_for_grok(index, hits, query),
        "time_travel": (
            "Retrieved stalks from prior corpus into present enter. "
            "Act with Grok/LFM; OPEN only after measures."
        ),
    }


def default_out() -> Path:
    return Path(__file__).resolve().parent.parent / "state" / "kb" / "manifold_index.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="KB dimensional manifold index")
    ap.add_argument("cmd", choices=["build", "query", "scan"], nargs="?", default="build")
    ap.add_argument("--root", action="append", default=[], help="Root path (repeatable)")
    ap.add_argument("--out", default=str(default_out()))
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--max-chunks-per-file", type=int, default=10)
    ap.add_argument("--max-total", type=int, default=80)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--query", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--target-dim",
        type=int,
        default=0,
        help="Matryoshka prefix truncate (e.g. 512); 0=full width",
    )
    ap.add_argument(
        "--quantize",
        default="",
        help="optional storage quant: sq8 (aboutness only; not agreement)",
    )
    args = ap.parse_args()

    roots = [Path(r) for r in args.root] if args.root else _default_roots()
    out = Path(args.out)

    if args.cmd == "scan":
        files = scan_files(roots, max_files=args.max_files)
        print(json.dumps({"n": len(files), "files": [str(f) for f in files]}, indent=2))
        return

    if args.cmd == "query":
        q = args.query or "manifold alignment geometry compute reduction OPEN STOP residue"
        r = query_kb(out, q, k=args.k)
        # drop full embeds from hit dump
        slim_hits = [{k: v for k, v in h.items() if k != "embed"} for h in r.get("hits") or []]
        print(json.dumps({**r, "hits": slim_hits, "lfm_pack": r["lfm_pack"][:1200]}, indent=2))
        return

    # build
    r = build_kb_index(
        roots,
        out,
        embed=not args.no_embed,
        max_files=args.max_files,
        max_chunks_per_file=args.max_chunks_per_file,
        target_dim=args.target_dim or None,
        quantize=args.quantize or None,
        max_total_chunks=args.max_total,
        query_probe=args.query
        or "geometry manifold LFM nomic OPEN STOP prime residual never forced",
    )
    print(json.dumps(r, indent=2)[:8000])


if __name__ == "__main__":
    main()
