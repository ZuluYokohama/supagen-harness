"""
Bilateral language projection — operational form.

Thesis: language is a *projection* of higher structure, from *both* sides:
  - Human side: natural language intent / claims / constraints
  - Domain side: code, geometry (E_ref), rplc certs, field packs, LM scout text

Neither side is the ground truth manifold. OPEN requires a section that
*glues* projections under restriction maps. Disagreement = obstruction → STOP
(residue never forced).

Sheaf reading (operational, not pure math claim):
  stalk_H  = human language features
  stalk_D  = domain language features
  ρ_HD     = restriction into shared interface vocabulary
  δ        = disagreement residual
  align    = 1 - normalized residual  (measure, not certificate of truth)
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Shared interface vocabulary (restriction target) — extend per domain
INTERFACE = {
    "open", "stop", "residue", "measure", "audit", "restrict", "certify",
    "strain", "holonomy", "sequence", "prior", "energy", "geometry",
    "smoke", "test", "verify", "claim", "force", "park", "defer",
    "plane", "pack", "ingest", "scout", "local", "model",
    "goal", "non_goal", "success", "constraint", "falsifier",
    "code", "file", "api", "diff", "patch", "import",
    "theta", "bishop", "rama", "pro", "gly", "backbone",
}

STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "be",
    "as", "at", "by", "with", "from", "that", "this", "it", "we", "you", "our",
    "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
    "not", "no", "yes", "if", "then", "than", "so", "but", "into", "via",
}


def tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_./+-]*", (text or "").lower())
    out = []
    for t in raw:
        t = t.strip("._-")
        if len(t) < 2 or t in STOP_WORDS:
            continue
        out.append(t)
    return out


def feature_set(text: str) -> set[str]:
    return set(tokenize(text))


def project_to_interface(feats: set[str]) -> set[str]:
    """Restriction map ρ: domain/human stalk → interface stalk."""
    hit = set()
    for f in feats:
        if f in INTERFACE:
            hit.add(f)
        # soft stems
        for iface in INTERFACE:
            if f.startswith(iface) or iface.startswith(f) and len(f) >= 4:
                hit.add(iface)
    return hit


@dataclass
class Projection:
    side: str  # human | domain
    domain: str  # language | code | rplc | eref | lm | field | custom
    raw: str
    features: list[str] = field(default_factory=list)
    interface: list[str] = field(default_factory=list)
    fingerprint: str = ""
    t: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_projection(
    side: str,
    domain: str,
    text: str,
    meta: dict[str, Any] | None = None,
) -> Projection:
    feats = feature_set(text)
    iface = project_to_interface(feats)
    fp = hashlib.sha256(
        (side + "|" + domain + "|" + " ".join(sorted(feats))).encode()
    ).hexdigest()[:16]
    return Projection(
        side=side,
        domain=domain,
        raw=text[:8000],
        features=sorted(feats)[:400],
        interface=sorted(iface),
        fingerprint=fp,
        meta=meta or {},
    )


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def align_projections(human: Projection, domain: Projection) -> dict[str, Any]:
    """
    Bilateral alignment measure.

    residual δ ~ symmetric difference on interface (primary) and full features (secondary).
    """
    h_i, d_i = set(human.interface), set(domain.interface)
    h_f, d_f = set(human.features), set(domain.features)

    iface_j = jaccard(h_i, d_i)
    feat_j = jaccard(h_f, d_f)
    # weight interface higher — shared law vocabulary
    align = 0.65 * iface_j + 0.35 * feat_j
    only_h = sorted(h_i - d_i)
    only_d = sorted(d_i - h_i)
    both = sorted(h_i & d_i)

    # Law-core words: if both sides project onto the same design-law spine,
    # that is meaningful glue even when feature vocab is heterophilic.
    LAW_CORE = {
        "open", "stop", "residue", "measure", "audit", "restrict", "certify",
        "force", "park", "prior", "claim",
    }
    law_hit = sorted(set(both) & LAW_CORE)
    # obstruction heuristic (not λ₁ claim): interface glue + law-core
    if (align >= 0.42 and len(both) >= 2) or (len(law_hit) >= 3 and iface_j >= 0.2):
        regime = "glue_ok"
        openable = True
    elif align >= 0.22 or len(law_hit) >= 2 or len(both) >= 4:
        regime = "partial_section"
        openable = False
    else:
        regime = "frustrated"
        openable = False

    return {
        "ok": True,
        "mode": "language_projection_align",
        "align": round(align, 4),
        "interface_jaccard": round(iface_j, 4),
        "feature_jaccard": round(feat_j, 4),
        "shared_interface": both,
        "law_core_shared": law_hit,
        "human_only_interface": only_h,
        "domain_only_interface": only_d,
        "regime": regime,  # glue_ok | partial_section | frustrated
        "openable_candidate": openable,
        "sheaf_tag": "alignment_helper",  # not pure obstruction energy
        "note": (
            "Language is projection from both sides. "
            "align/law_core_shared measure glue — not force-OPEN authority. "
            "Heterophily of full vocab is expected; interface law-core matters more."
        ),
        "human_fp": human.fingerprint,
        "domain_fp": domain.fingerprint,
        "human_domain": human.domain,
        "domain_domain": domain.domain,
    }


# ---------------------------------------------------------------------------
# Domain extractors — domain "language" as projection of substrate
# ---------------------------------------------------------------------------

def extract_code_language(workspace: str, max_files: int = 40) -> str:
    """Project code side: names + docs strings surface (not full AST)."""
    ws = Path(workspace)
    chunks: list[str] = []
    patterns = ("*.py", "*.ts", "*.js", "*.md", "*.json")
    files: list[Path] = []
    for pat in patterns:
        files.extend(ws.rglob(pat))
    # skip heavy dirs
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "_gob_extract", "_sandbox_extract"}
    files = [
        f for f in files
        if not any(p in skip for p in f.parts)
    ][:max_files]
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")[:4000]
        except Exception:
            continue
        # identifiers + headings
        ids = re.findall(r"\bdef ([a-zA-Z_][\w]*)|class ([a-zA-Z_][\w]*)|#+\s+(.+)", text)
        flat = []
        for m in ids:
            flat.extend([x for x in m if x])
        chunks.append(f.name + " " + " ".join(flat[:40]))
        # first docstring-ish lines
        for line in text.splitlines()[:30]:
            if line.strip().startswith(('"""', "'''", "#", "OPEN", "STOP", "residue")):
                chunks.append(line.strip()[:120])
    return "\n".join(chunks) if chunks else f"empty workspace {workspace}"


def extract_rplc_language(workspace: str) -> str:
    ws = Path(workspace)
    parts = []
    for name in ("docs/CLAIMS.md", "README.md", "docs/SPARSE_ALU.md"):
        p = ws / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="ignore")[:6000])
    # synthetic law words if package present
    if (ws / "rplc_sheaf.py").exists():
        parts.append(
            "rplc sheaf open stop residue measure audit restrict certify "
            "lambda holonomy smoke verify certificate force never"
        )
    return "\n".join(parts) or "rplc domain not found"


def extract_eref_language(workspace: str) -> str:
    roots = [Path(workspace)]
    try:
        from workspace import workspace_root

        roots.append(workspace_root() / "topology-sees-sequence")
    except Exception:
        pass
    parts = []
    for root in roots:
        for name in (
            "NOTES_EREF_PRIOR.md",
            "CORRECTION_NOTE.md",
            "STATUS_P1_P2.md",
            "README.md",
        ):
            p = root / name
            if p.exists():
                parts.append(p.read_text(encoding="utf-8", errors="ignore")[:5000])
        if (root / "derive.py").exists():
            parts.append(
                "E_ref reference connection strain sequence prior geometry "
                "bishop holonomy not sheaf theta park P2 backbone energy "
                "residue poly-pro omega cis_pro trans"
            )
            break
    return "\n".join(parts) or "eref domain not found"


def extract_field_language(workspace: str) -> str:
    """Multiplane / harness / drilling vocabulary projection."""
    roots = [Path(workspace)]
    try:
        from workspace import harness_root, prime_root

        roots.extend([harness_root(), prime_root()])
    except Exception:
        pass
    parts = [
        "multiplane LIVE DEAD Mag-PI WITS dump emz ingest pack certify "
        "OPEN STOP scout local LFM bonsai plane inventory residue"
    ]
    for root in roots:
        for name in ("docs/FIELD_RUNBOOK.md", "README.md", "docs/README.md"):
            p = root / name
            if p.exists():
                parts.append(p.read_text(encoding="utf-8", errors="ignore")[:4000])
    return "\n".join(parts)


def extract_domain(domain: str, workspace: str, text: str = "") -> Projection:
    domain = (domain or "code").lower()
    if text:
        raw = text
    elif domain in ("code", "repo", "software"):
        raw = extract_code_language(workspace)
        domain = "code"
    elif domain in ("rplc", "sheaf_alu"):
        raw = extract_rplc_language(workspace)
        domain = "rplc"
    elif domain in ("eref", "topology", "geometry", "sequence"):
        raw = extract_eref_language(workspace)
        domain = "eref"
    elif domain in ("field", "harness", "multiplane", "mwd"):
        raw = extract_field_language(workspace)
        domain = "field"
    elif domain in ("lm", "scout"):
        raw = text or "local scout measure not proof open stop residue"
        domain = "lm"
    else:
        raw = text or domain
        domain = "custom"
    return make_projection("domain", domain, raw, meta={"workspace": workspace})


def extract_human(text: str, domain_tag: str = "language") -> Projection:
    return make_projection("human", domain_tag, text, meta={})


def bilateral_measure(
    human_text: str,
    workspace: str,
    domains: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Most effective operationalization: project human language once,
    project each domain, align each pair, aggregate.
    """
    domains = list(domains or ("code", "rplc", "eref", "field"))
    human = extract_human(human_text)
    rows = []
    for d in domains:
        dom = extract_domain(d, workspace)
        al = align_projections(human, dom)
        rows.append(
            {
                "domain": d,
                "align": al["align"],
                "regime": al["regime"],
                "openable_candidate": al["openable_candidate"],
                "shared_interface": al["shared_interface"],
                "law_core_shared": al.get("law_core_shared") or [],
                "interface_jaccard": al.get("interface_jaccard"),
                "feature_jaccard": al.get("feature_jaccard"),
                "human_only": al["human_only_interface"],
                "domain_only": al["domain_only_interface"],
                "domain_fp": dom.fingerprint,
            }
        )
    rows_sorted = sorted(rows, key=lambda r: (r["align"], len(r.get("shared_interface") or [])), reverse=True)
    best = rows_sorted[0] if rows_sorted else None
    mean_align = sum(r["align"] for r in rows) / max(len(rows), 1)
    any_glue = any(r["regime"] == "glue_ok" for r in rows)
    all_frustrated = all(r["regime"] == "frustrated" for r in rows) if rows else True
    # promote best row law_core into summary
    if best and best.get("shared_interface"):
        best = dict(best)

    return {
        "ok": True,
        "mode": "bilateral_language_projection",
        "thesis": "language is a projection from both sides",
        "human": {
            "fingerprint": human.fingerprint,
            "interface": human.interface,
            "n_features": len(human.features),
        },
        "domains": rows_sorted,
        "best_domain": best,
        "mean_align": round(mean_align, 4),
        "any_glue_ok": any_glue,
        "all_frustrated": all_frustrated,
        "recommendation": (
            "Prefer domain with highest glue; CODE only where align is partial+; "
            "STOP if all frustrated; never force-OPEN on language alone."
        ),
        "sheaf_tag": "alignment_helper",
    }
