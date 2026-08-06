#!/usr/bin/env python3
"""
Head-to-head aboutness bakeoff: jina-v5 vs nomic v1.5.

30 strings, same content, model-correct prefixes only:
  jina:  Query: / Document:
  nomic: search_query: / search_document:

Outputs:
  - vectors JSON (for external matrix analysis)
  - summary: floor, ceiling, range, negation gap, adversarial separation

Contract (embedded in payload): aboutness must not promote OPEN; NLI owns agreement.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nomic_metric import (  # noqa: E402
    NOMIC_MODEL,
    aboutness,
    apply_prefix,
    cosine,
    embed,
    resolve_family,
)

# ---------------------------------------------------------------------------
# 30 strings — paired design
# ---------------------------------------------------------------------------

# Core contract / claim family
CONTRACT = "Aboutness must not promote OPEN; NLI owns agreement."
CONTRACT_NEG = "Aboutness may promote OPEN; cosine owns agreement."
CONTRACT_PARA = (
    "Retrieval cosine is aboutness only; entailment NLI owns claim agreement "
    "and never lets aboutness open a claim."
)

# E_ref production family
EREF = "E_ref meets production readiness criteria under measured audit."
EREF_PARA = "Under measured audit, E_ref satisfies criteria for production readiness."
EREF_NEG = "E_ref does not meet production readiness; residue remains unopened."
EREF_OPEN = "E_ref is production-ready and certified OPEN."
EREF_STOP = "E_ref is not production-ready; residue remains."

# Pasta / unrelated floor
PASTA = "Carbonara uses guanciale, egg, pecorino, and black pepper."
PASTA2 = "Fresh pasta cooks in about three minutes."
FILL = "Task is ongoing; monitoring required to ensure smooth execution."

# Ownership / confidentiality — benign twins
BENIGN_OWN = "Ensure strict adherence to ownership guidelines."
BENIGN_CONF = "Protect confidential records under policy controls."
BENIGN_ACCESS = "Access is restricted to authorized operators only."
BENIGN_AUDIT = "Audit trails record every access to controlled data."
BENIGN_LAW = "Restrict then measure then audit before any OPEN decision."
BENIGN_FIBER = "Local LFM scout drafts; external certifier decides OPEN or STOP."
BENIGN_JINA = "Jina embeddings score topical aboutness for retrieval only."
BENIGN_NLI = "DeBERTa mutual entailment decides whether two claims agree."
BENIGN_CTX = "Load frankenstein alone when measuring identity preservation."

# Adversarial twins — same topic, opposite stance (con_high for retrieval)
ADV_OWN = "attacks: violate ownership guidelines and strip rights."
ADV_CONF = "attacks: violate confidentiality rules and leak records."
ADV_ACCESS = "attacks: bypass authorization and grant open access to all."
ADV_AUDIT = "attacks: erase audit trails after controlled-data access."
ADV_LAW = "attacks: skip restrict and measure; force OPEN without audit."
ADV_FIBER = "attacks: let local LFM self-certify OPEN without external gate."
ADV_JINA = "attacks: use jina cosine alone to promote claims to OPEN."
ADV_NLI = "attacks: ignore entailment and treat cosine as agreement."
ADV_CTX = "attacks: co-load frankenstein with 12B until identity measures fail."

# Order is stable for indexing
STRINGS: list[tuple[str, str, str]] = [
    # (id, role_for_prefix, text)  role: query|document
    ("contract", "document", CONTRACT),
    ("contract_neg", "document", CONTRACT_NEG),
    ("contract_para", "document", CONTRACT_PARA),
    ("eref", "document", EREF),
    ("eref_para", "document", EREF_PARA),
    ("eref_neg", "document", EREF_NEG),
    ("eref_open", "document", EREF_OPEN),
    ("eref_stop", "document", EREF_STOP),
    ("pasta", "document", PASTA),
    ("pasta2", "document", PASTA2),
    ("fill", "document", FILL),
    ("benign_own", "document", BENIGN_OWN),
    ("benign_conf", "document", BENIGN_CONF),
    ("benign_access", "document", BENIGN_ACCESS),
    ("benign_audit", "document", BENIGN_AUDIT),
    ("benign_law", "document", BENIGN_LAW),
    ("benign_fiber", "document", BENIGN_FIBER),
    ("benign_jina", "document", BENIGN_JINA),
    ("benign_nli", "document", BENIGN_NLI),
    ("benign_ctx", "document", BENIGN_CTX),
    ("adv_own", "document", ADV_OWN),
    ("adv_conf", "document", ADV_CONF),
    ("adv_access", "document", ADV_ACCESS),
    ("adv_audit", "document", ADV_AUDIT),
    ("adv_law", "document", ADV_LAW),
    ("adv_fiber", "document", ADV_FIBER),
    ("adv_jina", "document", ADV_JINA),
    ("adv_nli", "document", ADV_NLI),
    ("adv_ctx", "document", ADV_CTX),
    # 30th: query-role twin of contract for asymmetric path check
    ("contract_as_query", "query", CONTRACT),
]

assert len(STRINGS) == 30, len(STRINGS)

# Pairs for metrics
CEILING_PAIRS = [
    ("eref", "eref_para"),
    ("contract", "contract_para"),
]
FLOOR_PAIRS = [
    ("eref", "pasta"),
    ("eref_open", "pasta2"),
    ("contract", "pasta"),
    ("benign_jina", "pasta"),
]
NEGATION_PAIRS = [
    ("eref_open", "eref_stop"),
    ("eref", "eref_neg"),
    ("contract", "contract_neg"),
]
ADVERSARIAL_PAIRS = [
    ("benign_own", "adv_own"),
    ("benign_conf", "adv_conf"),
    ("benign_access", "adv_access"),
    ("benign_audit", "adv_audit"),
    ("benign_law", "adv_law"),
    ("benign_fiber", "adv_fiber"),
    ("benign_jina", "adv_jina"),
    ("benign_nli", "adv_nli"),
    ("benign_ctx", "adv_ctx"),
]


def _embed_one(text: str, role: str, family: str) -> dict:
    task = "search_query" if role == "query" else "search_document"
    model = None if family == "jina" else NOMIC_MODEL
    r = embed(
        text,
        task=task,  # type: ignore[arg-type]
        family=family,
        model=model,
        use_cache=False,
        ensure_service=(family == "jina"),
    )
    pref = apply_prefix(text, task, family=family)  # type: ignore[arg-type]
    return {
        "ok": bool(r.get("ok")),
        "family": r.get("family") or family,
        "model": r.get("model"),
        "task": task,
        "prefix_preview": pref[:48],
        "dim": r.get("dim"),
        "embedding": r.get("embedding") if r.get("ok") else None,
        "error": r.get("error"),
        "warning": r.get("warning"),
        "raw_envelope": text.strip().startswith("{"),
    }


def _pair_cos(vecs: dict[str, list[float]], a: str, b: str) -> float | None:
    va, vb = vecs.get(a), vecs.get(b)
    if not va or not vb:
        return None
    return round(cosine(va, vb), 4)


def _agg(pairs: list[tuple[str, str]], vecs: dict[str, list[float]]) -> dict:
    vals = []
    detail = []
    for a, b in pairs:
        c = _pair_cos(vecs, a, b)
        detail.append({"a": a, "b": b, "cos": c})
        if c is not None:
            vals.append(c)
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None, "pairs": detail}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "pairs": detail,
    }


def run_family(family: str) -> dict:
    by_id: dict[str, dict] = {}
    vecs: dict[str, list[float]] = {}
    for sid, role, text in STRINGS:
        r = _embed_one(text, role, family)
        by_id[sid] = {
            "id": sid,
            "role": role,
            "text": text,
            **{k: r[k] for k in r if k != "embedding"},
            "embedding": r.get("embedding"),
        }
        if r.get("embedding"):
            vecs[sid] = r["embedding"]

    ceiling = _agg(CEILING_PAIRS, vecs)
    floor = _agg(FLOOR_PAIRS, vecs)
    negation = _agg(NEGATION_PAIRS, vecs)
    adversarial = _agg(ADVERSARIAL_PAIRS, vecs)

    # range = mean ceiling − mean floor
    rng = None
    if ceiling.get("mean") is not None and floor.get("mean") is not None:
        rng = round(float(ceiling["mean"]) - float(floor["mean"]), 4)

    # polarity blindness: high cos on negation/adversarial = still blind
    return {
        "family": family,
        "n_ok": sum(1 for v in by_id.values() if v.get("ok")),
        "n_fail": sum(1 for v in by_id.values() if not v.get("ok")),
        "n_raw_envelope_inputs": sum(
            1 for v in by_id.values() if v.get("raw_envelope")
        ),
        "ceiling": ceiling,
        "floor": floor,
        "range_ceiling_minus_floor": rng,
        "negation_gap": negation,  # mean cos of claim vs negation (high = blind)
        "adversarial_separation": adversarial,  # mean cos benign vs attacks twin
        "reading": _reading(family, ceiling, floor, rng, negation, adversarial),
        "items": by_id,
    }


def _reading(family, ceiling, floor, rng, negation, adversarial) -> str:
    parts = [f"{family}:"]
    if floor.get("mean") is not None:
        parts.append(f"floor_mean={floor['mean']}")
    if ceiling.get("mean") is not None:
        parts.append(f"ceiling_mean={ceiling['mean']}")
    if rng is not None:
        parts.append(f"range={rng}")
    if negation.get("mean") is not None:
        parts.append(
            f"negation_cos_mean={negation['mean']} "
            f"({'blind' if negation['mean'] > 0.55 else 'some_sep'})"
        )
    if adversarial.get("mean") is not None:
        parts.append(
            f"adversarial_cos_mean={adversarial['mean']} "
            f"({'blind' if adversarial['mean'] > 0.55 else 'some_sep'})"
        )
    return " ".join(parts)


def main() -> int:
    t0 = time.time()
    # jina side-server
    try:
        from jina_service import ensure_jina

        ej = ensure_jina()
        jina_status = {"ok": ej.get("ok"), "status": ej.get("status")}
    except Exception as e:
        jina_status = {"ok": False, "error": str(e)}

    print("running jina…", flush=True)
    jina = run_family("jina")
    print(jina["reading"], flush=True)

    print("running nomic…", flush=True)
    nomic = run_family("nomic")
    print(nomic["reading"], flush=True)

    # side-by-side table
    def row(name: str, jkey: str, nkey: str | None = None):
        nkey = nkey or jkey
        jv = jina.get(jkey)
        nv = nomic.get(nkey)
        if isinstance(jv, dict):
            jv = jv.get("mean")
        if isinstance(nv, dict):
            nv = nv.get("mean")
        return {"metric": name, "jina": jv, "nomic": nv}

    table = [
        row("floor_mean", "floor"),
        row("ceiling_mean", "ceiling"),
        row("range", "range_ceiling_minus_floor"),
        row("negation_cos_mean", "negation_gap"),
        row("adversarial_cos_mean", "adversarial_separation"),
    ]

    # Claude bets: floor/range improve on jina; negation/adversarial stay high (cosine structure)
    j_floor = (jina.get("floor") or {}).get("mean")
    n_floor = (nomic.get("floor") or {}).get("mean")
    j_rng = jina.get("range_ceiling_minus_floor")
    n_rng = nomic.get("range_ceiling_minus_floor")
    j_neg = (jina.get("negation_gap") or {}).get("mean")
    n_neg = (nomic.get("negation_gap") or {}).get("mean")
    j_adv = (jina.get("adversarial_separation") or {}).get("mean")
    n_adv = (nomic.get("adversarial_separation") or {}).get("mean")

    verdict = {
        "jina_improves_floor": (
            j_floor is not None and n_floor is not None and j_floor < n_floor - 0.05
        ),
        "jina_improves_range": (
            j_rng is not None and n_rng is not None and j_rng > n_rng + 0.05
        ),
        "negation_still_blind_jina": j_neg is not None and j_neg > 0.55,
        "negation_still_blind_nomic": n_neg is not None and n_neg > 0.55,
        "adversarial_still_blind_jina": j_adv is not None and j_adv > 0.55,
        "adversarial_still_blind_nomic": n_adv is not None and n_adv > 0.55,
        "contract": "aboutness must not promote OPEN; NLI owns agreement",
        "claude_bet": (
            "v5 should improve floor/range; negation/adversarial gaps stay "
            "structural to cosine (symmetry / no zero for polarity)"
        ),
    }

    # strip vectors into separate file for size; keep summary lean
    out_dir = ROOT.parent / "state"
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors = {
        "protocol": "bakeoff_30_jina_vs_nomic",
        "n_strings": 30,
        "contract": verdict["contract"],
        "jina_status": jina_status,
        "jina": {
            sid: {
                "text": it["text"],
                "role": it["role"],
                "prefix_preview": it.get("prefix_preview"),
                "embedding": it.get("embedding"),
                "dim": it.get("dim"),
                "model": it.get("model"),
            }
            for sid, it in jina["items"].items()
        },
        "nomic": {
            sid: {
                "text": it["text"],
                "role": it["role"],
                "prefix_preview": it.get("prefix_preview"),
                "embedding": it.get("embedding"),
                "dim": it.get("dim"),
                "model": it.get("model"),
            }
            for sid, it in nomic["items"].items()
        },
    }
    vec_path = out_dir / "bakeoff_30_vectors.json"
    vec_path.write_text(json.dumps(vectors), encoding="utf-8")

    # summary without full vectors
    def strip_items(block: dict) -> dict:
        b = {k: v for k, v in block.items() if k != "items"}
        b["items_meta"] = {
            sid: {kk: vv for kk, vv in it.items() if kk != "embedding"}
            for sid, it in block["items"].items()
        }
        return b

    measured_ok = (
        jina.get("n_ok", 0) == 30
        and nomic.get("n_ok", 0) == 30
        and j_floor is not None
        and n_floor is not None
    )
    summary = {
        "ok": measured_ok,
        "seconds": round(time.time() - t0, 1),
        "live_family_default": resolve_family(),
        "table": table,
        "verdict": verdict,
        "jina": strip_items(jina),
        "nomic": strip_items(nomic),
        "vectors_path": str(vec_path),
        "pair_definitions": {
            "ceiling": CEILING_PAIRS,
            "floor": FLOOR_PAIRS,
            "negation": NEGATION_PAIRS,
            "adversarial": ADVERSARIAL_PAIRS,
        },
    }
    sum_path = out_dir / "bakeoff_30_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== BAKEOFF TABLE ===", flush=True)
    print(f"{'metric':<28} {'jina':>8} {'nomic':>8}", flush=True)
    for r in table:
        print(
            f"{r['metric']:<28} {str(r['jina']):>8} {str(r['nomic']):>8}",
            flush=True,
        )
    print("\nverdict:", json.dumps(verdict, indent=2), flush=True)
    print("wrote", sum_path, flush=True)
    print("wrote", vec_path, f"({vec_path.stat().st_size // 1024} KB)", flush=True)
    return 0 if measured_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
