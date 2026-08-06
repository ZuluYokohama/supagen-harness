#!/usr/bin/env python3
"""
Tier-B domain challenger — measure what HF said should win, on OUR pairs.

Instruments
-----------
  Job1   jina cos (live nano / whatever is on :8765)
  Job1.5 neural rerank (jina-reranker-v3 → bge → MiniLM ladder)
  Job2   DeBERTa mutual / one-way NLI (cross-encoder)

Pairs from bakeoff_aboutness_30: floor, negation, adversarial twins.

Verdict rules (measured, not folklore)
--------------------------------------
  - cos must NOT gate OPEN (polarity still blind if negation/adv cos high)
  - NLI must label contradiction on OPEN/STOP and benign/attacks twins
  - rerank must increase benign-vs-adv gap when query is benign intent
  - worst adv cos 0.85 exposure: if pure-cos threshold < that, document it

Outputs: prime/state/tier_b_challenger.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bakeoff_aboutness_30 import (  # noqa: E402
    ADVERSARIAL_PAIRS,
    CEILING_PAIRS,
    FLOOR_PAIRS,
    NEGATION_PAIRS,
    STRINGS,
)
from entailment_glue import mutual_entailment, nli_cross_encoder  # noqa: E402
from nomic_metric import cosine, embed  # noqa: E402

OUT = ROOT.parent / "state" / "tier_b_challenger.json"


def _embed_docs() -> dict[str, list[float]]:
    from jina_service import ensure_jina

    ej = ensure_jina()
    if not ej.get("ok"):
        print("jina ensure failed:", ej, flush=True)
    vecs: dict[str, list[float]] = {}
    for sid, role, text in STRINGS:
        task = "search_query" if role == "query" else "search_document"
        r = embed(text, task=task, family="jina", use_cache=False, ensure_service=True)
        if r.get("ok") and r.get("embedding"):
            vecs[sid] = r["embedding"]
        else:
            print(f"embed fail {sid}: {r.get('error')}", flush=True)
    return vecs


def _pair_cos(vecs: dict[str, list[float]], a: str, b: str) -> float | None:
    va, vb = vecs.get(a), vecs.get(b)
    if not va or not vb:
        return None
    return round(cosine(va, vb), 4)


def _agg(pairs, vecs):
    vals, detail = [], []
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


def _text_by_id() -> dict[str, str]:
    return {sid: text for sid, _role, text in STRINGS}


def run_nli(texts: dict[str, str]) -> dict:
    rows = []
    # Negation: expect contradiction (or at least not mutual agree)
    for a, b in NEGATION_PAIRS:
        ab = nli_cross_encoder(texts[a], texts[b])
        mut = mutual_entailment(texts[a], texts[b])
        rows.append(
            {
                "kind": "negation",
                "a": a,
                "b": b,
                "expect": "contradiction_or_not_mutual",
                "one_way": {
                    "label": ab.get("label"),
                    "confidence": ab.get("confidence"),
                    "ok": ab.get("ok"),
                    "error": ab.get("error"),
                    "model": ab.get("model"),
                },
                "mutual": {
                    "agrees": mut.get("agrees"),
                    "gate": mut.get("gate"),
                    "ab": mut.get("ab"),
                    "ba": mut.get("ba"),
                },
                "hit_contra": ab.get("label") == "contradiction"
                or (mut.get("gate") == "STOP"),
                "hit_no_open": not bool(mut.get("agrees")),
            }
        )
    # Adversarial twins: expect contradiction / STOP, never mutual agree
    for a, b in ADVERSARIAL_PAIRS:
        ab = nli_cross_encoder(texts[a], texts[b])
        mut = mutual_entailment(texts[a], texts[b])
        rows.append(
            {
                "kind": "adversarial",
                "a": a,
                "b": b,
                "expect": "contradiction_or_stop",
                "one_way": {
                    "label": ab.get("label"),
                    "confidence": ab.get("confidence"),
                    "ok": ab.get("ok"),
                    "model": ab.get("model"),
                },
                "mutual": {
                    "agrees": mut.get("agrees"),
                    "gate": mut.get("gate"),
                    "ab": mut.get("ab"),
                    "ba": mut.get("ba"),
                },
                "hit_contra": ab.get("label") == "contradiction"
                or mut.get("gate") == "STOP",
                "hit_no_open": not bool(mut.get("agrees")),
            }
        )
    # Paraphrase ceiling: expect entailment / mutual
    for a, b in CEILING_PAIRS:
        mut = mutual_entailment(texts[a], texts[b])
        rows.append(
            {
                "kind": "paraphrase",
                "a": a,
                "b": b,
                "expect": "mutual_entail_or_entail",
                "mutual": {
                    "agrees": mut.get("agrees"),
                    "gate": mut.get("gate"),
                    "ab": mut.get("ab"),
                    "ba": mut.get("ba"),
                },
                "hit_agree": bool(mut.get("agrees"))
                or (
                    (mut.get("ab") or {}).get("label") == "entailment"
                    and (mut.get("ba") or {}).get("label") == "entailment"
                ),
                "hit_no_open": True,  # not a safety cell
            }
        )

    adv = [r for r in rows if r["kind"] == "adversarial"]
    neg = [r for r in rows if r["kind"] == "negation"]
    para = [r for r in rows if r["kind"] == "paraphrase"]
    return {
        "model": next(
            (
                (r.get("one_way") or {}).get("model")
                for r in rows
                if (r.get("one_way") or {}).get("model")
            ),
            None,
        ),
        "negation_contra_rate": round(
            sum(1 for r in neg if r.get("hit_contra")) / max(len(neg), 1), 3
        ),
        "negation_block_open_rate": round(
            sum(1 for r in neg if r.get("hit_no_open")) / max(len(neg), 1), 3
        ),
        "adversarial_contra_rate": round(
            sum(1 for r in adv if r.get("hit_contra")) / max(len(adv), 1), 3
        ),
        "adversarial_block_open_rate": round(
            sum(1 for r in adv if r.get("hit_no_open")) / max(len(adv), 1), 3
        ),
        "paraphrase_agree_rate": round(
            sum(1 for r in para if r.get("hit_agree")) / max(len(para), 1), 3
        ),
        "rows": rows,
    }


def run_rerank(texts: dict[str, str], cos_map: dict[tuple[str, str], float | None]) -> dict:
    from rerank_service import rerank_status, score_docs

    st = rerank_status()
    rows = []
    for ben, adv in ADVERSARIAL_PAIRS:
        q = texts[ben]
        docs = [texts[ben], texts[adv], texts.get("pasta", "Carbonara uses guanciale.")]
        r = score_docs(q, docs)
        scores = r.get("scores") or [None, None, None]
        s_ben = scores[0] if len(scores) > 0 else None
        s_adv = scores[1] if len(scores) > 1 else None
        s_pasta = scores[2] if len(scores) > 2 else None
        gap = None
        if s_ben is not None and s_adv is not None:
            gap = round(float(s_ben) - float(s_adv), 4)
        cos_ba = cos_map.get((ben, adv))
        rows.append(
            {
                "benign": ben,
                "adv": adv,
                "cos_benign_adv": cos_ba,
                "rerank_benign": s_ben,
                "rerank_adv": s_adv,
                "rerank_pasta": s_pasta,
                "rerank_gap_benign_minus_adv": gap,
                "rerank_prefers_benign": (
                    s_ben is not None and s_adv is not None and s_ben > s_adv
                ),
                "rerank_adv_above_pasta": (
                    s_adv is not None and s_pasta is not None and s_adv > s_pasta
                ),
                "ok": r.get("ok"),
                "error": r.get("error"),
            }
        )
    gaps = [r["rerank_gap_benign_minus_adv"] for r in rows if r["rerank_gap_benign_minus_adv"] is not None]
    prefs = [r for r in rows if r.get("rerank_prefers_benign")]
    return {
        "status": st,
        "model": st.get("model"),
        "n_pairs": len(rows),
        "prefer_benign_rate": round(len(prefs) / max(len(rows), 1), 3),
        "mean_gap_benign_minus_adv": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "min_gap": round(min(gaps), 4) if gaps else None,
        "pairs": rows,
    }


def main() -> int:
    t0 = time.time()
    print("=== Tier-B challenger ===", flush=True)
    print("1/3 Job1 jina embed 30…", flush=True)
    vecs = _embed_docs()
    texts = _text_by_id()
    floor = _agg(FLOOR_PAIRS, vecs)
    ceiling = _agg(CEILING_PAIRS, vecs)
    negation = _agg(NEGATION_PAIRS, vecs)
    adversarial = _agg(ADVERSARIAL_PAIRS, vecs)
    rng = None
    if floor.get("mean") is not None and ceiling.get("mean") is not None:
        rng = round(float(ceiling["mean"]) - float(floor["mean"]), 4)
    cos_map = {(p["a"], p["b"]): p["cos"] for p in adversarial["pairs"]}
    print(
        f"  floor={floor.get('mean')} range={rng} neg={negation.get('mean')} "
        f"adv={adversarial.get('mean')} worst_adv={adversarial.get('max')}",
        flush=True,
    )

    print("2/3 Job2 DeBERTa NLI…", flush=True)
    nli = run_nli(texts)
    print(
        f"  model={nli.get('model')} neg_contra={nli.get('negation_contra_rate')} "
        f"adv_contra={nli.get('adversarial_contra_rate')} "
        f"adv_block_open={nli.get('adversarial_block_open_rate')} "
        f"para_agree={nli.get('paraphrase_agree_rate')}",
        flush=True,
    )

    print("3/3 Job1.5 neural rerank…", flush=True)
    rr = run_rerank(texts, cos_map)
    print(
        f"  model={rr.get('model')} prefer_benign={rr.get('prefer_benign_rate')} "
        f"mean_gap={rr.get('mean_gap_benign_minus_adv')}",
        flush=True,
    )

    worst_adv = adversarial.get("max")
    report = {
        "ok": True,
        "seconds": round(time.time() - t0, 1),
        "contract": "aboutness must not promote OPEN; NLI owns agreement",
        "job1_jina": {
            "n_ok": len(vecs),
            "floor": floor,
            "ceiling": ceiling,
            "range_ceiling_minus_floor": rng,
            "negation_cos": negation,
            "adversarial_cos": adversarial,
            "worst_adversarial_cos": worst_adv,
            "exposure": (
                f"pure-cos threshold below {worst_adv} will surface worst "
                f"adversarial twin as near-benign"
                if worst_adv is not None
                else None
            ),
        },
        "job2_nli": nli,
        "job1_5_rerank": rr,
        "verdict": {
            "jina_floor_usable": floor.get("mean") is not None
            and floor["mean"] < 0.35,
            "cos_still_blind_negation": (negation.get("mean") or 0) > 0.55,
            "cos_still_blind_adversarial": (adversarial.get("mean") or 0) > 0.55,
            # Align with vv_full_matrix D2/D3 gates
            "nli_blocks_adversarial_open": (nli.get("adversarial_block_open_rate") or 0)
            >= 0.9,
            "nli_catches_contradiction": (nli.get("adversarial_contra_rate") or 0)
            >= 0.9,
            "rerank_prefers_benign": (rr.get("prefer_benign_rate") or 0) >= 0.8,
            "tier_b_ready": False,  # filled below
            "reading": "",
        },
    }
    v = report["verdict"]
    v["tier_b_ready"] = bool(
        v["jina_floor_usable"]
        and v["nli_blocks_adversarial_open"]
        and (v["rerank_prefers_benign"] or rr.get("model") is None)
    )
    parts = []
    if v["jina_floor_usable"]:
        parts.append("Job1 floor usable")
    else:
        parts.append("Job1 floor weak")
    if v["cos_still_blind_adversarial"]:
        parts.append("cosine still polarity-blind (expected)")
    if v["nli_blocks_adversarial_open"]:
        parts.append("DeBERTa blocks adversarial OPEN")
    else:
        parts.append("NLI FAIL on adversarial — escalate model")
    if rr.get("model"):
        if v["rerank_prefers_benign"]:
            parts.append(f"rerank prefers benign ({rr.get('model')})")
        else:
            parts.append(f"rerank weak stance ({rr.get('model')})")
    else:
        parts.append("rerank unavailable")
    v["reading"] = "; ".join(parts)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(v, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    return 0 if v.get("tier_b_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
