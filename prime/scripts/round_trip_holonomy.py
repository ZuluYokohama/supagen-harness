"""
Round-trip holonomy on claims — TOPICALITY, not AGREEMENT.

  C ──write RESTRICT──▶ R ──reconstruct──▶ C′
  └──────────── loop (drift) ────────────┘

Drift = nomic aboutness distance (1 − cosine) after strip+prefix.
Purpose: TOPICALITY — uses the instrument that *passed* its null (range ~0.45).
Not AGREEMENT (NLI). Cosine never claims entailment.

Null (run first):
  real claim     → expect low drift
  paraphrase     → similar drift
  pasta          → floor (high drift)

If all three drift the same → reconstruction noise, not claim structure.
Second-circuit probe: drift should not keep growing if map contracts.

Law: restrict → measure → audit → OPEN|STOP. Residue never forced.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from lms_layers import DEFAULT_BASE, DEFAULT_LFM, l2_chat
from metric_text import strip_envelope, strip_prompt_chrome
from nomic_metric import aboutness
from purpose_gate import TOPICALITY


# --- LFM legs: write RESTRICT / reconstruct claim ---------------------------

WRITE_SYSTEM = (
    "ROLE=RESTRICT_WRITER. From the CLAIM, write a RESTRICT block only. "
    "Output STRICT JSON: "
    '{"goal":"<one sentence>","non_goals":["..."],"success":["..."],"constraints":["..."]} '
    "Be concrete. No ellipsis placeholders. No filler. No OPEN/STOP verdicts."
)

RECON_SYSTEM = (
    "ROLE=CLAIM_RECONSTRUCTOR. You see only a RESTRICT JSON block. "
    "You do NOT see the original claim. "
    "Reconstruct the underlying claim as one clear sentence. "
    "Output STRICT JSON: {\"claim\":\"<one sentence>\"} "
    "No preamble. No ellipsis."
)


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if t.startswith("```"):
        import re

        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        o = json.loads(t)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        import re

        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                o = json.loads(m.group(0))
                return o if isinstance(o, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def write_restrict(
    claim: str,
    *,
    model: str = DEFAULT_LFM,
    base: str = DEFAULT_BASE,
    compression: str = "full",
) -> dict[str, Any]:
    """
    C → R. compression: full | sentence | clause | five_words
    (terseness instruction for the S-curve sweep)
    """
    hint = {
        "full": "Full RESTRICT with goal, 2-3 non_goals, 2 success checks, 2 constraints.",
        "sentence": "Compress hard: goal only, one sentence. Other fields one short item each.",
        "clause": "Extreme: goal is a single clause under 15 words. Minimal other fields.",
        "five_words": "Maximal compression: goal is exactly five words. Other fields empty lists.",
    }.get(compression, "full")
    user = f"CLAIM:\n{claim}\n\n{hint}\nEmit RESTRICT JSON."
    r = l2_chat(
        user,
        model=model,
        system=WRITE_SYSTEM,
        temperature=0.1,
        max_tokens=200,
        store=False,
        context_length=4096,
        base=base,
    )
    obj = _parse_json_obj(r.get("content") or "") if r.get("ok") else None
    return {
        "ok": bool(r.get("ok") and obj),
        "restrict": obj,
        "raw": (r.get("content") or "")[:800],
        "compression": compression,
        "cost": r.get("cost"),
        "error": r.get("error"),
    }


def reconstruct_claim(
    restrict: dict[str, Any] | str,
    *,
    model: str = DEFAULT_LFM,
    base: str = DEFAULT_BASE,
) -> dict[str, Any]:
    """R → C′ with no access to original C."""
    payload = restrict if isinstance(restrict, str) else json.dumps(restrict, ensure_ascii=False)
    user = f"RESTRICT:\n{payload}\n\nReconstruct the claim JSON."
    r = l2_chat(
        user,
        model=model,
        system=RECON_SYSTEM,
        temperature=0.1,
        max_tokens=120,
        store=False,
        context_length=2048,
        base=base,
    )
    obj = _parse_json_obj(r.get("content") or "") if r.get("ok") else None
    claim = ""
    if obj:
        claim = str(obj.get("claim") or obj.get("goal") or "").strip()
        # model sometimes nests broken JSON as the claim string
        if claim.startswith("{") or "claim" in claim[:20]:
            inner = _parse_json_obj(claim)
            if inner:
                claim = str(inner.get("claim") or inner.get("goal") or claim).strip()
            else:
                import re

                m = re.search(r'"claim"\s*:\s*"([^"]+)"', claim)
                if m:
                    claim = m.group(1).strip()
    if not claim and r.get("ok"):
        claim = strip_envelope(r.get("content") or "")[:400]
    # never leave JSON braces / broken claim"> artifacts in C′
    claim = strip_envelope(claim) if claim.strip().startswith("{") else claim
    import re

    claim = re.sub(r'^\{?"?claim"?\s*[>:=]\s*"?', "", claim.strip())
    claim = claim.strip().strip('"').rstrip('"}').strip()
    return {
        "ok": bool(claim) and len(claim) > 8,
        "claim_prime": claim,
        "raw": (r.get("content") or "")[:500],
        "cost": r.get("cost"),
        "error": r.get("error"),
    }


def topicality_drift(a: str, b: str, base: str = DEFAULT_BASE) -> dict[str, Any]:
    """1 − cosine aboutness. Purpose TOPICALITY only."""
    ab = aboutness(
        strip_prompt_chrome(a),
        strip_prompt_chrome(b),
        a_task="search_query",
        b_task="search_document",
        base=base,
    )
    cos = ab.get("cosine")
    drift = None if cos is None else round(1.0 - float(cos), 4)
    return {
        "ok": ab.get("ok"),
        "purpose": TOPICALITY,
        "cosine": cos,
        "drift": drift,  # holonomy magnitude proxy
        "not_agreement": True,
        "error": ab.get("error"),
    }


def single_hop(
    claim: str,
    *,
    model: str = DEFAULT_LFM,
    base: str = DEFAULT_BASE,
    compression: str = "full",
) -> dict[str, Any]:
    """One circuit: C → R → C′ + drift(C, C′)."""
    t0 = time.time()
    w = write_restrict(claim, model=model, base=base, compression=compression)
    if not w.get("ok") or not w.get("restrict"):
        return {
            "ok": False,
            "error": "restrict_write_failed",
            "write": w,
            "elapsed_s": round(time.time() - t0, 2),
        }
    rec = reconstruct_claim(w["restrict"], model=model, base=base)
    if not rec.get("ok"):
        return {
            "ok": False,
            "error": "reconstruct_failed",
            "write": w,
            "recon": rec,
            "elapsed_s": round(time.time() - t0, 2),
        }
    d = topicality_drift(claim, rec["claim_prime"], base=base)
    return {
        "ok": bool(d.get("ok")),
        "claim": claim,
        "restrict": w["restrict"],
        "claim_prime": rec["claim_prime"],
        "drift": d.get("drift"),
        "cosine": d.get("cosine"),
        "purpose": TOPICALITY,
        "compression": compression,
        "holonomy": {
            "kind": "round_trip_restrict",
            "closed": (d.get("drift") is not None and d["drift"] < 0.25),
            "drift": d.get("drift"),
            "note": "closed iff topicality drift below soft threshold — not agreement",
        },
        "write_cost": w.get("cost"),
        "recon_cost": rec.get("cost"),
        "elapsed_s": round(time.time() - t0, 2),
        "thesis": (
            "Round-trip is holonomy: went around RESTRICT, came back. "
            "Drift is TOPICALITY (nomic). Certificate still owns OPEN."
        ),
    }


def second_circuit(first: dict[str, Any], *, model: str = DEFAULT_LFM, base: str = DEFAULT_BASE) -> dict[str, Any]:
    """Asymmetry: C′ → R′ → C″; compare drift growth."""
    c1 = first.get("claim_prime") or ""
    if not c1:
        return {"ok": False, "error": "no_claim_prime"}
    hop2 = single_hop(c1, model=model, base=base, compression=first.get("compression") or "full")
    d1 = first.get("drift")
    d2 = hop2.get("drift")
    return {
        "ok": hop2.get("ok"),
        "hop1_drift": d1,
        "hop2_drift": d2,
        "hop2": hop2,
        "contracting": (
            d1 is not None and d2 is not None and d2 < d1 + 0.05
        ),
        "note": (
            "If hop2_drift keeps growing, no attractor — closure framing weak. "
            "If hop2 ≤ hop1, map contracts toward fixed point."
        ),
    }


def null_three(
    *,
    model: str = DEFAULT_LFM,
    base: str = DEFAULT_BASE,
) -> dict[str, Any]:
    """
    Null: real claim | paraphrase | pasta — compare drifts.
    If all similar → measuring reconstruction noise, not claim structure.
    """
    real = (
        "Prime dual metric: nomic measures aboutness for retrieval only; "
        "NLI measures agreement; production OPEN requires audit under design law; residue never forced."
    )
    para = (
        "Under the design law, retrieval uses nomic aboutness while agreement uses NLI; "
        "nothing production-OPENs without audit; residue is never forced."
    )
    pasta = (
        "Carbonara is made with guanciale, egg, pecorino romano, and black pepper; "
        "no cream is traditional in the Roman preparation."
    )
    rows = []
    for name, text in (("real_claim", real), ("paraphrase", para), ("pasta", pasta)):
        h = single_hop(text, model=model, base=base, compression="full")
        rows.append({"name": name, **{k: h.get(k) for k in (
            "ok", "drift", "cosine", "claim_prime", "elapsed_s", "error"
        )}})
    by = {r["name"]: r.get("drift") for r in rows}
    real_d, para_d, pasta_d = by.get("real_claim"), by.get("paraphrase"), by.get("pasta")
    drifts = [d for d in (real_d, para_d, pasta_d) if d is not None]
    if len(drifts) == 3:
        spread = max(drifts) - min(drifts)
        # Structure signal: real & paraphrase behave similarly; pasta differs
        rp = abs(real_d - para_d)
        r_pasta = abs(real_d - pasta_d)
        structure = rp < r_pasta and spread >= 0.10
        # Inverted floor is OK: simple closed claims can round-trip *better*
        # (logogram survives; complex claims shed). Mirror = flat across all three.
        mirror = spread < 0.08
        earned = structure and not mirror
        pasta_minus_real = pasta_d - real_d
    else:
        spread = pasta_minus_real = None
        structure = False
        mirror = True
        earned = False
        rp = r_pasta = None
    return {
        "ok": True,
        "purpose": TOPICALITY,
        "rows": rows,
        "spread": round(spread, 4) if spread is not None else None,
        "pasta_minus_real": round(pasta_minus_real, 4) if pasta_minus_real is not None else None,
        "real_para_gap": round(rp, 4) if rp is not None else None,
        "real_pasta_gap": round(r_pasta, 4) if r_pasta is not None else None,
        "mirror": mirror,
        "structure_signal": structure if len(drifts) == 3 else False,
        "earned": earned,
        "verdict": (
            "EARNED_TOPICALITY_HOLONOMY"
            if earned
            else ("MIRROR_NOISE" if mirror else "WEAK_SEPARATION")
        ),
        "reading": (
            "Real≈paraphrase, pasta differs — round-trip tracks claim structure (TOPICALITY). "
            + (
                "Pasta lower drift: simple closed propositions survive better (logogram-shaped)."
                if pasta_minus_real is not None and pasta_minus_real < 0
                else "Pasta higher drift: expected floor."
            )
            if earned
            else (
                "All drifts similar — reconstruction noise / decoration."
                if mirror
                else "Some separation; inspect claim_prime. Inverted pasta is informative if real≈para."
            )
        ),
        "thesis": (
            "Round-trip holonomy uses nomic TOPICALITY (passed aboutness null). "
            "Not AGREEMENT. Not production OPEN. "
            "C→R→C′ is the semasiographic test: did R contain the claim?"
        ),
    }


def compression_sweep(
    claim: str,
    *,
    model: str = DEFAULT_LFM,
    base: str = DEFAULT_BASE,
    levels: list[str] | None = None,
) -> dict[str, Any]:
    """S-curve: drift vs compression level (minimum description knee)."""
    levels = levels or ["full", "sentence", "clause", "five_words"]
    points = []
    for lev in levels:
        h = single_hop(claim, model=model, base=base, compression=lev)
        r = h.get("restrict") or {}
        rlen = len(json.dumps(r, ensure_ascii=False)) if r else 0
        points.append({
            "compression": lev,
            "restrict_chars": rlen,
            "drift": h.get("drift"),
            "cosine": h.get("cosine"),
            "claim_prime": (h.get("claim_prime") or "")[:200],
            "ok": h.get("ok"),
        })
    return {
        "ok": all(p.get("ok") for p in points),
        "claim": claim[:300],
        "curve": points,
        "purpose": TOPICALITY,
        "note": "Knee = compression where drift jumps — MDL proxy under this agent",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Round-trip holonomy (TOPICALITY)")
    ap.add_argument("cmd", nargs="?", default="null", choices=["null", "hop", "sweep", "second"])
    ap.add_argument("--claim", default="")
    ap.add_argument("--compression", default="full")
    ap.add_argument("--model", default=DEFAULT_LFM)
    args = ap.parse_args()

    if args.cmd == "null":
        print(json.dumps(null_three(model=args.model), indent=2))
        return
    claim = args.claim or (
        "Design law: restrict measure audit then OPEN only with purpose-matched measures; residue never forced."
    )
    if args.cmd == "hop":
        print(json.dumps(single_hop(claim, model=args.model, compression=args.compression), indent=2))
        return
    if args.cmd == "sweep":
        print(json.dumps(compression_sweep(claim, model=args.model), indent=2))
        return
    if args.cmd == "second":
        h1 = single_hop(claim, model=args.model, compression=args.compression)
        h2 = second_circuit(h1, model=args.model)
        print(json.dumps({"hop1": h1, "second": h2}, indent=2))


if __name__ == "__main__":
    main()
