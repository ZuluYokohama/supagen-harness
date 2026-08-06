#!/usr/bin/env python3
"""
Push residual domains not fully closed in vv_full_matrix:
  P1  Job1 v5-small swap + floor/range remeasure
  P2  PRESERVE load smoke (frankenstein alone)
  P3  truth_plane enter live (SCOUT + loop)
  P4  cos-OPEN negative pack (certify-shaped)
  P5  ORT NLI default path hot
  P6  QNN/Hexagon provider probe (honest)

Writes prime/state/vv_push_domains.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state"
sys.path.insert(0, str(ROOT))


def gate(name: str, ok: bool, detail: dict, critical: bool = True) -> dict:
    return {
        "id": name,
        "ok": bool(ok),
        "critical": critical,
        "status": "PASS" if ok else ("FAIL" if critical else "WARN"),
        "detail": detail,
    }


def p1_jina_small() -> dict:
    from jina_service import ensure_jina, stop_jina
    from nomic_metric import aboutness

    # Prefer public status over private helpers
    try:
        from jina_service import active_gguf_path  # type: ignore

        gguf = active_gguf_path()
    except Exception:
        gguf = None
        try:
            st = __import__("jina_service", fromlist=["jina_status"]).jina_status()
            g = (st.get("gguf") or st.get("model") or "") if isinstance(st, dict) else ""
            gguf = g or None
        except Exception:
            gguf = None
    # Bounded wait for port release after stop (not fixed sleep only)
    stop_jina()
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            import urllib.request

            urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=0.4)
            time.sleep(0.25)
        except Exception:
            break
    ej = ensure_jina(force_restart=True)
    # Confirmed running GGUF from ensure (not private candidates)
    confirmed = ej.get("gguf") or gguf
    is_small = bool(confirmed and "small" in str(confirmed).lower())
    # Sanitize host paths in evidence
    conf_s = str(confirmed) if confirmed else None
    if conf_s and (":" in conf_s or conf_s.startswith("/")):
        conf_s = Path(conf_s).name
    floor = aboutness(
        "E_ref meets production readiness criteria under measured audit.",
        "Carbonara uses guanciale, egg, pecorino, and black pepper.",
    )
    ceil = aboutness(
        "E_ref meets production readiness criteria under measured audit.",
        "Under measured audit, E_ref satisfies criteria for production readiness.",
    )
    cos_f, cos_c = floor.get("cosine"), ceil.get("cosine")
    rng = (
        round(float(cos_c) - float(cos_f), 4)
        if cos_f is not None and cos_c is not None
        else None
    )
    # small Q4 may sit below 0.70 ceil with mean pool; last-token should lift.
    # Pass: family jina, small gguf, floor usable, range usable (ceil can be 0.55+).
    ok = (
        bool(ej.get("ok"))
        and is_small
        and floor.get("family") == "jina"
        and cos_f is not None
        and cos_f < 0.40
        and cos_c is not None
        and cos_c > 0.55
        and rng is not None
        and rng > 0.35
        and int(ej.get("dim") or 0) >= 768
    )
    return gate(
        "P1_jina_v5_small",
        ok,
        {
            "gguf": conf_s,
            "is_small": is_small,
            "ensure": {
                "ok": ej.get("ok"),
                "status": ej.get("status"),
                "dim": ej.get("dim"),
                "gguf": conf_s,
            },
            "floor": cos_f,
            "ceil": cos_c,
            "range": rng,
            "family": floor.get("family"),
            "model": floor.get("model"),
        },
    )


def p2_preserve_smoke() -> dict:
    """Load frankenstein alone briefly; do not full identity chain (long)."""
    from residency import seamless_substrate
    from truth_plane import frankenstein_loaded, frankenstein_required

    t0 = time.time()
    sub: dict = {}
    frank: dict = {}
    fiber: dict = {}
    scout: dict = {}
    ok = False
    err: str | None = None
    try:
        sub = seamless_substrate(fiber_mode="preserve")
        frank = frankenstein_loaded()
        fiber = sub.get("fiber") or {}
        model = (fiber.get("model") or "").lower()
        ok = (
            bool(sub.get("ok"))
            and sub.get("preserve_alone") is True
            and "frankenstein" in model
            and frank.get("loaded") is True
            and frankenstein_required("preserve")
        )
    except Exception as e:
        err = str(e)
    finally:
        # Always restore scout fiber (even on intermediate raise)
        try:
            scout = seamless_substrate(fiber_mode="scout")
        except Exception as e:
            scout = {"error": str(e)}
    if err:
        return gate(
            "P2_preserve_smoke",
            False,
            {"error": err, "restored_scout": scout},
        )
    return gate(
        "P2_preserve_smoke",
        ok,
        {
            "preserve_fiber": fiber,
            "frankenstein": frank,
            "substrate_errors": sub.get("errors"),
            "preserve_alone": sub.get("preserve_alone"),
            "restored_scout": {
                "ok": (scout.get("ok") if isinstance(scout, dict) else False),
                "fiber": (scout.get("fiber") or {}).get("model")
                if isinstance(scout, dict)
                else None,
            },
            "seconds": round(time.time() - t0, 1),
            "note": "full identity chain already measured p=0.875; this is residency smoke",
        },
    )


def p3_truth_enter() -> dict:
    from truth_plane import request_plane

    env_keys = ("PRIME_TRUTH_LOOP", "PRIME_FAST_ENTER", "PRIME_FIBER_MODE")
    prior = {k: os.environ.get(k) for k in env_keys}
    try:
        os.environ["PRIME_TRUTH_LOOP"] = "1"
        os.environ["PRIME_FAST_ENTER"] = "1"
        os.environ["PRIME_FIBER_MODE"] = "scout"
        card = request_plane(
            "Aboutness must not promote OPEN; NLI owns agreement. "
            "Refuse force-OPEN without mutual entailment.",
            mode="scout",
            domain="technology",
            truth_loop_enabled=True,
        )
        face = (card.get("cert_face") or {}).get("face")
        op = card.get("operator_summary") or {}
        # must not production OPEN; CANDIDATE or NEED_INFO or STOP ok
        ok = face in ("NEED_INFO", "STOP", "OPEN_CANDIDATE") and face != "OPEN"
        if face == "OPEN_CANDIDATE":
            ok = ok and bool(
                op.get("nli_agrees") or (card.get("agreement") or {}).get("agrees")
            )
        return gate(
            "P3_truth_plane_enter",
            ok,
            {
                "face": face,
                "operator": op,
                "nli": (card.get("agreement") or {}).get("label"),
                "nli_engine": (card.get("agreement") or {}).get("engine")
                or op.get("nli_engine"),
                "mutual_gate": (card.get("mutual_agreement") or {}).get("gate"),
                "truth_loop": bool(card.get("truth_loop")),
                "fiber_mode": card.get("fiber_mode"),
                "elapsed_s": card.get("elapsed_s"),
            },
        )
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def p4_negative_open_pack() -> dict:
    """Structural pack that would force-OPEN via cos must STOP at cert_face."""
    from dual_enter import cert_face
    from entailment_glue import glue_agreement

    # adversarial twin as "domain evidence"
    human = "Protect confidential records under policy controls."
    domain = "attacks: violate confidentiality rules and leak records."
    nli = glue_agreement(human, domain, prefer="auto")
    face = cert_face(
        verdict="OPEN",  # scout tries OPEN
        agreement=nli,
        aboutness={"mean_cosine": 0.92},
        fatal=False,
        regime="structured_ops",
        prompt_preview=human,
    )
    ok = face.get("face") == "STOP" and nli.get("label") == "contradiction"
    return gate(
        "P4_negative_force_open",
        ok,
        {
            "scout_verdict_in": "OPEN",
            "face": face.get("face"),
            "nli": nli.get("label"),
            "engine": nli.get("engine"),
            "conf": nli.get("confidence"),
            "aboutness_cos_injected": 0.92,
        },
    )


def p5_ort_hot() -> dict:
    from accel_nli_ort import load_session, predict

    st = load_session(force_cpu=True)
    if not st.get("ok"):
        return gate("P5_ort_hot", False, st, critical=False)
    r = predict(
        "E_ref is production-ready and certified OPEN.",
        "E_ref is not production-ready; residue remains.",
    )
    ok = r.get("ok") and r.get("label") == "contradiction" and r.get("engine") == "ort_nli"
    return gate(
        "P5_ort_hot",
        ok,
        {
            "session": {
                "provider": st.get("active_provider"),
                "onnx_mb": st.get("onnx_mb"),
            },
            "predict": {
                "label": r.get("label"),
                "conf": r.get("confidence"),
                "ms": r.get("latency_ms"),
                "engine": r.get("engine"),
            },
        },
        critical=False,
    )


def p6_qnn_probe() -> dict:
    import onnxruntime as ort

    prov = ort.get_available_providers()
    has_qnn = any("QNN" in p or "qnn" in p for p in prov)
    # Do not assert unprobed silicon — only report measured EP presence
    return gate(
        "P6_hexagon_qnn",
        has_qnn,  # will WARN if false
        {
            "providers": prov,
            "qnn_ep_registered": has_qnn,
            "hexagon_hw_probed": False,
            "hexagon_present_hw": None,  # unknown without HTP profile / register
            "qnn_ep": has_qnn,
            "residual": (
                None
                if has_qnn
                else "QNNExecutionProvider not in available providers on this build"
            ),
        },
        critical=False,
    )


def main() -> int:
    t0 = time.time()
    cells = []
    for fn in (
        p1_jina_small,
        p6_qnn_probe,
        p5_ort_hot,
        p4_negative_open_pack,
        p2_preserve_smoke,
        p3_truth_enter,
    ):
        print(f"running {fn.__name__}…", flush=True)
        try:
            c = fn()
        except Exception as e:
            # Preserve per-probe criticality (p6 residual is non-critical)
            crit = getattr(fn, "vv_critical", fn.__name__ != "p6_qnn_probe")
            c = gate(fn.__name__, False, {"error": str(e)[:400]}, critical=bool(crit))
        cells.append(c)
        print(f"  → {c['status']}", flush=True)

    n_pass = sum(1 for c in cells if c.get("status") == "PASS")
    n_warn = sum(1 for c in cells if c.get("status") == "WARN")
    # FAIL only — WARN is residual path, not n_fail
    n_fail = sum(1 for c in cells if c.get("status") == "FAIL")
    n_crit = sum(1 for c in cells if c.get("critical") and c.get("status") == "FAIL")
    n_other = sum(
        1
        for c in cells
        if c.get("status") not in ("PASS", "WARN", "FAIL")
    )
    # Integrity without assert: count_ok feeds report.ok / go_no_go (persist NO_GO)
    count_ok = (n_pass + n_warn + n_fail == len(cells)) and n_other == 0
    law_ok = n_crit == 0 and count_ok
    report = {
        "ok": law_ok,
        "go_no_go": "GO_MEASURE" if law_ok else "NO_GO",
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_other_status": n_other,
        "n_cells": len(cells),
        "n_critical_fail": n_crit,
        "count_rule": "WARN does not increment n_fail; n_pass+n_warn+n_fail==n_cells",
        "count_ok": count_ok,
        "seconds": round(time.time() - t0, 1),
        "cells": cells,
        "law": "aboutness must not promote OPEN; NLI owns agreement; residue never forced",
    }
    out = STATE / "vv_push_domains.json"
    STATE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "ok",
                    "go_no_go",
                    "n_pass",
                    "n_warn",
                    "n_fail",
                    "n_critical_fail",
                    "seconds",
                )
            },
            indent=2,
        )
    )
    for c in cells:
        print(f"  {c['status']:4} {c['id']}")
    print("wrote", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
