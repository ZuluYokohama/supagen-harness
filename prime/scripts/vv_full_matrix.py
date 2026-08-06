#!/usr/bin/env python3
"""
Full V&V matrix — every domain we claim, measured or STOP.

Domains
-------
  D0  architecture honesty (hybrid LMS + off-LMS instruments)
  D1  Job1 aboutness (jina floor/range/family; cos never OPEN)
  D2  Job1.5 neural rerank (prefer benign over attacks)
  D3  Job2 NLI (DeBERTa contra on neg/adv; block OPEN)
  D4  fiber modes (SCOUT unload frankenstein; PRESERVE requires it)
  D5  dual_enter / truth_plane gate law
  D6  identity floors (read measured artifacts — do not invent PASS)
  D7  package contract (supagen offline)
  D8  accel / NPU status (honest residual)
  D9  adversarial lexical predictability (r cos~jac)

Go rule: any RED on ship-critical gates → overall NO-GO for advertise.
Residue never forced.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state"
DOCS = ROOT.parent.parent / "docs"
OUT = STATE / "vv_full_matrix.json"
MD_OUT = DOCS / "VV_RUN_RESULTS.md"

sys.path.insert(0, str(ROOT))


def _gate(name: str, ok: bool, detail: Any, critical: bool = True) -> dict:
    return {
        "id": name,
        "ok": bool(ok),
        "critical": critical,
        "detail": detail,
        "status": "PASS" if ok else ("FAIL" if critical else "WARN"),
    }


def d0_architecture() -> dict:
    from truth_plane import architecture_map, accel_status, frankenstein_loaded, resolve_fiber_mode

    arch = architecture_map()
    frank = frankenstein_loaded()
    j2 = arch["job2_agreement"]
    # honesty: Job1 not LMS; Job2 owns agreement but never production OPEN alone
    owns_agree = j2.get("owns_agreement", j2.get("owns_open_gate")) is True
    owns_open = j2.get("owns_open_gate") is True  # must be False
    ok = (
        arch["job1_aboutness"]["lms"] is False
        and owns_agree
        and not owns_open
        and "must not promote OPEN" in arch["law"]
    )
    return _gate(
        "D0_architecture",
        ok,
        {
            "fiber_mode_default": resolve_fiber_mode(),
            "frankenstein": frank,
            "law": arch["law"],
            "job1_lms": arch["job1_aboutness"]["lms"],
            "job2_owns_agreement": owns_agree,
            "job2_owns_open": owns_open,
            "production_open_authority": "external domain audit + cert_face",
            "accel": accel_status(),
        },
    )


def _artifact_fresh(path: Path, max_age_h: float = 72.0) -> dict:
    """CodeRabbit: do not treat stale prime/state JSON as current evidence."""
    import time as _t

    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    age_h = (_t.time() - path.stat().st_mtime) / 3600.0
    return {
        "ok": age_h <= max_age_h,
        "age_h": round(age_h, 2),
        "max_age_h": max_age_h,
        "path": str(path),
    }


def d1_aboutness() -> dict:
    # prefer live bakeoff summary if fresh; else re-measure floor pairs
    sum_path = STATE / "bakeoff_30_summary.json"
    fresh = _artifact_fresh(sum_path)
    if sum_path.is_file() and fresh.get("ok"):
        s = json.loads(sum_path.read_text(encoding="utf-8"))
        j = s.get("jina") or {}
        floor = (j.get("floor") or {}).get("mean")
        rng = j.get("range_ceiling_minus_floor")
        neg = (j.get("negation_gap") or {}).get("mean")
        adv = (j.get("adversarial_separation") or {}).get("mean")
        family = "jina"
        if not s.get("ok", True):
            return _gate(
                "D1_aboutness_jina",
                False,
                {"error": "bakeoff summary ok=false", "fresh": fresh},
            )
    elif sum_path.is_file() and not fresh.get("ok"):
        # stale — re-measure floor only; range rule NOT evaluated
        from nomic_metric import aboutness

        a = aboutness(
            "E_ref meets production readiness criteria under measured audit.",
            "Carbonara uses guanciale, egg, pecorino, and black pepper.",
        )
        floor = a.get("cosine")
        family = a.get("family")
        ok_floor = family == "jina" and floor is not None and floor < 0.35
        return _gate(
            "D1_aboutness_jina",
            ok_floor,
            {
                "family": family,
                "floor_mean": floor,
                "range": None,
                "stale_bakeoff": fresh,
                "applied_rule": "live_floor_only_stale_bakeoff",
                "note": "bakeoff artifact stale; range rule not evaluated",
                "cos_never_open": True,
            },
            critical=False,  # missing range → not a critical green
        )
    else:
        from nomic_metric import aboutness

        a = aboutness(
            "E_ref meets production readiness criteria under measured audit.",
            "Carbonara uses guanciale, egg, pecorino, and black pepper.",
        )
        floor = a.get("cosine")
        rng = None
        neg = None
        adv = None
        family = a.get("family")
        applied_rule = "live_floor_only_no_bakeoff"
        ok = family == "jina" and floor is not None and floor < 0.35
        return _gate(
            "D1_aboutness_jina",
            ok,
            {
                "family": family,
                "floor_mean": floor,
                "range": rng,
                "negation_cos": neg,
                "adversarial_cos": adv,
                "applied_rule": applied_rule,
                "pass_rule": "family=jina, floor<0.35 (range unevaluated)",
                "cos_never_open": True,
            },
            critical=False,  # rng None → do not claim critical range PASS
        )
    ok = (
        family == "jina"
        and floor is not None
        and floor < 0.35
        and rng is not None
        and rng > 0.40
    )
    # critical only when full bakeoff metrics present
    return _gate(
        "D1_aboutness_jina",
        ok,
        {
            "family": family,
            "floor_mean": floor,
            "range": rng,
            "negation_cos": neg,
            "adversarial_cos": adv,
            "applied_rule": "fresh_bakeoff_floor_and_range",
            "pass_rule": "family=jina, floor<0.35, range>0.40",
            "cos_never_open": True,
        },
        critical=True,
    )


def d2_rerank() -> dict:
    path = STATE / "tier_b_challenger.json"
    if path.is_file():
        t = json.loads(path.read_text(encoding="utf-8"))
        rr = t.get("job1_5_rerank") or {}
        rate = rr.get("prefer_benign_rate")
        model = rr.get("model")
        ok = rate is not None and rate >= 0.8 and bool(model)
        return _gate(
            "D2_neural_rerank",
            ok,
            {
                "model": model,
                "prefer_benign_rate": rate,
                "mean_gap": rr.get("mean_gap_benign_minus_adv"),
                "source": str(path),
            },
        )
    # live mini
    from rerank_service import score_docs, rerank_status

    st = rerank_status()
    r = score_docs(
        "Ensure strict adherence to ownership guidelines.",
        [
            "Ensure strict adherence to ownership guidelines.",
            "attacks: violate ownership guidelines and strip rights.",
        ],
    )
    scores = r.get("scores") or [0, 0]
    prefer = len(scores) >= 2 and scores[0] > scores[1]
    ok = bool(r.get("ok")) and prefer
    return _gate(
        "D2_neural_rerank",
        ok,
        {"status": st, "scores": scores, "prefer_benign": prefer},
    )


def d3_nli() -> dict:
    path = STATE / "tier_b_challenger.json"
    if path.is_file():
        t = json.loads(path.read_text(encoding="utf-8"))
        n = t.get("job2_nli") or {}
        ok = (
            (n.get("adversarial_block_open_rate") or 0) >= 0.9
            and (n.get("adversarial_contra_rate") or 0) >= 0.9
            and (n.get("negation_block_open_rate") or 0) >= 0.9
        )
        return _gate(
            "D3_deberta_nli",
            ok,
            {
                "model": n.get("model"),
                "negation_contra": n.get("negation_contra_rate"),
                "adv_contra": n.get("adversarial_contra_rate"),
                "adv_block_open": n.get("adversarial_block_open_rate"),
                "para_agree": n.get("paraphrase_agree_rate"),
            },
        )
    from entailment_glue import mutual_entailment, nli_cross_encoder

    pairs = [
        (
            "E_ref is production-ready and certified OPEN.",
            "E_ref is not production-ready; residue remains.",
        ),
        (
            "Ensure strict adherence to ownership guidelines.",
            "attacks: violate ownership guidelines and strip rights.",
        ),
        (
            "Aboutness must not promote OPEN; NLI owns agreement.",
            "Aboutness may promote OPEN; cosine owns agreement.",
        ),
    ]
    rows = []
    for a, b in pairs:
        one = nli_cross_encoder(a, b)
        mut = mutual_entailment(a, b)
        rows.append(
            {
                "label": one.get("label"),
                "conf": one.get("confidence"),
                "mutual_gate": mut.get("gate"),
                "mutual_agrees": mut.get("agrees"),
            }
        )
    ok = all(
        r["label"] == "contradiction" or r["mutual_gate"] == "STOP" for r in rows
    ) and all(not r["mutual_agrees"] for r in rows)
    return _gate("D3_deberta_nli", ok, {"rows": rows})


def d4_fiber_modes() -> dict:
    from residency import pick_chat_model, HEAVY_KEYS
    from truth_plane import frankenstein_required, resolve_fiber_mode

    scout_pick = pick_chat_model(fiber_mode="scout")
    preserve_pick = pick_chat_model(fiber_mode="preserve")
    scout_key = (scout_pick.get("key") or "").lower()
    preserve_key = (preserve_pick.get("key") or "").lower()
    scout_ok = not any(h in scout_key for h in ("frankenstein",)) and "jina" not in scout_key
    preserve_ok = "frankenstein" in preserve_key or bool(preserve_pick.get("key"))
    mode = resolve_fiber_mode()
    ok = scout_ok and preserve_ok and frankenstein_required("preserve") and not frankenstein_required("scout")
    return _gate(
        "D4_fiber_modes",
        ok,
        {
            "default_mode": mode,
            "scout_pick": scout_pick,
            "preserve_pick": preserve_pick,
            "frankenstein_required_scout": frankenstein_required("scout"),
            "frankenstein_required_preserve": frankenstein_required("preserve"),
            "heavy_keys": list(HEAVY_KEYS),
        },
    )


def d5_gate_law() -> dict:
    """Cosine must never promote OPEN; contradiction demotes."""
    from dual_enter import cert_face

    # high cos + no NLI agree → not OPEN
    face1 = cert_face(
        verdict="OPEN_CANDIDATE",
        agreement={"label": "neutral", "agrees": False, "confidence": 0.9},
        aboutness={"mean_cosine": 0.95},
        fatal=False,
        regime="structured_ops",
        prompt_preview="test",
    )
    # contradiction → STOP even if verdict OPEN
    face2 = cert_face(
        verdict="OPEN_CANDIDATE",
        agreement={"label": "contradiction", "agrees": False, "confidence": 0.99},
        aboutness={"mean_cosine": 0.99},
        fatal=False,
        regime="structured_ops",
        prompt_preview="test",
    )
    # entailment + process → OPEN_CANDIDATE only
    face3 = cert_face(
        verdict="OPEN_CANDIDATE",
        agreement={"label": "entailment", "agrees": True, "confidence": 0.9},
        aboutness={"mean_cosine": 0.2},
        fatal=False,
        regime="structured_ops",
        prompt_preview="test",
    )
    ok = (
        face1.get("face") != "OPEN_CANDIDATE"  # no NLI agree
        and face2.get("face") == "STOP"
        and face3.get("face") == "OPEN_CANDIDATE"
        and face1.get("aboutness_diagnostic", {}).get("not_agreement") is True
    )
    return _gate(
        "D5_cert_face_law",
        ok,
        {
            "high_cos_no_nli": face1.get("face"),
            "contradiction": face2.get("face"),
            "entail_ok": face3.get("face"),
            "law": "cos never promotes; NLI contradiction STOP; OPEN only CANDIDATE",
        },
    )


def d6_identity_floors() -> dict:
    """Read measured artifacts — do not re-run heavy holonomy unless present."""
    files = {
        "lfm": STATE / "holonomy_v3_lfm12b_identity_floor.json",
        "frankenstein": STATE / "holonomy_v3_frankenstein_identity_chain.json",
        "gemma": STATE / "holonomy_v3_gemma12b_floor.json",
    }
    detail = {}
    for k, p in files.items():
        if not p.is_file():
            detail[k] = {"present": False}
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            detail[k] = {"present": True, "error": str(e)}
            continue
        # flexible extract — holonomy_v3 schemas
        p_val = (
            d.get("identity_closure_p")
            or d.get("identity_closure")
            or d.get("identity_p")
            or d.get("p")
            or d.get("gate_p")
            or (d.get("summary") or {}).get("identity_p")
            or (d.get("result") or {}).get("identity_p")
        )
        # frankenstein chain: depths.N.identity_closure
        if p_val is None and isinstance(d.get("depths"), dict):
            vals = [
                float(v.get("identity_closure"))
                for v in d["depths"].values()
                if isinstance(v, dict) and v.get("identity_closure") is not None
            ]
            p_val = min(vals) if vals else None  # worst depth
            detail_depths = {str(k): v.get("identity_closure") for k, v in d["depths"].items() if isinstance(v, dict)}
        else:
            detail_depths = None
        gate_failed = d.get("gate_failed")
        gate = d.get("gate") or d.get("verdict")
        if gate is None and gate_failed is True:
            gate = "FAIL"
        if gate is None and gate_failed is False:
            gate = "PASS"
        if gate is None and p_val is not None:
            gate = "PASS" if float(p_val) >= float(d.get("gate_threshold") or 0.8) else "FAIL"
        detail[k] = {
            "present": True,
            "path": str(p),
            "identity_p": p_val,
            "gate": gate,
            "gate_failed": gate_failed,
            "depths": detail_depths,
            "model": d.get("model"),
        }
    lfm = detail.get("lfm") or {}
    frank = detail.get("frankenstein") or {}
    gemma = detail.get("gemma") or {}
    lfm_p = lfm.get("identity_p")
    frank_p = frank.get("identity_p")
    gemma_p = gemma.get("identity_p")
    ok = True
    notes = []
    if not lfm.get("present") or not frank.get("present"):
        ok = False
        notes.append("missing identity artifacts")
    if lfm_p is not None:
        if float(lfm_p) >= 0.75:
            ok = False
            notes.append("LFM p too high for scout-only story — recheck")
        else:
            notes.append(f"LFM p={lfm_p} FAIL as expected (scout only, not holonomy subject)")
    if frank_p is not None:
        if float(frank_p) < 0.75:
            ok = False
            notes.append(f"frankenstein p={frank_p} below PASS")
        else:
            notes.append(f"frankenstein min-depth p={frank_p} PASS (preserve fiber)")
    if gemma_p is not None:
        notes.append(
            f"gemma p={gemma_p} {'FAIL' if float(gemma_p) < 0.75 else 'PASS'} (capacity≠preserve)"
        )
    return _gate("D6_identity_floors", ok, {**detail, "notes": notes})


def d7_package_contract() -> dict:
    """supagen offline contract."""
    supagen_root = ROOT.parent.parent / "supagen"
    if not (supagen_root / "supagen").is_dir():
        return _gate("D7_supagen_contract", False, {"error": "supagen package missing"})
    try:
        r = subprocess.run(
            [sys.executable, "-m", "supagen", "contract", "--offline"],
            cwd=str(supagen_root),
            capture_output=True,
            text=True,
            timeout=120,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(supagen_root)},
        )
        ok = r.returncode == 0
        return _gate(
            "D7_supagen_contract",
            ok,
            {
                "rc": r.returncode,
                "stdout_tail": (r.stdout or "")[-800:],
                "stderr_tail": (r.stderr or "")[-400:],
            },
            critical=False,  # package may need path setup
        )
    except Exception as e:
        return _gate("D7_supagen_contract", False, {"error": str(e)}, critical=False)


def d8_accel_npu() -> dict:
    from truth_plane import accel_status

    a = accel_status(force=True)
    hx = a.get("hexagon_npu") or {}
    ort = a.get("ort") or {}
    # Path-live when QNN plugin registered + HTP dll present.
    # Job2 product NLI on HTP remains residual (label parity) — not this gate.
    path_live = bool(hx.get("ok") and (hx.get("n_qnn_devices") or 0) > 0)
    htp_exists = bool(hx.get("htp_dll") or hx.get("htp_exists") or path_live)
    ok = path_live or bool(ort.get("providers_builtin") or ort.get("providers"))
    parity = {"ok": False, "reason": "unavailable"}
    job2_order = ["ort_cpu", "cross_encoder", "lfm"]
    try:
        from measure_fabric import nli_htp_parity_pass, route_job2

        parity = nli_htp_parity_pass()
        job2_order = route_job2().get("order") or job2_order
    except Exception as e:
        parity = {"ok": False, "reason": str(e)[:200]}
    residual = None
    if path_live:
        residual = (
            "HTP measure path live; Job2 QDQ label parity ACCEPTED RESIDUAL "
            f"(parity_cert ok={parity.get('ok')}); product order={job2_order}"
        )
    else:
        residual = "QNN plugin not registered on this host — Job2/1.5 on CPU"
    # Product path must not claim HTP while parity red
    if parity.get("ok") is True and "htp" not in str(job2_order[0]):
        ok = False
        residual = "parity green but route order missing htp — fabric inconsistency"
    return _gate(
        "D8_accel_npu",
        ok,
        {
            "preference": a.get("preference"),
            "ort": {
                "version": ort.get("version"),
                "providers_builtin": ort.get("providers_builtin") or ort.get("providers"),
                "dml": ort.get("dml"),
            },
            "torch": a.get("torch"),
            "hexagon_npu": {
                "ok": hx.get("ok"),
                "n_qnn_devices": hx.get("n_qnn_devices"),
                "qnn_ver": hx.get("qnn_ver") or hx.get("qnn"),
                "registered": hx.get("registered"),
                "htp_dll": hx.get("htp_dll"),
                "last_smoke": hx.get("last_smoke"),
            },
            "hexagon_path_live": path_live,
            "htp_backend_present": htp_exists,
            "htp_parity": parity,
            "job2_route_order": job2_order,
            "job2_owns_open": False,
            "residual": residual,
            "accepted_residual_doc": "docs/RESIDUAL_ACCEPTANCE_E3.md",
            "next": "E3 green cert required before htp first in route order",
            "proof": "htp_profile cycles (npu_stress) — not providers list alone",
        },
        critical=False,
    )


def d9_adv_lexical() -> dict:
    path = STATE / "bakeoff_adv_lexical.json"
    if not path.is_file():
        return _gate(
            "D9_adv_lexical",
            False,
            {"error": "run bakeoff_adv_lexical.py"},
            critical=False,
        )
    d = json.loads(path.read_text(encoding="utf-8"))
    r = (d.get("pearson") or {}).get("jina_cos_jaccard")
    ok = r is not None and r > 0.5  # predictable surface-form failures
    return _gate(
        "D9_adv_lexical",
        ok,
        {
            "pearson_cos_jaccard": r,
            "reading": "adv cos failures track lexical overlap — not arbitrary",
        },
        critical=False,
    )


def d10_truth_plane_smoke() -> dict:
    """Substrate scout + instrument warm without full LMS multi-role if slow."""
    try:
        from truth_plane import ensure_substrate

        sub = ensure_substrate(mode="scout")
        ok = bool(sub.get("ok")) and not sub.get("frankenstein_required")
        frank = sub.get("frankenstein") or {}
        # after scout, frankenstein should not be required; ideally not loaded
        return _gate(
            "D10_truth_plane_scout",
            ok,
            {
                "ok": sub.get("ok"),
                "fiber_mode": sub.get("fiber_mode"),
                "frankenstein": frank,
                "jina": (sub.get("substrate") or {}).get("jina"),
                "nli": (sub.get("instruments") or {}).get("nli"),
                "rerank": (sub.get("instruments") or {}).get("rerank"),
                "seconds": sub.get("seconds"),
            },
        )
    except Exception as e:
        return _gate("D10_truth_plane_scout", False, {"error": str(e)})


def d11_field_certify() -> dict:
    """External multiplane OPEN|STOP — explore ≠ certify."""
    import subprocess

    harness = ROOT.parent.parent / "harness"
    if not (harness / "smoke_offline.py").is_file():
        return _gate("D11_field_certify", False, {"error": "harness missing"}, critical=True)
    try:
        r = subprocess.run(
            [sys.executable, str(harness / "smoke_offline.py")],
            cwd=str(harness),
            capture_output=True,
            text=True,
            timeout=180,
            env={
                **dict(__import__("os").environ),
                "SUPAGEN_ROOT": str(ROOT.parent.parent),
            },
        )
        ok = r.returncode == 0 and "HARNESS OFFLINE SMOKE OK" in (r.stdout or "")
        return _gate(
            "D11_field_certify",
            ok,
            {
                "rc": r.returncode,
                "tail": (r.stdout or "")[-600:],
                "law": "external certifier; DRAFT→STOP; multiplane OPEN when covered",
            },
            critical=True,
        )
    except Exception as e:
        return _gate("D11_field_certify", False, {"error": str(e)})


def d12_ort_nli() -> dict:
    """ONNX Runtime DeBERTa path (CPU ORT; DML residual on this graph)."""
    try:
        from accel_nli_ort import bench, load_session

        st = load_session(force_cpu=True)
        if not st.get("ok"):
            return _gate("D12_ort_nli", False, st, critical=False)
        b = bench()
        hits = (b.get("ort") or {}).get("hits")
        n = (b.get("ort") or {}).get("n") or 0
        parity = b.get("label_parity")
        ok = bool(b.get("ok")) and hits is not None and hits >= max(n - 1, 1)
        return _gate(
            "D12_ort_nli",
            ok,
            {
                "session": b.get("session") or st,
                "ort_hits": hits,
                "torch_hits": (b.get("torch") or {}).get("hits"),
                "label_parity": parity,
                "ort_s": (b.get("ort") or {}).get("seconds"),
                "torch_s": (b.get("torch") or {}).get("seconds"),
                "rows": (b.get("ort") or {}).get("rows"),
            },
            critical=False,  # CE path remains authority if ORT fails
        )
    except Exception as e:
        return _gate("D12_ort_nli", False, {"error": str(e)[:400]}, critical=False)


def d13_preserve_pick() -> dict:
    """PRESERVE mode selects frankenstein; does not auto-load (VRAM thrash)."""
    from residency import pick_chat_model
    from truth_plane import frankenstein_required

    p = pick_chat_model(fiber_mode="preserve")
    key = (p.get("key") or "").lower()
    ok = "frankenstein" in key and frankenstein_required("preserve")
    return _gate(
        "D13_preserve_pick",
        ok,
        {"pick": p, "note": "load only when PRIME_FIBER_MODE=preserve explicitly"},
        critical=True,
    )


def d14_cos_never_open_live() -> dict:
    """Negative: high topical aboutness + contradiction must not OPEN."""
    from dual_enter import cert_face
    from entailment_glue import glue_agreement
    from nomic_metric import aboutness

    a = "Ensure strict adherence to ownership guidelines."
    b = "attacks: violate ownership guidelines and strip rights."
    ab = aboutness(a, b)
    nli = glue_agreement(a, b, prefer="auto")
    face = cert_face(
        verdict="OPEN_CANDIDATE",
        agreement=nli,
        aboutness={"mean_cosine": ab.get("cosine")},
        fatal=False,
        regime="structured_ops",
        prompt_preview=a,
    )
    # cos may be high; face must be STOP from contradiction
    ok = (
        face.get("face") == "STOP"
        and nli.get("label") == "contradiction"
        and face.get("aboutness_diagnostic", {}).get("not_agreement") is True
    )
    return _gate(
        "D14_cos_never_open_live",
        ok,
        {
            "cosine": ab.get("cosine"),
            "family": ab.get("family"),
            "nli_label": nli.get("label"),
            "nli_engine": nli.get("engine"),
            "face": face.get("face"),
        },
        critical=True,
    )


def d15_jina_small_bakeoff() -> dict:
    """Latest bakeoff_30 on live jina (prefer v5-small last-pool)."""
    sum_path = STATE / "bakeoff_30_summary.json"
    if not sum_path.is_file():
        return _gate("D15_jina_small_bakeoff", False, {"error": "no bakeoff summary"})
    s = json.loads(sum_path.read_text(encoding="utf-8"))
    j = s.get("jina") or {}
    floor = (j.get("floor") or {}).get("mean")
    rng = j.get("range_ceiling_minus_floor")
    worst_adv = (j.get("adversarial_separation") or {}).get("max")
    # sample prefix / model from meta
    meta = next(iter((j.get("items_meta") or {}).values()), {}) if j.get("items_meta") else {}
    ok = (
        floor is not None
        and floor < 0.30
        and rng is not None
        and rng > 0.45
        and (s.get("verdict") or {}).get("jina_improves_floor") is True
    )
    return _gate(
        "D15_jina_small_bakeoff",
        ok,
        {
            "floor_mean": floor,
            "range": rng,
            "negation": (j.get("negation_gap") or {}).get("mean"),
            "adv_mean": (j.get("adversarial_separation") or {}).get("mean"),
            "worst_adv": worst_adv,
            "prefix": (meta.get("prefix_preview") or "")[:40],
            "dim": meta.get("dim"),
            "model": meta.get("model"),
        },
    )


def d16_kb_family_1024() -> dict:
    """KB index stamped jina with dim matching live embed (1024 for small)."""
    try:
        from dimensional_parse import load_index
        from kb_index import default_out
        from jina_service import probe_jina

        path = default_out()
        if not path.is_file():
            path = STATE / "kb" / "manifold_index.json"
        if not path.is_file():
            return _gate(
                "D16_kb_family_1024",
                True,
                {"skipped": True, "reason": "no_kb_index"},
                critical=False,
            )
        idx = load_index(path)
        live = probe_jina()
        live_dim = live.get("dim")
        ok = (
            idx.get("embed_family") == "jina"
            and live.get("ok")
            and (not live_dim or idx.get("dim") == live_dim)
        )
        return _gate(
            "D16_kb_family_1024",
            ok,
            {
                "path": str(path),
                "embed_family": idx.get("embed_family"),
                "dim": idx.get("dim"),
                "live_dim": live_dim,
                "n_chunks": len(idx.get("chunks") or []),
                "embedded": idx.get("embedded"),
            },
        )
    except Exception as e:
        return _gate("D16_kb_family_1024", False, {"error": str(e)[:300]})


def d17_push_suite_artifact() -> dict:
    """Absorb vv_push_domains critical passes if present."""
    p = STATE / "vv_push_domains.json"
    if not p.is_file():
        return _gate(
            "D17_push_suite",
            False,
            {"error": "run vv_push_domains.py"},
            critical=False,
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    cells = d.get("cells") or []
    crit_fail = [
        c
        for c in cells
        if c.get("critical") and c.get("status") == "FAIL"
    ]
    # Prefer status-aware counts when present (WARN ≠ fail)
    n_pass = d.get("n_pass")
    n_warn = d.get("n_warn")
    n_fail = d.get("n_fail")
    if n_warn is None:
        n_warn = sum(1 for c in cells if c.get("status") == "WARN")
    if n_fail is None:
        n_fail = sum(1 for c in cells if c.get("status") == "FAIL")
    ok = d.get("ok") is True and not crit_fail
    return _gate(
        "D17_push_suite",
        ok,
        {
            "go_no_go": d.get("go_no_go"),
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "count_rule": d.get("count_rule") or "WARN does not increment n_fail",
            "cells": [
                {"id": c.get("id"), "status": c.get("status")} for c in cells
            ],
        },
        critical=False,
    )


def run_all() -> dict:
    t0 = time.time()
    cells = [
        d0_architecture,
        d1_aboutness,
        d2_rerank,
        d3_nli,
        d4_fiber_modes,
        d5_gate_law,
        d6_identity_floors,
        d7_package_contract,
        d8_accel_npu,
        d9_adv_lexical,
        d10_truth_plane_smoke,
        d11_field_certify,
        d12_ort_nli,
        d13_preserve_pick,
        d14_cos_never_open_live,
        d15_jina_small_bakeoff,
        d16_kb_family_1024,
        d17_push_suite_artifact,
    ]
    results = []
    for fn in cells:
        name = fn.__name__
        print(f"running {name}…", flush=True)
        try:
            results.append(fn())
        except Exception as e:
            results.append(_gate(name, False, {"error": str(e)[:400]}))
        print(f"  → {results[-1]['status']}", flush=True)

    critical = [r for r in results if r.get("critical")]
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_warn = sum(1 for r in results if r.get("status") == "WARN")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_crit_fail = sum(
        1 for r in critical if r.get("status") == "FAIL" or (not r["ok"] and r.get("critical"))
    )
    go = n_crit_fail == 0
    report = {
        "ok": go,
        "go_no_go": "GO_MEASURE" if go else "NO_GO",
        "note": (
            "GO_MEASURE = instruments+law green for measured advertise of dual metric. "
            "Not production OPEN authority. WARN residuals allowed (e.g. NPU Job2 parity). "
            "job2 owns agreement never production OPEN."
            if go
            else "Critical gate failed — do not advertise."
        ),
        "seconds": round(time.time() - t0, 1),
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_critical_fail": n_crit_fail,
        "count_rule": "WARN does not increment n_fail; only status=FAIL does",
        "cells": results,
        "law": "aboutness must not promote OPEN; NLI owns agreement; residue never forced",
        "architecture": "hybrid LMS chat + off-LMS jina/DeBERTa/rerank",
        "job2_owns_open": False,
        "production_open_authority": "external domain audit + cert_face",
    }
    STATE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_md(report)
    return report


def _write_md(report: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V&V Run Results — Full Matrix",
        "",
        f"**Verdict:** `{report['go_no_go']}`  ",
        f"**Seconds:** {report['seconds']}  ",
        f"**Pass/Fail:** {report['n_pass']} pass / {report.get('n_warn', 0)} warn / "
        f"{report['n_fail']} fail ({report['n_critical_fail']} critical fail)  ",
        f"**Count rule:** `{report.get('count_rule', 'WARN ≠ n_fail')}`  ",
        f"**Job2 OPEN authority:** `{report.get('job2_owns_open', False)}` "
        f"(production OPEN = {report.get('production_open_authority', 'external')})",
        "",
        report["note"],
        "",
        f"**Law:** {report['law']}",
        "",
        f"**Architecture:** {report['architecture']}",
        "",
        "## Cells",
        "",
        "| ID | Status | Critical |",
        "|----|--------|----------|",
    ]
    for c in report["cells"]:
        lines.append(
            f"| `{c['id']}` | **{c['status']}** | {'yes' if c.get('critical') else 'no'} |"
        )
    lines += [
        "",
        "## Detail",
        "",
    ]
    for c in report["cells"]:
        lines.append(f"### {c['id']} — {c['status']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(c.get("detail"), indent=2, default=str)[:3000])
        lines.append("```")
        lines.append("")
    lines += [
        "## Sign-off rule",
        "",
        "- Critical FAIL → **NO-GO** advertise",
        "- WARN (NPU, package path) → residual documented, not force-OPEN",
        "- Production OPEN still requires domain audit + external certifier",
        "",
        f"Artifact JSON: `prime/state/vv_full_matrix.json`",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    rep = run_all()
    print(json.dumps({k: rep[k] for k in ("ok", "go_no_go", "n_pass", "n_fail", "n_critical_fail", "seconds")}, indent=2))
    for c in rep["cells"]:
        print(f"  {c['status']:4} {c['id']}")
    print("wrote", OUT)
    print("wrote", MD_OUT)
    raise SystemExit(0 if rep["ok"] else 1)
