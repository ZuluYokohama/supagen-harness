#!/usr/bin/env python3
"""
Measure fabric — device routing for dual-metric instruments.

Law
---
- Job1 aboutness / Job1.5 rerank / Job2 agreement are *measure* planes.
- Production OPEN is never authorized by this module (cert_face + external audit).
- Hexagon HTP may run QDQ measure graphs only when an explicit parity certificate
  is present and green. Session-ready QDQ is *not* enough (E3 residual).

Prefer order for Job2 agreement (product path):
  ort_cpu → cross_encoder → lfm
HTP is inserted *only* when nli_htp_parity_pass() is True.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state"
PARITY_CERT = STATE / "npu" / "nli_htp_parity_cert.json"


def nli_htp_parity_pass(*, max_age_h: float = 168.0) -> dict[str, Any]:
    """
    Explicit E3 gate. Requires tracked local cert with:
      ok=true, hits/n >= min_hit_rate, recipe, timestamp
    Absent or stale → False (CPU remains agreement authority).
    """
    out: dict[str, Any] = {
        "ok": False,
        "path": str(PARITY_CERT),
        "reason": "missing",
    }
    if not PARITY_CERT.is_file():
        return out
    try:
        import time

        cert = json.loads(PARITY_CERT.read_text(encoding="utf-8"))
        age_h = (time.time() - PARITY_CERT.stat().st_mtime) / 3600.0
        min_rate = float(os.environ.get("PRIME_NLI_HTP_MIN_HIT", "0.9"))
        hits = int(cert.get("hits") or cert.get("label_parity_n") or 0)
        n = int(cert.get("n") or cert.get("label_parity_den") or 0)
        rate = float(
            cert.get("label_parity_rate")
            if cert.get("label_parity_rate") is not None
            else ((hits / n) if n else 0.0)
        )
        # Fail-closed: green only with complete same-run evidence fields
        required_ok = (
            bool(cert.get("ok"))
            and rate >= min_rate
            and age_h <= max_age_h
            and cert.get("job2_owns_open") is not True
            and cert.get("cpu_fallback") is False
            and bool(cert.get("qnn_ep_registered") or cert.get("on_qnn"))
            and bool(cert.get("held_out") or cert.get("label_parity_rate") is not None)
            and bool(cert.get("recipe") or cert.get("model_id"))
            and not bool(cert.get("probe_only") or cert.get("strict_qnn_failed"))
            and not bool(cert.get("uncalibrated_probe"))
        )
        green = required_ok
        out.update(
            {
                "ok": green,
                "age_h": round(age_h, 2),
                "hits": hits,
                "n": n,
                "hit_rate": round(rate, 3),
                "recipe": cert.get("recipe"),
                "verdict": cert.get("verdict"),
                "cpu_fallback": cert.get("cpu_fallback"),
                "held_out": cert.get("held_out"),
                "reason": "pass" if green else "cert_not_green_or_incomplete",
            }
        )
        return out
    except Exception as e:
        out["reason"] = f"read_error:{e}"
        return out


def route_job2() -> dict[str, Any]:
    """Declare the agreement backend order for this host."""
    parity = nli_htp_parity_pass()
    order = ["ort_cpu", "cross_encoder", "lfm"]
    if parity.get("ok"):
        order = ["htp_qdq"] + order
    return {
        "job": "agreement_nli",
        "order": order,
        "htp_parity": parity,
        "job2_owns_open": False,
        "law": "agreement measure only; production OPEN needs domain audit + cert_face",
    }


def route_job1() -> dict[str, Any]:
    return {
        "job": "aboutness_embed",
        "primary": "jina_side_8765",
        "fallback": "nomic_degraded",
        "device_today": "cpu_llama_server",
        "npu": "not_product_path",
        "owns_open": False,
    }


def fabric_status() -> dict[str, Any]:
    try:
        from npu_qnn import register

        npu = register()
    except Exception as e:
        npu = {"ok": False, "error": str(e)}
    return {
        "job1": route_job1(),
        "job2": route_job2(),
        "npu_register": {
            "ok": npu.get("ok"),
            "n_qnn_devices": npu.get("n_qnn_devices"),
            "htp_exists": npu.get("htp_exists"),
            "htp_dll": npu.get("htp_dll"),
        },
        "parity_cert_path": str(PARITY_CERT),
        "law": (
            "aboutness must not promote OPEN; NLI owns agreement; "
            "residue never forced; HTP Job2 only after E3 parity cert"
        ),
    }


def write_parity_cert_from_report(report_path: Path | str) -> dict[str, Any]:
    """
    Convert npu_nli_qdq report → parity cert.
    Green only with held-out ORT label_parity_rate + no CPU fallback probe.
    Never sets job2_owns_open.
    """
    from datetime import datetime, timezone

    report_path = Path(report_path)
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    run = rep.get("run") or {}
    hits = int(run.get("label_parity_n") or run.get("hits") or 0)
    n = int(run.get("label_parity_den") or run.get("n") or 0)
    min_rate = float(os.environ.get("PRIME_NLI_HTP_MIN_HIT", "0.9"))
    rate = float(
        run.get("label_parity_rate")
        if run.get("label_parity_rate") is not None
        else ((hits / n) if n else 0.0)
    )
    probe_only = bool(run.get("probe_only") or run.get("strict_qnn_failed"))
    cpu_fallback = probe_only or bool(run.get("cpu_fallback"))
    green = bool(
        run.get("ok")
        and rate >= min_rate
        and hits >= 2
        and not probe_only
        and not cpu_fallback
        and bool(run.get("held_out") or run.get("label_parity_rate") is not None)
        and bool(run.get("qnn_ep_registered") or run.get("on_qnn"))
    )
    cert = {
        "ok": green,
        "hits": hits,
        "n": n,
        "hit_rate": round(rate, 3),
        "label_parity_rate": round(rate, 3),
        "label_parity_n": hits,
        "label_parity_den": n,
        "held_out": bool(run.get("held_out")),
        "recipe": rep.get("recipe")
        or {
            "act": (rep.get("quantize") or {}).get("act"),
            "weight": (rep.get("quantize") or {}).get("weight"),
        },
        "model_id": (rep.get("export") or {}).get("model_id")
        or "cross-encoder/nli-deberta-v3-base",
        "verdict": rep.get("verdict"),
        "qnn_ep_registered": bool(run.get("qnn_ep_registered") or run.get("on_qnn")),
        "on_qnn": bool(run.get("on_qnn") or run.get("qnn_ep_registered")),
        "cpu_fallback": cpu_fallback,
        "probe_only": probe_only,
        "strict_qnn_failed": bool(run.get("strict_qnn_failed")),
        "uncalibrated_probe": not green,
        "job2_owns_open": False,
        "measured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_report": "<repo>/docs/evidence/npu/ or prime/state/npu_nli_qdq_report.json",
        "law": "parity cert is measure-only; never production OPEN authority",
    }
    PARITY_CERT.parent.mkdir(parents=True, exist_ok=True)
    PARITY_CERT.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    return cert


if __name__ == "__main__":
    print(json.dumps(fabric_status(), indent=2, default=str))
