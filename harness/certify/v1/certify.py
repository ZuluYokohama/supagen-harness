#!/usr/bin/env python3
"""External certifier — OPEN | STOP for multi-plane claim bundles.

Law: explore ≠ certify. LLMs (local or cloud) may draft bundles; only this gate OPENs.

Usage:
  python harness/certify/v1/certify.py path/to/bundle.json
  python harness/certify/v1/certify.py --demo filmore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plane_index(planes: list[dict]) -> dict[str, dict]:
    return {p["id"]: p for p in planes}


def certify_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Deterministic structural certifier (v1).

    OPEN a claim only if:
      - every required plane exists
      - every required plane has present=True
      - no required plane state in {MISSING, DEAD} unless claim tags allow_dead_plane
      - confidence_requested is not DRAFT
      - relation_summary non-empty for multi-plane claims (len(required_planes)>1)
      - evidence_refs non-empty on each required plane

    This is continuity/cover certification — not semantic truth of geology.
    """
    planes = plane_index(bundle.get("planes") or [])
    results = []
    opened = []
    stopped = []

    for claim in bundle.get("claims") or []:
        cid = claim.get("id", "?")
        req = claim.get("required_planes") or []
        tags = set(claim.get("tags") or [])
        checks = []
        ok = True

        def chk(name: str, passed: bool, detail: str = "") -> None:
            nonlocal ok
            checks.append({"check": name, "pass": passed, "detail": detail})
            if not passed:
                ok = False

        conf = str(claim.get("confidence_requested", "DRAFT")).upper()
        chk("not_draft", conf != "DRAFT", conf)
        chk("has_text", bool((claim.get("text") or "").strip()))
        chk("has_required_planes", len(req) >= 1, str(req))

        if len(req) > 1:
            rel = (claim.get("relation_summary") or "").strip()
            chk("multiplane_relation_summary", bool(rel), "relation_summary required")

        for pid in req:
            p = planes.get(pid)
            chk(
                f"plane_exists:{pid}",
                p is not None,
                "ok" if p is not None else "missing from bundle.planes",
            )
            if not p:
                continue
            chk(f"plane_present:{pid}", bool(p.get("present")), str(p.get("present")))
            state = str(p.get("state") or "UNKNOWN").upper()
            refs = p.get("evidence_refs") or []
            chk(f"plane_evidence:{pid}", len(refs) >= 1, f"n_refs={len(refs)}")
            if state in ("MISSING",):
                chk(f"plane_not_missing:{pid}", False, state)
            if state == "DEAD" and "allow_dead_plane" not in tags:
                # DEAD is allowed as a *fact plane* if present=True and tagged
                # e.g. decoder DEAD is evidence; use tag allow_dead_plane
                chk(
                    f"plane_dead_needs_tag:{pid}",
                    False,
                    "state=DEAD requires tag allow_dead_plane on claim",
                )

        # cover completeness: at least one LIVE or HOLD among required if multi-plane
        if len(req) > 1 and ok:
            states = []
            for pid in req:
                p = planes.get(pid) or {}
                states.append(str(p.get("state") or "").upper())
            chk(
                "not_all_unknown",
                any(s and s not in ("UNKNOWN", "") for s in states),
                str(states),
            )

        verdict = "OPEN" if ok else "STOP"
        row = {
            "claim_id": cid,
            "verdict": verdict,
            "confidence_requested": conf,
            "checks": checks,
        }
        results.append(row)
        if verdict == "OPEN":
            opened.append(cid)
        else:
            stopped.append(cid)

    payload = {
        "bundle_id": bundle.get("bundle_id"),
        "value_function": bundle.get("value_function"),
        "certified_at": datetime.now(timezone.utc).isoformat(),
        "certifier": "harness.certify.v1",
        "authority": "external_gate",
        "claim_results": results,
        "opened": opened,
        "stopped": stopped,
        "bundle_verdict": "OPEN" if opened and not stopped else ("MIXED" if opened else "STOP"),
    }
    # seal hash of stable content
    seal_src = json.dumps(
        {"bundle_id": bundle.get("bundle_id"), "opened": opened, "stopped": stopped},
        sort_keys=True,
    ).encode("utf-8")
    payload["seal_sha256"] = hashlib.sha256(seal_src).hexdigest()
    return payload


def demo_filmore() -> dict[str, Any]:
    """Structural demo aligned with golden Mag-PI multi-plane story."""
    return {
        "bundle_id": "demo_filmore_magpi_downhole_only",
        "value_function": "multi_plane_operational_truth_NPT_integrity",
        "expected_process": (
            "If tool events fire during RIH, RT decoder sessions should exist in-window "
            "or claim must explicitly treat decoder DEAD as a plane fact."
        ),
        "planes": [
            {
                "id": "tool_magpi",
                "modality": "tool_dump",
                "present": True,
                "state": "LIVE",
                "evidence_refs": [
                    "Post_Run_SSI_Reports/ACQ_RT/TD26324/bha_surface_fusion/rih_magpi_onset.json"
                ],
            },
            {
                "id": "wits_surface",
                "modality": "wits",
                "present": True,
                "state": "HOLD",
                "evidence_refs": [
                    "Post_Run_SSI_Reports/ACQ_RT/TD26324/bha_surface_fusion/magpi_burst_zoom.json"
                ],
            },
            {
                "id": "decoder_rt",
                "modality": "decoder",
                "present": True,
                "state": "DEAD",
                "evidence_refs": [
                    "Post_Run_SSI_Reports/ACQ_RT/TD26324/decoder_raw/decoder_raw_probe.json"
                ],
            },
        ],
        "claims": [
            {
                "id": "MAGPI_DOWNHOLE_ONLY_NO_RT_DECODE",
                "text": (
                    "Zero decoder sessions during Mag-PI burst; tool memory LIVE; "
                    "surface HOLD — Mag-PI is downhole-only, not RT decode glitch."
                ),
                "required_planes": ["tool_magpi", "wits_surface", "decoder_rt"],
                "confidence_requested": "HIGH",
                "relation_summary": (
                    "tool_magpi LIVE + decoder_rt DEAD + wits_surface HOLD in same window"
                ),
                "tags": ["allow_dead_plane", "golden_filmore"],
            },
            {
                "id": "DRAFT_SINGLE_PLANE_STORY",
                "text": "Nothing happened — surface was quiet.",
                "required_planes": ["wits_surface"],
                "confidence_requested": "DRAFT",
                "relation_summary": "",
                "tags": [],
            },
        ],
        "continuity_guards": [
            "reject_decode_glitch_without_decoder_plane",
            "reject_surface_quiet_equals_tool_quiet",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Certify multi-plane claim bundles")
    ap.add_argument("bundle", nargs="?", help="Path to bundle JSON")
    ap.add_argument("--demo", choices=["filmore"], help="Run built-in demo bundle")
    ap.add_argument("-o", "--out", help="Write certificate JSON to path")
    args = ap.parse_args()

    if args.demo == "filmore":
        bundle = demo_filmore()
    elif args.bundle:
        bundle = load_json(Path(args.bundle))
    else:
        ap.print_help()
        return 2

    cert = certify_bundle(bundle)
    text = json.dumps(cert, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(
        f"BUNDLE_VERDICT={cert['bundle_verdict']} "
        f"opened={cert['opened']} stopped={cert['stopped']}"
    )
    return 0 if cert["bundle_verdict"] in ("OPEN", "MIXED") else 1


if __name__ == "__main__":
    sys.exit(main())
