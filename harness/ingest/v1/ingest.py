#!/usr/bin/env python3
"""Ingest tool dump folder and/or ACQ .emz → plane inventory + draft pack → optional pipeline.

Usage:
  python harness/ingest/v1/ingest.py --dump path/to/MicroPulse_CSVs
  python harness/ingest/v1/ingest.py --emz path/to/well.emz
  python harness/ingest/v1/ingest.py --dump D:\\dumps --emz D:\\well.emz --run-pipeline
  python harness/ingest/v1/ingest.py --sandbox-filmore   # use extracted SandBox paths
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# file .../ingest/v1/ingest.py → parents[0]=v1, [1]=ingest, [2]=harness package root
_HERE = Path(__file__).resolve().parent
HARNESS = _HERE.parents[1]
if not (HARNESS / "pipeline" / "v1").is_dir():
    for p in _HERE.parents:
        if (p / "pipeline" / "v1").is_dir() and (p / "certify" / "v1").is_dir():
            HARNESS = p
            break
# Workspace for sandbox extracts (portable)
import os as _os

_ws_candidates = [
    Path(_os.environ["SUPAGEN_ROOT"]) if _os.environ.get("SUPAGEN_ROOT") else None,
    Path(_os.environ["PRIMEDEV_ROOT"]) if _os.environ.get("PRIMEDEV_ROOT") else None,
    HARNESS.parent if (HARNESS.parent / "123abc").is_dir() else None,
    HARNESS.parent if (HARNESS.parent / "prime").is_dir() else None,
    _HERE.parents[3] if len(_HERE.parents) > 3 else None,
]
ROOT = next(
    (p for p in _ws_candidates if p is not None and p.is_dir()),
    HARNESS.parent,
)
OUT = _HERE / "out"

# Match after MicroPulse_SN_ prefix — avoid matching "PULSE" inside "MICROPULSE"
# Patterns are substrings of the TYPE segment (e.g. _PULSE_, _SHOCK_VIBE_)
DUMP_PATTERNS = [
    ("_PULSE_", "tool_pulse", "tool_dump"),
    ("_SURVEY_", "tool_survey", "tool_dump"),
    ("_GAMMA_", "tool_gamma", "tool_dump"),
    ("_SHOCK", "tool_shock", "tool_dump"),
    ("_SYSTEM_", "tool_magpi", "tool_dump"),  # Mag-PI / system events
    ("_CONTINUOUS", "tool_continuous", "tool_dump"),
    ("_TELEM_", "tool_telem", "tool_dump"),
    ("_VOLTAGE", "tool_voltages", "tool_dump"),
    ("_CONFIG_", "tool_config", "tool_dump"),
    ("_ENVIRONMENT", "tool_environment", "tool_dump"),
]


def scan_dump(dump_dir: Path) -> list[dict[str, Any]]:
    planes = []
    if not dump_dir.is_dir():
        return planes
    for p in sorted(dump_dir.glob("*.csv")):
        name = p.name.upper()
        matched = False
        for key, pid, mod in DUMP_PATTERNS:
            if key in name:
                planes.append(
                    {
                        "id": pid,
                        "modality": mod,
                        "present": True,
                        "state": "LIVE",
                        "evidence_refs": [str(p.resolve())],
                    }
                )
                matched = True
                break
        if not matched:
            planes.append(
                {
                    "id": f"tool_csv_{p.stem[:40]}",
                    "modality": "tool_dump",
                    "present": True,
                    "state": "LIVE",
                    "evidence_refs": [str(p.resolve())],
                }
            )
    return planes


def scan_emz(emz_path: Path) -> list[dict[str, Any]]:
    planes = []
    if not emz_path.is_file():
        return planes
    refs_wits = []
    refs_dec = []
    refs_db = []
    try:
        with zipfile.ZipFile(emz_path, "r") as z:
            for n in z.namelist():
                nl = n.replace("\\", "/").lower()
                if "/wits" in nl or "wits logs" in nl:
                    refs_wits.append(f"emz:{emz_path.name}:{n}")
                if "decoder" in nl:
                    refs_dec.append(f"emz:{emz_path.name}:{n}")
                if nl.endswith(".db") or nl.endswith("phm.db"):
                    refs_db.append(f"emz:{emz_path.name}:{n}")
    except zipfile.BadZipFile:
        planes.append(
            {
                "id": "acq_emz",
                "modality": "acq_db",
                "present": True,
                "state": "UNKNOWN",
                "evidence_refs": [str(emz_path.resolve())],
            }
        )
        return planes

    if refs_wits:
        planes.append(
            {
                "id": "wits_surface",
                "modality": "wits",
                "present": True,
                "state": "LIVE",
                "evidence_refs": refs_wits[:5] + ([f"...+{len(refs_wits)-5} more"] if len(refs_wits) > 5 else []),
            }
        )
    if refs_dec:
        planes.append(
            {
                "id": "decoder_rt",
                "modality": "decoder",
                "present": True,
                "state": "LIVE",
                "evidence_refs": refs_dec[:5] + ([f"...+{len(refs_dec)-5} more"] if len(refs_dec) > 5 else []),
            }
        )
    if refs_db:
        planes.append(
            {
                "id": "acq_db",
                "modality": "acq_db",
                "present": True,
                "state": "LIVE",
                "evidence_refs": refs_db[:8],
            }
        )
    planes.append(
        {
            "id": "acq_emz_package",
            "modality": "acq_db",
            "present": True,
            "state": "LIVE",
            "evidence_refs": [str(emz_path.resolve())],
        }
    )
    return planes


def build_inventory_pack(
    planes: list[dict[str, Any]],
    pack_id: str,
) -> dict[str, Any]:
    """Inventory-only pack: DRAFT cover claim so certify STOPs until upgraded."""
    ids = [p["id"] for p in planes if p.get("present")]
    return {
        "bundle_id": pack_id,
        "value_function": "multi_plane_operational_truth_NPT_integrity",
        "expected_process": "Ingested files form a multi-plane cover; claims need explicit upgrade to HIGH with relation_summary.",
        "planes": planes,
        "claims": [
            {
                "id": "INGEST_COVER_DRAFT",
                "text": f"Ingest saw {len(ids)} present planes: {', '.join(ids[:12])}",
                "required_planes": ids[:3] if len(ids) >= 1 else ["missing"],
                "confidence_requested": "DRAFT",
                "relation_summary": "inventory only — not certified multi-plane truth",
                "tags": ["from_ingest", "explore_only"],
            }
        ],
        "continuity_guards": ["ingest_draft_never_auto_open"],
        "meta": {"plane_count": len(planes), "present_ids": ids},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest dumps/emz → pack → optional pipeline")
    ap.add_argument("--dump", type=Path, help="Tool dump folder (CSV)")
    ap.add_argument("--emz", type=Path, help="ACQ .emz export")
    ap.add_argument("--run-pipeline", action="store_true", help="Run pipeline after writing pack")
    ap.add_argument(
        "--sandbox-filmore",
        action="store_true",
        help="Use SandBox extract dump+emz if present",
    )
    ap.add_argument("-o", "--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    dump = args.dump
    emz = args.emz
    if args.sandbox_filmore:
        sb_candidates = [
            ROOT / "123abc" / "_sandbox_extract" / "SandBox",
            ROOT / "_sandbox_extract" / "SandBox",
            ROOT / "123abc" / "_sandbox_extract" / "SandBox",
        ]
        sb = next((p for p in sb_candidates if p.is_dir()), sb_candidates[0])
        dump = sb / "MWD_Run001_DumpFiles"
        emzs = list((sb / "Acquisition_System_Exports").glob("*Filmore*.emz")) if (
            sb / "Acquisition_System_Exports"
        ).is_dir() else []
        emz = emzs[0] if emzs else None

    if not dump and not emz:
        ap.error("provide --dump and/or --emz (or --sandbox-filmore)")

    planes: list[dict[str, Any]] = []
    if dump:
        planes.extend(scan_dump(dump))
    if emz:
        planes.extend(scan_emz(emz))

    # de-dupe plane ids (keep first)
    seen = set()
    uniq = []
    for p in planes:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        uniq.append(p)
    planes = uniq

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = build_inventory_pack(planes, f"ingest_{ts}")
    pack_path = out_dir / f"pack_ingest_{ts}.json"
    inv_path = out_dir / f"inventory_{ts}.json"
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    inv_path.write_text(json.dumps({"planes": planes, "ts": ts}, indent=2), encoding="utf-8")
    print(f"planes_present={len(planes)}")
    print(f"inventory → {inv_path}")
    print(f"pack → {pack_path}")
    for p in planes:
        print(f"  - {p['id']:24} {p.get('state')}  refs={len(p.get('evidence_refs') or [])}")

    if args.run_pipeline:
        # pipeline expects pack name under packs/ OR path — pass absolute path as pack
        # build_bundle load_pack tries packs/name.json then Path(name)
        cmd = [
            sys.executable,
            str(HARNESS / "pipeline/v1/pipeline.py"),
            "--pack",
            str(pack_path),
        ]
        print(">", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(HARNESS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
