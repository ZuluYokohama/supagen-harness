#!/usr/bin/env python3
"""
Buddy L8 offline runner — clean-ish isolation for L8-01…04.

Runs the offline buddy protocol and writes a signed-style JSON evidence
artifact under docs/evidence/. This is still *author machine* unless invoked
from a clean clone/venv; the artifact records isolation flags honestly.

Usage:
  python prime/scripts/buddy_l8_offline.py
  python prime/scripts/buddy_l8_offline.py --role author_offline_isolated
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "evidence"


def _run(cmd: list[str], env: dict) -> dict:
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return {
        "cmd": cmd,
        "rc": p.returncode,
        "seconds": round(time.time() - t0, 2),
        "tail": out[-1200:],
        "ok": p.returncode == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--role",
        default="author_offline_protocol_runner",
        help="recorded in evidence; use independent_buddy only if truly external",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env = os.environ.copy()
    env["SUPAGEN_ROOT"] = str(ROOT)
    env["PRIMEDEV_ROOT"] = str(ROOT)
    env["GOLDEN_SCHEMA_ONLY"] = "1"
    py = sys.executable

    steps = {
        "L8-01_offline_contract": _run(
            [py, "-m", "supagen", "contract", "--offline"], env
        ),
        "L8-02_smoke": _run([py, "-m", "supagen", "smoke"], env),
        "L8-03_harness": _run([py, "-m", "supagen", "harness", "smoke"], env),
        "L8-04_golden_schema": _run(
            [py, str(ROOT / "golden_paths" / "filmore_multiplane_v1" / "verify_golden.py")],
            env,
        ),
    }
    # golden may be nested under harness smoke already; still explicit
    all_ok = all(s.get("ok") for s in steps.values())
    rec = {
        "role": a.role,
        "independent_buddy": a.role == "independent_buddy",
        "protocol": "docs/BUDDY_L8_SIGNOFF.md",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": py,
        "root": str(ROOT),
        "checks": {
            k: {"ok": v["ok"], "rc": v["rc"], "seconds": v["seconds"]}
            for k, v in steps.items()
        },
        "tails": {k: v["tail"][-400:] for k, v in steps.items()},
        "offline_verdict": "PASS" if all_ok else "FAIL",
        "note": (
            "Offline L8-01…04 only. Live L8-05…08 require LMS/jina. "
            "Production OPEN marketing remains NO-GO."
        ),
        "law": "this artifact is package portability evidence, not production OPEN",
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else EVIDENCE / "buddy_l8_offline_protocol_run.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("offline_verdict", "checks", "role")}, indent=2))
    print("wrote", out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
