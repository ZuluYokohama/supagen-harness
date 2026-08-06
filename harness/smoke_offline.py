#!/usr/bin/env python3
"""Offline harness smoke — no LM Studio required.

Layout-aware: works when this package is:
  - C:\\PRIMEdEV-1\\harness\\  (dev tree under workspace)
  - ~/.grok/installed-plugins/<plugin>/  (Grok install; package root IS harness)

Runs:
  1) golden_paths/filmore multi-plane sealed claim verify
  2) certify/v1 Filmore OPEN|STOP demo
  3) pipeline packs (filmore_magpi, frozen_lakes_surface)
  4) ingest sandbox filmore inventory (if sandbox present)

Online scout smoke (needs LMS :1234):
  python local_mode/lfm_scout_v1/smoke_local.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# This file lives at harness package root (not inside harness/ subfolder when installed)
HARNESS = Path(__file__).resolve().parent


def workspace_root() -> Path:
    """Monorepo root (portable) — golden_paths / sandbox if present."""
    import os

    env = os.environ.get("SUPAGEN_ROOT") or os.environ.get("PRIMEDEV_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    candidates = [
        HARNESS.parent,  # monorepo/harness → parent
        HARNESS.parents[1] if len(HARNESS.parents) > 1 else HARNESS,
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "golden_paths" / "filmore_multiplane_v1" / "verify_golden.py").is_file():
            return c
        if (c / "harness").is_dir() and (c / "prime").is_dir():
            return c
    return HARNESS.parent


ROOT = workspace_root()


def run(label: str, argv: list[str], cwd: Path | None = None) -> int:
    print(f"\n=== {label} ===")
    print(">", " ".join(argv))
    p = subprocess.run(argv, cwd=str(cwd or HARNESS))
    print(f"exit={p.returncode}")
    return p.returncode


def main() -> int:
    py = sys.executable
    codes: list[int] = []

    golden = ROOT / "golden_paths/filmore_multiplane_v1/verify_golden.py"
    if golden.is_file():
        codes.append(run("golden_filmore", [py, str(golden)], cwd=ROOT))
    else:
        print(f"\n=== golden_filmore ===\nSKIP missing {golden}")
        codes.append(0)  # soft skip if golden not on disk

    certify = HARNESS / "certify/v1/certify.py"
    codes.append(run("certify_filmore_demo", [py, str(certify), "--demo", "filmore"]))

    pipeline = HARNESS / "pipeline/v1/pipeline.py"
    codes.append(
        run("pipeline_filmore_pack", [py, str(pipeline), "--pack", "filmore_magpi"])
    )
    codes.append(
        run(
            "pipeline_frozen_lakes_pack",
            [py, str(pipeline), "--pack", "frozen_lakes_surface"],
        )
    )

    ingest = HARNESS / "ingest/v1/ingest.py"
    sandbox = ROOT / "123abc/_sandbox_extract/SandBox"
    if sandbox.is_dir() or (ROOT / "_sandbox_extract").is_dir():
        codes.append(
            run("ingest_sandbox_inventory", [py, str(ingest), "--sandbox-filmore"])
        )
    else:
        print("\n=== ingest_sandbox_inventory ===\nSKIP no sandbox extract")
        codes.append(0)

    print("\n=== local_scout (optional online) ===")
    print("Skip in offline smoke. When LM Studio server is up:")
    print(f"  {py} {HARNESS / 'local_mode/lfm_scout_v1/smoke_local.py'}")

    failed = [c for c in codes if c != 0]
    if not failed:
        print("\nHARNESS OFFLINE SMOKE OK")
        print(f"HARNESS={HARNESS}")
        print(f"ROOT={ROOT}")
        print("Tree: golden Q + external certify gate live. Scout waits on LMS.")
        return 0
    print("\nHARNESS OFFLINE SMOKE FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
