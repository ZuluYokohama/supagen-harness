#!/usr/bin/env python3
"""Scout → bundle → certify pipeline.

Law: scout/LLM explores; this pipeline builds the bundle; certify OPENs or STOPs.

Examples:
  # pack only (no LLM)
  python harness/pipeline/v1/pipeline.py --pack filmore_magpi

  # pack + existing scout markdown
  python harness/pipeline/v1/pipeline.py --pack filmore_magpi --scout path/to/scout.md

  # pack + live LFM scout then certify
  python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout lfm

  # pack + live Bonsai fast scout
  python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout bonsai
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# HERE=.../pipeline/v1
# Plugin install: parents[1] = package root (certify/ sibling). Dev tree: same (harness/).
HARNESS = HERE.parents[1]
if not (HARNESS / "certify" / "v1" / "certify.py").is_file():
    # legacy nested harness/harness mistake — walk up
    for p in HERE.parents:
        if (p / "certify" / "v1" / "certify.py").is_file():
            HARNESS = p
            break
# Workspace root for golden_paths / cwd (portable)
import os as _os

_ws_candidates = [
    Path(_os.environ["SUPAGEN_ROOT"]) if _os.environ.get("SUPAGEN_ROOT") else None,
    Path(_os.environ["PRIMEDEV_ROOT"]) if _os.environ.get("PRIMEDEV_ROOT") else None,
    HARNESS.parent if (HARNESS.parent / "golden_paths").is_dir() else None,
    HARNESS.parent if (HARNESS.parent / "prime").is_dir() else None,
    HERE.parents[2] if len(HERE.parents) > 2 else None,
]
ROOT = next(
    (p for p in _ws_candidates if p is not None and p.is_dir()),
    HARNESS.parent,
)
sys.path.insert(0, str(HERE))

from build_bundle import bundle_from_pack, write_bundle  # noqa: E402

# import certify from harness package path
sys.path.insert(0, str(HARNESS / "certify" / "v1"))
from certify import certify_bundle  # noqa: E402

OUT = HERE / "out"


def run_live_scout(which: str) -> Path:
    """Run local scout_turn; return path to newest scout_*.md in runs/."""
    if which == "lfm":
        script = HARNESS / "local_mode/lfm_scout_v1/scout_turn.py"
        runs = HARNESS / "local_mode/lfm_scout_v1/runs"
    elif which == "bonsai":
        script = HARNESS / "local_mode/bonsai_scout_v1/scout_turn.py"
        runs = HARNESS / "local_mode/bonsai_scout_v1/runs"
    else:
        raise ValueError("live-scout must be lfm or bonsai")

    cmd = [sys.executable, str(script)]
    if which == "bonsai":
        cmd.append("--fast")
    print(">", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"live scout failed exit={rc}")

    md_files = sorted(runs.glob("scout_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not md_files:
        raise FileNotFoundError(f"no scout_*.md in {runs}")
    return md_files[0]


def extract_reply(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    if "## Reply" in text:
        return text.split("## Reply", 1)[1].strip()
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Scout → bundle → certify")
    ap.add_argument("--pack", default="filmore_magpi", help="pack name under packs/ or path")
    ap.add_argument("--scout", type=Path, help="path to scout .md/.txt")
    ap.add_argument("--live-scout", choices=["lfm", "bonsai"], help="run local LMS scout first")
    ap.add_argument("-o", "--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    scout_text = None
    scout_path = None
    if args.live_scout:
        scout_path = run_live_scout(args.live_scout)
        scout_text = extract_reply(scout_path)
        print(f"live scout → {scout_path}")
    elif args.scout:
        scout_path = args.scout
        scout_text = extract_reply(scout_path) if scout_path.suffix == ".md" else scout_path.read_text(encoding="utf-8")

    tag = "pack"
    if scout_path:
        tag = "pack_scout"
    if args.live_scout:
        tag = f"pack_live_{args.live_scout}"

    bundle = bundle_from_pack(args.pack, scout_text=scout_text, bundle_id_suffix=ts)
    if scout_path:
        bundle.setdefault("meta", {})
        bundle["meta"]["scout_path"] = str(scout_path)

    bundle_path = out_dir / f"bundle_{tag}_{ts}.json"
    cert_path = out_dir / f"certificate_{tag}_{ts}.json"
    write_bundle(bundle, bundle_path)
    print(f"bundle → {bundle_path}")

    cert = certify_bundle(bundle)
    cert_path.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    print(f"certificate → {cert_path}")
    print(
        f"BUNDLE_VERDICT={cert['bundle_verdict']} "
        f"opened={cert['opened']} stopped={cert['stopped']} "
        f"seal={cert['seal_sha256'][:16]}..."
    )

    # human summary
    summary = out_dir / f"summary_{tag}_{ts}.md"
    lines = [
        f"# Pipeline run {ts}",
        "",
        f"- pack: `{args.pack}`",
        f"- scout: `{scout_path or 'none'}`",
        f"- bundle: `{bundle_path.name}`",
        f"- certificate: `{cert_path.name}`",
        f"- verdict: **{cert['bundle_verdict']}**",
        f"- opened: {', '.join(cert['opened']) or '—'}",
        f"- stopped: {', '.join(cert['stopped']) or '—'}",
        f"- seal: `{cert['seal_sha256']}`",
        "",
        "## Law",
        "Scout explores. Certify OPENs or STOPs. Residue is allowed.",
        "",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary → {summary}")

    # 0 if anything OPEN (MIXED or OPEN); 1 if total STOP
    return 0 if cert["bundle_verdict"] in ("OPEN", "MIXED") else 1


if __name__ == "__main__":
    sys.exit(main())
