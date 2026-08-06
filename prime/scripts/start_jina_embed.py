#!/usr/bin/env python3
"""CLI shim → jina_service (always-on aboutness embed server).

  python start_jina_embed.py           # ensure (auto-start if down)
  python start_jina_embed.py --check   # probe only
  python start_jina_embed.py status
  python start_jina_embed.py stop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jina_service import ensure_jina, jina_status, probe_jina, stop_jina  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Jina aboutness service (seamless)")
    ap.add_argument(
        "cmd",
        nargs="?",
        default="ensure",
        choices=("ensure", "status", "stop", "probe", "check"),
    )
    ap.add_argument("--check", action="store_true", help="alias for probe")
    ap.add_argument("--force", action="store_true", help="restart if running")
    a = ap.parse_args()
    cmd = "probe" if a.check or a.cmd == "check" else a.cmd

    if cmd == "ensure":
        r = ensure_jina(force_restart=a.force)
    elif cmd == "status":
        r = jina_status()
    elif cmd == "stop":
        r = stop_jina()
    else:
        r = probe_jina()
    print(json.dumps(r, indent=2))
    if isinstance(r, dict) and "ok" in r:
        return 0 if r.get("ok") else 1
    if isinstance(r, dict) and "probe" in r:
        return 0 if (r.get("probe") or {}).get("ok") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
