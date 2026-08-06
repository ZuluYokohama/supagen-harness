#!/usr/bin/env python3
"""Load frankenstein alone, run holonomy v3 ladder depths 2-4."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from holonomy_capacity_bench import ensure_embed, load_model, unload_all_llms


def main() -> int:
    print("=== frankenstein ladder: unload → load → v3 depths 2 3 4 ===", flush=True)
    unload_all_llms()
    ensure_embed()
    lr = load_model("frankenstein-2.0-i1", 16384)
    if not lr.get("ok"):
        print("LOAD FAIL", lr, flush=True)
        return 1
    # run v3 main
    sys.argv = [
        "semantic_holonomy_v3.py",
        "--model",
        "frankenstein-2.0-i1",
        "--depths",
        "2",
        "3",
        "4",
        "--seeds",
        "8",
        "--out",
        str(ROOT.parent / "state" / "holonomy_v3_frankenstein_ladder.json"),
    ]
    from semantic_holonomy_v3 import main as v3main

    code = v3main()
    unload_all_llms()
    ensure_embed()
    # restore LFM max for daily work
    load_model("liquid/lfm2.5-1.2b", 128000)
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
