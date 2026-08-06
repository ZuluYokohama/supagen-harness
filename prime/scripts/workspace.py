"""Portable workspace root — no C:\\PRIMEdEV-1 hardcodes for buddy clones."""
from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    env = os.environ.get("SUPAGEN_ROOT") or os.environ.get("PRIMEDEV_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    # prime/scripts/workspace.py → prime → monorepo
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2],  # .../prime/scripts → monorepo
        here.parents[1].parent if len(here.parents) > 1 else here,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        if (c / "prime" / "scripts").is_dir() or (c / "supagen").is_dir():
            return c
        if (c / "scripts" / "nomic_metric.py").is_file() and (c / "docs").is_dir():
            # bare prime package layout
            return c.parent if (c.parent / "harness").is_dir() else c
    return here.parents[2]


def prime_root() -> Path:
    w = workspace_root()
    if (w / "prime").is_dir():
        return w / "prime"
    return w


def harness_root() -> Path:
    w = workspace_root()
    if (w / "harness").is_dir():
        return w / "harness"
    return w / "harness"


def rplc_root() -> Path:
    w = workspace_root()
    for name in ("123abc", "rplc-sheaf"):
        p = w / name
        if (p / "rplc_sheaf.py").is_file() or (p / "tests").is_dir():
            return p
    return w / "123abc"
