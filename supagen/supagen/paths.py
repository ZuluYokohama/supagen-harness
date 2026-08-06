"""Resolve monorepo roots (dev tree or installed layout)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _has_prime(p: Path) -> bool:
    return (p / "prime" / "scripts" / "nomic_metric.py").is_file() or (
        p / "scripts" / "nomic_metric.py"
    ).is_file()


def _has_harness(p: Path) -> bool:
    return (p / "harness" / "smoke_offline.py").is_file() or (
        p / "smoke_offline.py"
    ).is_file()


def workspace_root() -> Path:
    """PRIMEdEV-1 (or checkout root that contains prime/ + harness/)."""
    env = os.environ.get("SUPAGEN_ROOT") or os.environ.get("PRIMEDEV_ROOT")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p

    # state/SUPAGEN_ROOT.txt written by bootstrap
    here = Path(__file__).resolve()
    marker = here.parents[1] / "state" / "SUPAGEN_ROOT.txt"
    if marker.is_file():
        try:
            p = Path(marker.read_text(encoding="utf-8").strip())
            if p.is_dir() and (_has_prime(p) or _has_harness(p)):
                return p
        except Exception:
            pass

    candidates = [
        here.parents[2],  # monorepo/supagen/supagen → monorepo
        here.parents[1],  # monorepo/supagen
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        c = c.resolve()
        if _has_prime(c) or _has_harness(c):
            return c
        if _has_prime(c.parent) or _has_harness(c.parent):
            return c.parent
    return here.parents[2]


def prime_scripts() -> Path:
    root = workspace_root()
    for p in (root / "prime" / "scripts", root / "scripts"):
        if p.is_dir():
            return p
    return root / "prime" / "scripts"


def harness_root() -> Path:
    root = workspace_root()
    for p in (root / "harness", root):
        if (p / "smoke_offline.py").is_file() or (p / "pipeline").is_dir():
            return p
    return root / "harness"


def ensure_sys_path() -> dict[str, str]:
    """Put prime scripts + harness on sys.path (import dual_enter, jina_service, …)."""
    ps = prime_scripts()
    hr = harness_root()
    for p in (ps, hr, hr / "certify" / "v1", hr / "pipeline" / "v1"):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    return {
        "workspace": str(workspace_root()),
        "prime_scripts": str(ps),
        "harness": str(hr),
    }
