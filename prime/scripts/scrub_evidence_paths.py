#!/usr/bin/env python3
"""Scrub host-specific paths from docs/evidence text artifacts."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "docs" / "evidence"
HOST_RE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/")


def scrub_str(s: str) -> str:
    s = s.replace("\\", "/")
    s = re.sub(r"C:/Users/[^/\s\"]+/", "<user>/", s)
    s = re.sub(r"/Users/[^/\s\"]+/", "<user>/", s)
    s = re.sub(r"/home/[^/\s\"]+/", "<user>/", s)
    if "PRIMEdEV" in s or re.search(r"[A-Za-z]:/", s):
        for m in ("prime/", "docs/", "supagen/", "golden_paths/", "123abc/"):
            i = s.lower().find(m)
            if i >= 0:
                return s[i:]
        if re.search(r"[A-Za-z]:/", s):
            return Path(s).name
    return s


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, str):
        return scrub_str(o)
    return o


def main() -> int:
    still = []
    for p in EV.rglob("*"):
        if p.suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix.lower() == ".json":
            try:
                data = json.loads(raw)
            except Exception as e:
                print("skip", p, e)
                continue
            p.write_text(json.dumps(scrub(data), indent=2) + "\n", encoding="utf-8")
        else:
            out = scrub_str(raw)
            # line-wise for md
            lines = []
            for line in raw.splitlines(keepends=True):
                lines.append(scrub_str(line) if HOST_RE.search(line) else line)
            p.write_text("".join(lines), encoding="utf-8")
        check = p.read_text(encoding="utf-8", errors="replace")
        if HOST_RE.search(check):
            still.append(str(p.relative_to(ROOT)))
    print("still_host_paths", still or "NONE")
    return 1 if still else 0


if __name__ == "__main__":
    raise SystemExit(main())
