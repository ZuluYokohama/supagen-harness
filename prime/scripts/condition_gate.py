"""Condition gate — prefer isoz rotary transducer; fall back to local energy proxy."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _try_isoz():
    roots = [
        Path(r"C:\Users\Deving-1\.grok\installed-plugins\isoz-core-basic-e8236488\scripts"),
        Path(__file__).resolve().parent,
    ]
    for r in roots:
        if (r / "rotary_condition_state.py").exists():
            if str(r) not in sys.path:
                sys.path.insert(0, str(r))
            from rotary_condition_state import pulse, read_state  # type: ignore
            try:
                from oracle import handle_obstruction, ORACLE  # type: ignore
            except Exception:
                handle_obstruction, ORACLE = None, None
            return pulse, read_state, handle_obstruction, ORACLE
    return None, None, None, None


def pulse_condition(artifact: str, context: str = "prime") -> dict[str, Any]:
    pulse, read_state, handle_obstruction, ORACLE = _try_isoz()
    if pulse is None:
        # lightweight proxy: length + uniqueness entropy stand-in (never claims sheaf math)
        n = max(len(artifact), 1)
        uniq = len(set(artifact.split())) / max(len(artifact.split()), 1)
        energy = max(0.0, 1.0 - uniq) * (n / 500.0)
        kernel = energy < 0.15
        return {
            "verdict": "CONSISTENT" if kernel else "OBSTRUCTED",
            "energy": energy,
            "kernel_member": kernel,
            "engine": "prime-proxy",
            "recommendation": "OK to proceed" if kernel else "Revise artifact or escalate",
            "note": "isoz rotary not found; proxy only",
        }
    pr = pulse(artifact, context)
    out = {
        "verdict": pr.get("verdict"),
        "energy": pr.get("energy"),
        "kernel_member": pr.get("kernel_member"),
        "obstruction": pr.get("obstruction"),
        "delta_lambda": pr.get("delta_lambda"),
        "hot_linkages": pr.get("hot_linkages"),
        "engine": "isoz-rotary",
        "pulse_summary": pr,
    }
    if not pr.get("kernel_member") and pr.get("obstruction") and handle_obstruction:
        out["oracle_action"] = handle_obstruction(
            pr.get("energy"), pr.get("obstruction"), []
        )
        out["recommendation"] = "DO NOT EMIT until obstruction resolved"
    else:
        out["recommendation"] = "OK to proceed" if pr.get("kernel_member") else "Review obstruction"
    return out


def read_condition() -> dict[str, Any]:
    pulse, read_state, handle_obstruction, ORACLE = _try_isoz()
    if read_state is None:
        return {"ok": True, "engine": "prime-proxy", "note": "no persistent transducer"}
    st = read_state()
    if ORACLE:
        st["oracle"] = ORACLE.get_stats()
    st["engine"] = "isoz-rotary"
    return st
