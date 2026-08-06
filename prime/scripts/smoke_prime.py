#!/usr/bin/env python3
"""Offline smoke for Prime core (no MCP transport)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compute_graph import plan_graph, advance  # noqa: E402
from measures import lm_list_models, measure_rplc, measure_smoke  # noqa: E402
from session_store import SessionStore  # noqa: E402


def _domain_root() -> Path:
    """rplc domain if present; else soft missing."""
    try:
        from workspace import rplc_root, workspace_root

        c = rplc_root()
        if c.is_dir():
            return c
        return workspace_root() / "123abc"
    except Exception:
        pass
    for c in (Path.cwd() / "123abc", Path(__file__).resolve().parents[2] / "123abc"):
        if (c / "rplc_sheaf.py").is_file() or (c / "tests" / "test_smoke.py").is_file():
            return c
    return Path.cwd() / "123abc"


def main() -> int:
    domain = _domain_root()
    # graph
    g = plan_graph("smoke intent", modes=["code", "lm", "rplc"])
    assert "META_META" in g.nodes
    r = advance(g, "META")
    assert r["ok"], r
    bad = advance(g, "OPEN")
    assert not bad["ok"], "illegal jump to OPEN should fail"

    # session
    store = SessionStore(ROOT.parent / "state" / "smoke_state")
    st = store.start(str(domain), intent="smoke", modes=["rplc", "lm"])
    assert st["ok"]
    store.set_restrict("smoke goal", ["force open"], ["measures"], ["law"])
    # force open without measure should STOP
    store.add_audit("OPEN", ["try force"])
    store2 = SessionStore(ROOT.parent / "state" / "smoke_state2")
    store2.start(str(domain), "x", ["rplc"])
    a2 = store2.add_audit("OPEN", ["no measures"])
    assert a2["audit"]["verdict"] == "STOP", a2

    # measures (soft if domain missing)
    if domain.is_dir():
        sm = measure_smoke(str(domain))
        print("smoke", sm.get("ok"), sm.get("mode"))
        rp = measure_rplc(str(domain), n=48)
        print("rplc", rp.get("ok"), "opened", rp.get("opened_steps"), "verify", rp.get("verify_ok"))
    else:
        print("smoke SKIP no domain", domain)
    lm = lm_list_models()
    print("lm", lm.get("ok"), lm.get("models"))

    print("prime smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
