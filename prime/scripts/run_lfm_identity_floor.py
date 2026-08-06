#!/usr/bin/env python3
"""LFM 1.2B identity floor — DeBERTa mutual entailment p (cos is diagnostic only)."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from residency import seamless_substrate  # noqa: E402
from semantic_holonomy_v3 import SEEDS, Judge, round_trip  # noqa: E402


def main() -> int:
    model = "liquid/lfm2.5-1.2b"
    print("seamless substrate…", flush=True)
    sub = seamless_substrate(chat_model=model)
    print(
        "substrate",
        sub.get("ok"),
        "ctx",
        (sub.get("fiber") or {}).get("loaded_ctx"),
        flush=True,
    )
    model = (sub.get("fiber") or {}).get("model") or model

    print("loading DeBERTa judge…", flush=True)
    judge = Judge()
    floor = []
    t0 = time.time()
    for s in SEEDS[:8]:
        try:
            r = round_trip(model, judge, s, 1, identity=True)
        except Exception as e:
            print(f"  ERR {e}", flush=True)
            r = None
        if r:
            floor.append(r)
            print(
                f"  {r['mode']:<10} cos {r['cos']:.3f}  closed={r['closed']}  "
                f"fwd {r['fwd'][:4]}/rev {r['rev'][:4]}  {s[:48]}",
                flush=True,
            )
    if not floor:
        print("no floor rows", flush=True)
        return 1

    fc = float(np.mean([r["closed"] for r in floor]))
    cos75 = float(np.mean([r["cos"] >= 0.75 for r in floor]))
    med = float(np.median([r["cos"] for r in floor]))
    modes = dict(Counter(r["mode"] for r in floor))
    out = {
        "model": model,
        "protocol": "v3 identity floor — closure=mutual entailment; cos diagnostic only",
        "identity_closure_p": round(fc, 4),
        "cosine_would_say_ge_0.75": round(cos75, 4),
        "median_cos": round(med, 4),
        "gate_threshold": 0.80,
        "gate_failed": fc < 0.80,
        "modes": modes,
        "n": len(floor),
        "seconds": round(time.time() - t0, 1),
        "fiber_ctx": (sub.get("fiber") or {}).get("loaded_ctx"),
        "compare": {
            "frankenstein_prior": "~0.88–0.94 PASS",
            "gemma12b_prior": "0.38 FAIL",
            "lfm_prior_partial": 0.286,
        },
        "reading": (
            f"LFM1.2B identity p={fc:.2f} (need ≥0.80 to pass identity gate). "
            f"Cosine≥0.75 would claim {cos75:.2f} — do not gate on cos. "
            f"Modes={modes}."
        ),
        "rows": [{k: v for k, v in r.items() if k != "texts"} for r in floor],
    }
    path = ROOT.parent / "state" / "holonomy_v3_lfm12b_identity_floor.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + out["reading"], flush=True)
    print(
        f"GATE={'FAIL' if out['gate_failed'] else 'PASS'}  p={fc:.2f}  cos.75={cos75:.2f}",
        flush=True,
    )
    print("wrote", path, flush=True)
    return 0 if not out["gate_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
