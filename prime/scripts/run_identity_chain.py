#!/usr/bin/env python3
"""Identity-chain control: multi-hop pure-identity rewrites (no transform pairs).

Speculative decoding has a fixed draft→target anchor each step.
The holonomy ladder does not — each hop's output is the next hop's input.
This measures preservation under a chain of identity rewrites only, so any
drop is pure accumulation without intentional semantic transforms.

Usage:
  python run_identity_chain.py --model frankenstein-2.0-i1 --depths 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from holonomy_capacity_bench import ensure_embed, load_model, unload_all_llms  # noqa: E402
from semantic_holonomy_v3 import SEEDS, Judge, round_trip  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="frankenstein-2.0-i1")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="0 = ctx_policy resolve (preferred); never assume UI 4096",
    )
    ap.add_argument(
        "--out",
        default=str(
            ROOT.parent / "state" / "holonomy_v3_frankenstein_identity_chain.json"
        ),
    )
    ap.add_argument("--skip-load", action="store_true")
    a = ap.parse_args()

    if not a.skip_load:
        print("unload all LLMs…", flush=True)
        unload_all_llms()
        ensure_embed()
        ctx = a.ctx if a.ctx and a.ctx > 0 else None
        lr = load_model(a.model, ctx, purpose="floor")
        if not lr.get("ok"):
            print("LOAD FAIL", lr, flush=True)
            return 1

    judge = Judge()
    seed_list = SEEDS[: a.seeds]
    results = {}
    t0 = time.time()

    for d in a.depths:
        print("\n" + "=" * 60, flush=True)
        print(f"IDENTITY CHAIN depth={d}  (2d rewrites: d fwd + d rev)", flush=True)
        print("=" * 60, flush=True)
        rows = []
        for s in seed_list:
            try:
                r = round_trip(a.model, judge, s, d, identity=True)
            except Exception as e:
                print(f"  ERR {e}", flush=True)
                r = None
            if r:
                rows.append(r)
                print(
                    f"  {r['mode']:<10} cos {r['cos']:.3f}  "
                    f"fwd {r['fwd'][:4]}/rev {r['rev'][:4]}  {s[:48]}",
                    flush=True,
                )
        if not rows:
            results[str(d)] = {"ok": False, "n": 0}
            continue
        fc = float(np.mean([r["closed"] for r in rows]))
        cos_gate = float(np.mean([r["cos"] >= 0.75 for r in rows]))
        modes = dict(Counter(r["mode"] for r in rows))
        med_cos = float(np.median([r["cos"] for r in rows]))
        print(
            f"\n  depth {d}  identity_closure {fc:.2f}  "
            f"cos>=0.75 {cos_gate:.2f}  modes {modes}",
            flush=True,
        )
        results[str(d)] = {
            "ok": True,
            "identity_closure": fc,
            "cosine_would_say": cos_gate,
            "median_cos": med_cos,
            "modes": modes,
            "n": len(rows),
            "rows": [{k: v for k, v in r.items() if k != "texts"} for r in rows],
        }

    report = {
        "model": a.model,
        "protocol": "identity-chain control: pure identity multi-hop, DeBERTa mutual entailment",
        "thesis": (
            "If p stays high under identity chain but drops under transform ladder, "
            "divergence from speculative-decoding alpha is about transform content, "
            "not missing anchor. If identity chain also decays, accumulation without "
            "fixed anchor is the mechanism."
        ),
        "depths": results,
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 60, flush=True)
    print(f"{'depth':>6} {'id_close':>8} {'cos.75':>7} {'med_cos':>8} {'modes'}", flush=True)
    for d in a.depths:
        e = results.get(str(d)) or {}
        print(
            f"{d:>6} {e.get('identity_closure', float('nan')):>8.2f} "
            f"{e.get('cosine_would_say', float('nan')):>7.2f} "
            f"{e.get('median_cos', float('nan')):>8.3f} "
            f"{e.get('modes')}",
            flush=True,
        )
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
