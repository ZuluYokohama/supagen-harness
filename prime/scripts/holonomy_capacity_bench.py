"""
Capacity / precision / training axis — v3 identity floor only.

Order by information per minute (sister protocol):
  1. thinking-1.2b vs plain 1.2b  (same size/quant — training only)
  2. 230m F16
  3. frankenstein 7.2B Q4
  4. bonsai 27B Q1
  5. queen-opus MoE 8B-A1B Q4

MUST unload all LLMs before each load (16GB Snapdragon death otherwise).
Keep nomic resident.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lms_layers import DEFAULT_BASE, DEFAULT_EMBED, l0_post, l1_catalog, l1_free_ram_gb
from semantic_holonomy_v3 import Judge, SEEDS, round_trip
from collections import Counter
import numpy as np

# (short_name, lms_key, context_length)
BENCH = [
    ("lfm-1.2b-plain", "liquid/lfm2.5-1.2b", 128000),
    ("lfm-1.2b-thinking", "lfm2.5-1.2b-thinking-claude-4.6-opus-heretic-uncensored-distill", 128000),
    ("lfm-230m-f16", "lfm2.5-230m-code-math-exp", 32768),
    ("frankenstein-7b-q4", "frankenstein-2.0-i1", 16384),
    ("bonsai-27b-q1", "prism-ml/bonsai-27b", 8192),
    ("queen-opus-moe-a1b", "lfm2.5-queen-opus-4.7-8b-a1b-i1", 16384),
]


def unload_all_llms(base: str = DEFAULT_BASE) -> list:
    acts = []
    cat = l1_catalog(base=base)
    for m in cat.get("models") or []:
        if m.get("type") == "embedding":
            continue
        for inst in m.get("loaded_instances") or []:
            r = l0_post(
                "/api/v1/models/unload",
                {"instance_id": inst["id"]},
                base=base,
                timeout=180,
            )
            acts.append({"unload": inst["id"], "ok": r.ok, "err": r.error})
            print(f"  unload {inst['id']}: {r.ok}", flush=True)
    return acts


def ensure_embed(base: str = DEFAULT_BASE) -> None:
    cat = l1_catalog(base=base)
    for m in cat.get("models") or []:
        if m.get("key") == DEFAULT_EMBED and m.get("loaded"):
            return
    r = l0_post(
        "/api/v1/models/load",
        {"model": DEFAULT_EMBED, "context_length": 2048},
        base=base,
        timeout=180,
    )
    print(f"  load embed: {r.ok}", flush=True)


def load_model(key: str, ctx: int | None = None, base: str = DEFAULT_BASE, purpose: str = "chat") -> dict:
    free = l1_free_ram_gb()
    print(f"  free_gb before load: {free}", flush=True)
    try:
        from ctx_policy import resolve_load_context

        pol = resolve_load_context(key, ctx, purpose=purpose, free_gb=free, base=base)
        ctx = int(pol["context_length"])
        print(f"  ctx_policy: {pol.get('reading')}", flush=True)
    except Exception as e:
        ctx = int(ctx or 32768)
        print(f"  ctx_policy fallback ctx={ctx} ({e})", flush=True)
        pol = {"context_length": ctx}
    body = {"model": key, "context_length": int(ctx)}
    # try flash for small models
    if "1.2b" in key or "230m" in key:
        body["flash_attention"] = True
        body["eval_batch_size"] = 512
    r = l0_post("/api/v1/models/load", body, base=base, timeout=600)
    if not r.ok and "flash" in body:
        body.pop("flash_attention", None)
        r = l0_post("/api/v1/models/load", body, base=base, timeout=600)
    if not r.ok:
        # half-ctx retry (RAM)
        half = max(4096, int(ctx) // 2)
        body["context_length"] = half
        r = l0_post("/api/v1/models/load", body, base=base, timeout=600)
        if r.ok:
            ctx = half
    print(f"  load {key} ctx={ctx}: ok={r.ok} {r.error or r.data}", flush=True)
    return {
        "ok": r.ok,
        "error": r.error,
        "data": r.data,
        "body": body,
        "free_gb": free,
        "context_length": ctx,
        "ctx_policy": pol,
    }


def floor_v3(model: str, seeds: int = 8) -> dict:
    judge = Judge()
    seed_list = SEEDS[:seeds]
    floor = []
    for s in seed_list:
        try:
            r = round_trip(model, judge, s, 1, identity=True)
        except Exception as e:
            print(f"  ERR {e}", flush=True)
            r = None
        if r:
            floor.append(r)
            print(
                f"  {r['mode']:<10} cos {r['cos']:.3f}  "
                f"fwd {r['fwd'][:4]}/rev {r['rev'][:4]}  {s[:48]}",
                flush=True,
            )
    if not floor:
        return {"ok": False, "identity_closure": 0.0, "n": 0, "error": "no_floor_rows"}
    fc = float(np.mean([r["closed"] for r in floor]))
    cos_gate = float(np.mean([r["cos"] >= 0.75 for r in floor]))
    modes = dict(Counter(r["mode"] for r in floor))
    med_cos = float(np.median([r["cos"] for r in floor]))
    print(f"\n  identity_closure (mutual ent) {fc:.2f}   cosine>=0.75 {cos_gate:.2f}", flush=True)
    print(f"  modes {modes}  median_cos {med_cos:.3f}", flush=True)
    gated = fc < 0.80
    if gated:
        print("  GATE FAILED — cannot restate without changing claim.", flush=True)
    else:
        print("  GATE PASSED.", flush=True)
    return {
        "ok": True,
        "identity_closure": fc,
        "cosine_would_say": cos_gate,
        "median_cos": med_cos,
        "modes": modes,
        "n": len(floor),
        "gate_failed": gated,
        "rows": [{k: v for k, v in r.items() if k != "texts"} for r in floor],
    }


def main():
    out_path = Path(__file__).resolve().parent.parent / "state" / "holonomy_capacity_bench.json"
    results = []
    t0 = time.time()
    only = [x for x in sys.argv[1:] if not x.startswith("-")]
    bench = BENCH
    if only:
        bench = [b for b in BENCH if b[0] in only or b[1] in only]

    for name, key, ctx in bench:
        print("\n" + "=" * 70, flush=True)
        print(f"MODEL {name}  key={key}  ctx={ctx}", flush=True)
        print("=" * 70, flush=True)
        unload_all_llms()
        ensure_embed()
        lr = load_model(key, ctx)
        if not lr.get("ok"):
            results.append({"name": name, "key": key, "load_ok": False, "error": lr.get("error")})
            continue
        t1 = time.time()
        try:
            floor = floor_v3(key, seeds=8)
        except Exception as e:
            floor = {"ok": False, "error": str(e)}
            print(f"  floor exception: {e}", flush=True)
        elapsed = time.time() - t1
        entry = {
            "name": name,
            "key": key,
            "ctx": ctx,
            "load_ok": True,
            "load": lr,
            "floor": floor,
            "seconds": round(elapsed, 1),
            "free_gb_after": l1_free_ram_gb(),
        }
        results.append(entry)
        # unload this LLM immediately
        unload_all_llms()
        ensure_embed()
        # progressive save
        out_path.write_text(
            json.dumps({"results": results, "elapsed_total": round(time.time() - t0, 1)}, indent=2),
            encoding="utf-8",
        )
        print(f"  saved partial → {out_path}", flush=True)

    # summary table
    print("\n" + "=" * 70, flush=True)
    print(f"{'name':<22} {'id_close':>8} {'cos.75':>7} {'med_cos':>8} {'gate':>6} {'s':>6}", flush=True)
    for e in results:
        f = e.get("floor") or {}
        print(
            f"{e['name']:<22} {f.get('identity_closure', float('nan')):>8.2f} "
            f"{f.get('cosine_would_say', float('nan')):>7.2f} "
            f"{f.get('median_cos', float('nan')):>8.3f} "
            f"{'FAIL' if f.get('gate_failed') else ('PASS' if f.get('ok') else 'ERR'):>6} "
            f"{e.get('seconds', 0):>6.0f}",
            flush=True,
        )
    print("=" * 70, flush=True)
    out_path.write_text(
        json.dumps(
            {
                "results": results,
                "elapsed_total": round(time.time() - t0, 1),
                "protocol": "v3 identity floor, DeBERTa judge, unload-before-load",
                "thesis": (
                    "thinking vs plain 1.2b isolates training; "
                    "bonsai vs frankenstein isolates params vs precision; "
                    "queen isolates MoE active compute"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}", flush=True)
    # leave LFM max resident for daily work
    print("\nRestoring liquid/lfm2.5-1.2b @ max…", flush=True)
    unload_all_llms()
    ensure_embed()
    load_model("liquid/lfm2.5-1.2b", 128000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
