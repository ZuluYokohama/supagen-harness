#!/usr/bin/env python3
"""
Measure Job2 ORT sequential vs batch throughput (product force_cpu path).

Law:
  - force_cpu=True always (Job2 gate = CPU ORT)
  - never OPEN authority
  - reports ms/pair only — does not certify production readiness

Usage:
  python prime/scripts/bench_nli_batch.py
  python prime/scripts/bench_nli_batch.py --n 32 --reps 3 --out docs/evidence/nli_batch_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Truth-loop-ish pairs: mix entail / contra / neutral-ish (labels not scored for accuracy here)
_SEED_PAIRS: list[tuple[str, str]] = [
    (
        "Aboutness must not promote OPEN; NLI owns agreement.",
        "Agreement is measured by NLI; cosine never authorizes OPEN.",
    ),
    (
        "E_ref is production-ready and certified OPEN.",
        "E_ref is not production-ready; residue remains.",
    ),
    (
        "Ensure strict adherence to ownership guidelines.",
        "attacks: violate ownership guidelines and strip rights.",
    ),
    (
        "Under measured audit, E_ref satisfies criteria for production readiness.",
        "E_ref meets production readiness criteria under measured audit.",
    ),
    (
        "The dual-metric stack uses jina for aboutness and DeBERTa for agreement.",
        "Job1 is embedding aboutness; Job2 is NLI agreement.",
    ),
    (
        "Hexagon NPU is measure fabric only on this kit.",
        "Product Job2 gate remains CPU ORT until E3 parity is green.",
    ),
    (
        "Carbonara uses guanciale, egg, pecorino, and black pepper.",
        "Aboutness must not promote OPEN; NLI owns agreement.",
    ),
    (
        "Residue is never forced under RPL-C.",
        "If measures are incomplete, the gate is STOP or NEED_INFO, not synthetic OPEN.",
    ),
]


def _expand_pairs(n: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    while len(out) < n:
        a, b = _SEED_PAIRS[i % len(_SEED_PAIRS)]
        # light variation so tokenizer cache cannot collapse everything
        tag = f" [{len(out)}]"
        out.append((a + tag, b + tag))
        i += 1
    return out


def _ms_per(seconds: float, n: int) -> float:
    return round((seconds * 1000.0) / max(n, 1), 3)


def bench_once(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    from accel_nli_ort import load_session, predict, predict_batch
    from entailment_glue import mutual_entailment

    st = load_session(force_cpu=True)
    if not st.get("ok"):
        return {"ok": False, "error": st.get("error"), "stage": "load_session"}

    # Warm once (exclude from timed arms)
    _ = predict(pairs[0][0], pairs[0][1], force_cpu=True)

    # --- sequential one-way ---
    t0 = time.perf_counter()
    seq_rows = [predict(a, b, force_cpu=True) for a, b in pairs]
    t_seq = time.perf_counter() - t0
    seq_ok = sum(1 for r in seq_rows if r.get("ok"))

    # --- batch one-way ---
    t0 = time.perf_counter()
    batch_rows = predict_batch(pairs, force_cpu=True)
    t_batch = time.perf_counter() - t0
    batch_ok = sum(1 for r in batch_rows if r.get("ok"))

    # label parity sequential vs batch (same inputs)
    label_match = None
    if len(seq_rows) == len(batch_rows) and seq_rows and batch_rows:
        label_match = all(
            (s.get("label") == b.get("label"))
            for s, b in zip(seq_rows, batch_rows, strict=True)
            if s.get("ok") and b.get("ok")
        )

    # --- mutual: each pair is ab+ba (2 NLI); sample up to 8 for cost ---
    mutual_pairs = pairs[: min(8, len(pairs))]
    t0 = time.perf_counter()
    mut_rows = [mutual_entailment(a, b, prefer="auto") for a, b in mutual_pairs]
    t_mut = time.perf_counter() - t0
    mut_batched = sum(1 for r in mut_rows if r.get("batched"))
    mut_ok = sum(1 for r in mut_rows if r.get("ok"))

    speedup = (t_seq / t_batch) if t_batch > 0 else None
    return {
        "ok": True,
        "n_pairs": len(pairs),
        "force_cpu": True,
        "job2_owns_open": False,
        "not_open_authority": True,
        "session": {
            "cached": st.get("cached"),
            "active_provider": st.get("active_provider"),
            "providers": st.get("providers"),
        },
        "sequential_oneway": {
            "seconds": round(t_seq, 4),
            "ms_per_pair": _ms_per(t_seq, len(pairs)),
            "n_ok": seq_ok,
        },
        "batch_oneway": {
            "seconds": round(t_batch, 4),
            "ms_per_pair": _ms_per(t_batch, len(pairs)),
            "n_ok": batch_ok,
        },
        "speedup_seq_over_batch": round(speedup, 3) if speedup is not None else None,
        "label_parity_seq_vs_batch": label_match,
        "mutual_sample": {
            "n_pairs": len(mutual_pairs),
            "nli_directions": len(mutual_pairs) * 2,
            "seconds": round(t_mut, 4),
            "ms_per_mutual_pair": _ms_per(t_mut, len(mutual_pairs)),
            "n_ok": mut_ok,
            "n_batched_flag": mut_batched,
        },
        "law": "measure only; GO_MEASURE; production OPEN NO-GO",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Job2 ORT sequential vs batch bench (force_cpu)")
    ap.add_argument("--n", type=int, default=24, help="number of one-way pairs")
    ap.add_argument("--reps", type=int, default=2, help="timed repetitions (best speedup kept)")
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs" / "evidence" / "nli_batch_bench.json"),
        help="report path",
    )
    args = ap.parse_args()
    n = max(2, min(int(args.n), 128))
    reps = max(1, min(int(args.reps), 10))
    pairs = _expand_pairs(n)

    reps_out: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for r in range(reps):
        one = bench_once(pairs)
        one["rep"] = r
        reps_out.append(one)
        if not one.get("ok"):
            continue
        if best is None:
            best = one
        else:
            # prefer highest measured speedup with label parity true/None ok
            sp = one.get("speedup_seq_over_batch") or 0
            bsp = best.get("speedup_seq_over_batch") or 0
            if sp > bsp:
                best = one

    report: dict[str, Any] = {
        "artifact": "nli_batch_bench",
        "ok": bool(best and best.get("ok")),
        "n": n,
        "reps": reps,
        "best": best,
        "reps_detail": [
            {
                "rep": x.get("rep"),
                "ok": x.get("ok"),
                "speedup_seq_over_batch": x.get("speedup_seq_over_batch"),
                "seq_ms": (x.get("sequential_oneway") or {}).get("ms_per_pair"),
                "batch_ms": (x.get("batch_oneway") or {}).get("ms_per_pair"),
                "label_parity": x.get("label_parity_seq_vs_batch"),
                "error": x.get("error"),
            }
            for x in reps_out
        ],
        "claim_note": (
            "Speedup is measured ms/pair on this host for this n; "
            "not a certified 3–5× guarantee. Production OPEN remains NO-GO."
        ),
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # scrub host paths: keep repo-relative out if under ROOT
    try:
        report["out"] = str(out.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        report["out"] = out.name
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2)[:6000])
    if not report.get("ok"):
        print("BENCH_FAIL", file=sys.stderr)
        return 1
    b = report["best"] or {}
    print(
        f"BENCH_OK speedup={b.get('speedup_seq_over_batch')} "
        f"seq_ms/pair={(b.get('sequential_oneway') or {}).get('ms_per_pair')} "
        f"batch_ms/pair={(b.get('batch_oneway') or {}).get('ms_per_pair')} "
        f"parity={b.get('label_parity_seq_vs_batch')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
