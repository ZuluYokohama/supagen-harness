"""
score_nli_eval.py -- settle whether the agreement channel is real.

Three pairs is a smoke test. This is the null.

The eval set is built to SEPARATE entailment from aboutness. Cells:

  con_high   contradiction with high lexical overlap   <-- cosine CANNOT do this
  ent_low    entailment with low lexical overlap       <-- cosine CANNOT do this
  neu_high   neutral with high lexical overlap         <-- cosine says "similar"
  con_low / ent_high / neu_low                          control cells

A cosine metric scores near chance on con_high and ent_low by construction.
A working entailment channel does not. That gap is the whole measurement.

Usage
-----
    python score_nli_eval.py                          # LFM-NLI via LM Studio
    python score_nli_eval.py --model liquid/lfm2.5-1.2b
    python score_nli_eval.py --also-cosine            # run cosine as the null arm

Reports per-cell accuracy, a 3x3 confusion matrix, and the two numbers that
matter: accuracy on con_high and on ent_low.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE = "http://127.0.0.1:1234"
LABELS = ("entailment", "contradiction", "neutral")

# reason BEFORE label: the field order that flipped 7/7-neutral to 3/3 correct.
# do not reorder. do not add "prefer neutral when unsure".
SYSTEM = (
    "ROLE=NLI. Decide the logical relation between PREMISE and HYPOTHESIS.\n"
    "Think first, then commit. Output JSON only, exactly these keys in this order:\n"
    '{"reason": "<one sentence of analysis>", '
    '"label": "<entailment|contradiction|neutral>", '
    '"confidence": <0.0-1.0>}\n'
    "entailment: the premise makes the hypothesis true.\n"
    "contradiction: the premise makes the hypothesis false.\n"
    "neutral: the premise neither establishes nor refutes the hypothesis.\n"
    "Emit one of the three label strings literally. Never emit the enum template."
)


def post(path: str, body: dict, timeout: int = 180) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_text(resp: dict) -> str:
    out = resp.get("output")
    if isinstance(out, str):
        return out
    chunks = []
    for blk in out or []:
        if isinstance(blk, dict):
            c = blk.get("content") or blk.get("text")
            if isinstance(c, list):
                chunks += [x.get("text", "") for x in c if isinstance(x, dict)]
            elif isinstance(c, str):
                chunks.append(c)
    return "\n".join(chunks)


def parse_label(text: str) -> tuple[str, float, str]:
    """Return (label, confidence, reason). Enum-template copies count as invalid."""
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            lab = str(d.get("label", "")).strip().lower()
            # reject enum-template copying, e.g. "entailment|neutral"
            if lab in LABELS:
                return lab, float(d.get("confidence") or 0.0), str(d.get("reason", ""))
            return "INVALID", 0.0, str(d.get("reason", ""))
        except Exception:
            pass
    low = s.lower()
    hits = [l for l in LABELS if l in low]
    return (hits[0] if len(hits) == 1 else "INVALID"), 0.0, s[:160]


def nli(model: str, premise: str, hypothesis: str) -> tuple[str, float, str]:
    body = {
        "model": model,
        "system_prompt": SYSTEM,
        "input": f"PREMISE: {premise}\nHYPOTHESIS: {hypothesis}",
        "temperature": 0.0,
        "max_output_tokens": 200,
        "context_length": 2048,
        "store": False,          # no fiber chaining: each pair is independent
    }
    return parse_label(extract_text(post("/api/v1/chat", body)))


def embed(text: str, model: str = "text-embedding-nomic-embed-text-v1.5") -> list[float]:
    r = post("/v1/embeddings", {"model": model, "input": text}, timeout=90)
    return r["data"][0]["embedding"]


def cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(Path(__file__).with_name("nli_eval_v1.jsonl")))
    ap.add_argument("--model", default="liquid/lfm2.5-1.2b")
    ap.add_argument("--also-cosine", action="store_true",
                    help="run cosine as the null arm for comparison")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="nli_eval_results.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print(f"eval: {len(rows)} pairs  model: {a.model}\n")

    results, t0 = [], time.time()
    for k, r in enumerate(rows, 1):
        try:
            lab, conf, reason = nli(a.model, r["premise"], r["hypothesis"])
        except Exception as e:
            lab, conf, reason = "ERROR", 0.0, str(e)[:120]
        ok = lab == r["gold"]
        results.append({**r, "pred": lab, "conf": conf, "ok": ok, "reason": reason[:200]})
        mark = "." if ok else ("!" if lab in LABELS else "?")
        print(mark, end="", flush=True)
        if k % 40 == 0:
            print(f"  {k}/{len(rows)}")
    dt = time.time() - t0
    print(f"\n\ncompleted in {dt:.0f}s ({dt/max(len(rows),1):.1f}s/pair)\n")

    # ---- per cell -----------------------------------------------------------
    cells = defaultdict(list)
    for r in results:
        cells[r["cell"]].append(r["ok"])
    print("=" * 62)
    print("PER-CELL ACCURACY   (con_high and ent_low are the discriminating cells)")
    print("=" * 62)
    order = ["con_high", "ent_low", "neu_high", "con_low", "ent_high", "neu_low"]
    for c in order:
        v = cells.get(c) or []
        if not v:
            continue
        acc = sum(v) / len(v)
        star = "  <-- cosine cannot do this" if c in ("con_high", "ent_low") else ""
        print(f"  {c:10s} {sum(v):2d}/{len(v):2d}  {acc:6.1%}{star}")

    # ---- confusion ----------------------------------------------------------
    conf_m = Counter((r["gold"], r["pred"]) for r in results)
    preds = sorted({r["pred"] for r in results})
    print("\n" + "=" * 62)
    print("CONFUSION   rows = gold, cols = predicted")
    print("=" * 62)
    print(f"  {'':16s}" + "".join(f"{p[:9]:>11s}" for p in preds))
    for g in LABELS:
        print(f"  {g:16s}" + "".join(f"{conf_m[(g, p)]:>11d}" for p in preds))

    # ---- verdict ------------------------------------------------------------
    overall = sum(r["ok"] for r in results) / len(results)
    ch = cells.get("con_high") or []
    el = cells.get("ent_low") or []
    ch_acc = sum(ch) / len(ch) if ch else 0.0
    el_acc = sum(el) / len(el) if el else 0.0
    invalid = sum(1 for r in results if r["pred"] not in LABELS)
    never = [l for l in LABELS if not any(r["pred"] == l for r in results)]

    print("\n" + "=" * 62)
    print(f"  overall            {overall:6.1%}")
    print(f"  con_high           {ch_acc:6.1%}   (chance ~33%)")
    print(f"  ent_low            {el_acc:6.1%}   (chance ~33%)")
    print(f"  invalid / enum-copy {invalid:3d}")
    if never:
        print(f"  NEVER EMITTED: {never}  <-- collapsed channel, same failure as 7/7 neutral")
    earned = ch_acc >= 0.70 and el_acc >= 0.70 and not never and invalid <= len(results) * 0.05
    print("\n  VERDICT: " + (
        "EARNED. The channel decides something cosine cannot."
        if earned else
        "NOT EARNED. Reason-first fixed the smoke test, not the instrument.\n"
        "           Next: constrained decoding on the enum, then DeBERTa-MNLI."
    ))
    print("=" * 62)

    # ---- optional cosine null arm ------------------------------------------
    if a.also_cosine:
        print("\nNULL ARM: cosine on the same pairs (prefixed, envelope-free)")
        by = defaultdict(list)
        for r in results:
            try:
                c = cos(embed("search_query: " + r["premise"]),
                        embed("search_document: " + r["hypothesis"]))
            except Exception:
                continue
            by[r["gold"]].append(c)
        for g in LABELS:
            v = by.get(g) or []
            if v:
                print(f"  {g:14s} n={len(v):2d}  mean cos {sum(v)/len(v):.3f}"
                      f"  range {min(v):.3f}-{max(v):.3f}")
        e, c_ = by.get("entailment") or [0], by.get("contradiction") or [0]
        gap = abs(sum(e)/len(e) - sum(c_)/len(c_))
        print(f"\n  entailment vs contradiction separation: {gap:.3f}")
        print("  (near zero confirms cosine has no contradiction channel)")

    # Provenance + power warnings (do not overclaim on n=12/21 model golds)
    n_ch, n_el = len(ch), len(el)
    print("\n" + "=" * 62)
    print("PROVENANCE / POWER")
    print("=" * 62)
    srcs = Counter(str(r.get("label_source") or "unspecified") for r in results)
    print(f"  label_source counts: {dict(srcs)}")
    print(f"  con_high n={n_ch}  ent_low n={n_el}  (small n → wide CI; 70% bar may straddle)")
    print("  If golds are model-written, this grades model vs model — Step 0 unverified.")
    if any((r.get("label_source") or "").startswith("model") for r in results):
        print("  WARNING: model-sourced golds present; do not treat as final instrument verdict.")

    Path(a.out).write_text(json.dumps({
        "model": a.model, "n": len(results), "overall": overall,
        "con_high": ch_acc, "ent_low": el_acc, "invalid": invalid,
        "never_emitted": never, "earned": earned,
        "con_high_n": n_ch, "ent_low_n": n_el,
        "label_source_counts": dict(srcs),
        "seconds": round(dt, 1), "results": results,
        "epistemic_note": (
            "Earned bar requires con_high&ent_low >=70% AND adequate n AND "
            "human-verified golds on discriminating cells. 3/3 smoke is not this."
        ),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0 if earned else 1


if __name__ == "__main__":
    sys.exit(main())
