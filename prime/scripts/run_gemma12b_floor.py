#!/usr/bin/env python3
"""Gemma-12B identity floor via llama-server /completion (channel-aware).

LMS chat returns empty for this GGUF (gemma4 channel format). Cosine omitted —
DeBERTa mutual-entailment is the gate.
"""
from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

BASE = "http://127.0.0.1:8766"
NLI = "cross-encoder/nli-deberta-v3-base"
IDENTITY = "Restate this sentence with identical meaning, changing at most two words."
SYSTEM = (
    "Output only the rewritten sentence. No preamble, no quotes, no commentary. "
    "One sentence. No thinking aloud."
)
SEEDS = [
    "A claim may not open unless a restriction was recorded before the measures.",
    "Strain that no choice of dihedrals can remove belongs to the ring, not to any residue.",
    "The decoder was offline while the tool recorded fifty interference events downhole.",
    "An instrument that returns the same value on every input carries no information.",
    "Sequence sets the cost of ring closure and has no bearing on its feasibility.",
    "Where the mismatch is written down is bookkeeping; the total is not.",
    "A lift earns its keep only if something can be wrong upstairs that could not be wrong downstairs.",
    "Longer chains tolerate more accumulated drift before they stop cohering.",
]
META = (
    "the sentence",
    "the statement",
    "the phrase",
    "the claim change",
    "the idea is",
    "the concept",
    "the context",
    "the situation",
    "the scenario",
    "the observation",
    "we observe",
    "we analyze",
    "we examine",
    "we deduce",
    "we notice",
    "we begin",
    "first, we",
    "the analysis",
    "this is like",
    "imagine",
    "consider a",
    "think of",
)


def post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def strip_channels(s: str) -> str:
    if not s:
        return ""
    # Final visible answer usually after last <channel|>
    if "<channel|>" in s:
        s = s.split("<channel|>")[-1]
    # Drop residual markers / thought debris
    for m in (
        "<|channel>thought",
        "<channel|>",
        "<end_of_turn>",
        "<start_of_turn>",
        "model\n",
    ):
        s = s.replace(m, "")
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    lines = [
        ln
        for ln in lines
        if not ln.lower().startswith(
            ("the user", "i need", "the request", "should ", "response")
        )
    ]
    if not lines:
        return ""
    # Prefer the longest line that looks like a sentence
    cands = [ln for ln in lines if len(ln) > 12]
    if not cands:
        return lines[-1]
    return max(cands, key=len)


def complete(prompt: str, n: int = 100, temp: float = 0.0):
    d = post(
        "/completion",
        {
            "prompt": prompt,
            "n_predict": n,
            "temperature": temp,
            "stop": ["<end_of_turn>", "<start_of_turn>user"],
        },
    )
    raw = d.get("content") or ""
    return strip_channels(raw), raw, d.get("tokens_predicted")


def rewrite(instr: str, text: str, temp: float = 0.0):
    prompt = (
        f"<start_of_turn>user\n{SYSTEM}\n\n{instr}\n\nSENTENCE: {text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return complete(prompt, 100, temp)


def is_meta(s: str) -> bool:
    return s.strip().strip('"').lower().startswith(META)


def step(instr: str, text: str, tries: int = 3):
    for i in range(tries):
        o, raw, n = rewrite(instr, text, 0.0 if i == 0 else 0.3)
        print(f"   try {i} tok={n} out={o!r:.120}", flush=True)
        if o and not is_meta(o) and len(o) > 12:
            return o
    return None


def main() -> int:
    print("loading DeBERTa…", flush=True)
    judge = CrossEncoder(NLI)
    lab = judge.model.config.id2label

    def verdict(a: str, b: str) -> dict:
        s = judge.predict([(a, b), (b, a)])
        f, r = lab[int(np.argmax(s[0]))], lab[int(np.argmax(s[1]))]
        closed = f == "entailment" and r == "entailment"
        if closed:
            mode = "closed"
        elif f == "entailment":
            mode = "dropped"
        elif r == "entailment":
            mode = "added"
        elif "contradiction" in (f, r):
            mode = "inverted"
        else:
            mode = "unrelated"
        return dict(fwd=f, rev=r, closed=closed, mode=mode)

    w, raw, _ = rewrite(IDENTITY, SEEDS[0])
    print("warm", repr(w)[:200], flush=True)
    print("warm_raw", repr(raw)[:240], flush=True)

    rows = []
    t0 = time.time()
    for i, seed in enumerate(SEEDS):
        print(f"\nSEED {i+1}/8 {seed[:56]}", flush=True)
        mid = step(IDENTITY, seed)
        if not mid:
            print("  FAIL forward", flush=True)
            continue
        final = step(IDENTITY, mid)
        if not final:
            print("  FAIL reverse", flush=True)
            continue
        v = verdict(seed, final)
        rows.append({**v, "seed": seed, "mid": mid, "final": final})
        print(
            f"  {v['mode']:<10} {v['fwd'][:4]}/{v['rev'][:4]}  final={final[:90]}",
            flush=True,
        )

    fc = float(np.mean([r["closed"] for r in rows])) if rows else 0.0
    modes = dict(Counter(r["mode"] for r in rows))
    report = {
        "model": "gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2@q4_k_m",
        "backend": "llama-server :8766 ctx=512 flash_attn",
        "identity_closure": fc,
        "n": len(rows),
        "modes": modes,
        "gate_failed": fc < 0.80,
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
        "note": (
            "Cosine omitted (no nomic co-resident). LMS /api/v1/chat returned empty "
            "or 500 under RAM pressure; used llama-server /completion + channel strip."
        ),
    }
    out = Path(__file__).resolve().parent.parent / "state" / "holonomy_v3_gemma12b_floor.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== RESULT ===", flush=True)
    gate = "FAIL" if fc < 0.80 else "PASS"
    print(
        f"identity_closure {fc:.2f}  n={len(rows)}  modes={modes}  gate={gate}",
        flush=True,
    )
    print("wrote", out, flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
