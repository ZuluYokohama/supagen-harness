"""
semantic_holonomy_v3.py -- closure judged by entailment, not similarity.

WHY v2 FAILED (measured, on v2's own 48 loops)
----------------------------------------------
v2 gated closure on cosine >= threshold. On granite's IDENTITY arm -- restate
a sentence with identical meaning, twice -- that gate scored:

    cosine >= 0.75  :  0.00   (0 of 6)
    mutual entailment: 0.83   (5 of 6)

Same six loops. Same texts. The metric was the entire disagreement:

    seed : Where the mismatch is written down is bookkeeping; the total is not.
    final: Where the variance is documented comprises bookkeeping; the total does not.
           cos 0.598 -> v2 says OPEN.  entailment both ways -> actually CLOSED.

v2 was measuring lexical overlap and reporting it as semantic closure. That is
the cosine null one level up, inside the instrument built to escape the cosine
null. Fifth instance of the same failure in this program.

WHAT v3 CHANGES
---------------
1. CLOSURE = BIDIRECTIONAL ENTAILMENT.  seed |= final AND final |= seed.
   Cosine is still reported, but only as a diagnostic. It never gates.

2. THE JUDGE IS INDEPENDENT OF THE SUBJECT.  NLI runs locally on
   cross-encoder/nli-deberta-v3-base (184M, CPU, ~30ms/pair), NOT through the
   model under test. A 1.2B model cannot be both the thing measured and the
   thing measuring.
   Measured on a 65-pair discriminating eval: 95.2% on high-overlap
   contradictions -- the exact cell cosine is structurally blind to.
   Known limit: 16.7% on low-overlap inferential entailment. It judges
   "same claim?", not "does this follow from domain law?".

3. DIRECTION IS REPORTED SEPARATELY.  seed|=final without the converse means
   content was DROPPED. The converse alone means content was ADDED
   (hallucination). Both collapse to one number under cosine.
   Real example: LFM depth-1 scored one-way 0.69, mutual 0.15. It drops.
   Granite's floor scored 0.83/0.83. It preserves.

4. LADDER STARTS AT DEPTH 2.  A depth-1 round trip is 3 nodes, i.e. a
   triangle, which is planar, so Bishop holonomy is identically 0.000 -- and
   that is exactly what v2 measured. Depth 1 is kept as the GATE only.

USAGE
    python semantic_holonomy_v3.py --model ibm/granite-4-h-tiny
    python semantic_holonomy_v3.py --model X --floor-only     # gate in ~2 min
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from collections import Counter
from numpy.linalg import norm

BASE = "http://127.0.0.1:1234"
# Job1 default is jina (nomic_metric); holonomy cos is diagnostic only — never gates.
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

PAIRS = [
    ("Rewrite this more abstractly. Remove specific nouns, keep the logical form.",
     "Rewrite this more concretely. Restore specific referents, keep the logical form."),
    ("Rewrite this in formal academic register.",
     "Rewrite this in plain spoken English."),
    ("Rewrite this as a longer, fully explicit statement.",
     "Rewrite this as one compact sentence."),
    ("Rewrite this using vocabulary from a different technical field.",
     "Rewrite this in neutral, field-independent vocabulary."),
]
IDENTITY = ("Restate this sentence with identical meaning, changing at most two words.",) * 2

SYSTEM = ("Output only the rewritten sentence. No preamble, no quotes, no commentary "
          "about the sentence or about what changed. One sentence.")

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

META = ("the sentence", "the statement", "the phrase", "the claim change", "the idea is",
        "the concept", "the context", "the situation", "the scenario", "the observation",
        "we observe", "we analyze", "we examine", "we deduce", "we notice", "we begin",
        "first, we", "the analysis", "this is like", "imagine", "consider a", "think of")


# ---------------------------------------------------------------- the judge
class Judge:
    """Independent NLI. Never the model under test."""

    def __init__(self, name=NLI_MODEL):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(name)
        self.lab = self.m.model.config.id2label

    def relation(self, a, b):
        s = self.m.predict([(a, b), (b, a)])
        return self.lab[int(np.argmax(s[0]))], self.lab[int(np.argmax(s[1]))]

    def verdict(self, a, b):
        f, r = self.relation(a, b)
        closed = (f == "entailment" and r == "entailment")
        if closed:
            mode = "closed"
        elif f == "entailment":
            mode = "dropped"        # seed |= final only: content lost
        elif r == "entailment":
            mode = "added"          # final |= seed only: content invented
        elif "contradiction" in (f, r):
            mode = "inverted"       # the claim flipped
        else:
            mode = "unrelated"
        return dict(fwd=f, rev=r, closed=closed, mode=mode)


# ---------------------------------------------------------------- endpoints
def _post(path, body, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed(t):
    """Aboutness vector via nomic_metric (jina auto-ensure + family prefixes)."""
    try:
        from nomic_metric import embed as job1_embed

        r = job1_embed(t, task="clustering", use_cache=True)
        if r.get("ok") and r.get("embedding"):
            v = np.asarray(r["embedding"], float)
            return v / max(norm(v), 1e-12)
    except Exception:
        pass
    # last-resort LMS nomic (legacy)
    r = _post(
        "/v1/embeddings",
        {
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": "clustering: " + (t or "").strip(),
        },
        90,
    )
    v = np.asarray(r["data"][0]["embedding"], float)
    return v / max(norm(v), 1e-12)


def rewrite(model, instr, text, temp=0.0):
    # omit context_length — avoids LMS reload/OOM when multiple LLMs resident
    out = _post("/api/v1/chat", {
        "model": model, "system_prompt": SYSTEM,
        "input": f"{instr}\n\nSENTENCE: {text}", "temperature": temp,
        "max_output_tokens": 120, "store": False}).get("output")
    if isinstance(out, str):
        return out.strip()
    parts = []
    for b in out or []:
        c = b.get("content") if isinstance(b, dict) else None
        if isinstance(c, list):
            parts += [x.get("text", "") for x in c if isinstance(x, dict)]
        elif isinstance(c, str):
            parts.append(c)
    return " ".join(parts).strip()


def is_meta(s):
    return s.strip().strip('"').lower().startswith(META)


def step(model, instr, text, tries=3):
    for i in range(tries):
        o = rewrite(model, instr, text, 0.0 if i == 0 else 0.3)
        if o and not is_meta(o) and len(o) > 12:
            return o
    return None


# ---------------------------------------------------------------- geometry
def _rot(u, v):
    c = float(np.clip(u @ v, -1, 1)); ax = np.cross(u, v); s = norm(ax)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    ax = ax / s; th = math.atan2(s, c)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def bishop(P):
    n = len(P); T = np.array([P[(i + 1) % n] - P[i] for i in range(n)])
    nz = norm(T, axis=1, keepdims=True)
    if (nz < 1e-9).any():
        return float("nan")
    T = T / nz
    u = np.cross(T[0], [0., 0., 1.])
    if norm(u) < 1e-8:
        u = np.cross(T[0], [0., 1., 0.])
    u0 = u / norm(u); u = u0.copy()
    for i in range(n):
        u = _rot(T[i], T[(i + 1) % n]) @ u
    u -= (u @ T[0]) * T[0]; u /= norm(u)
    return float(math.atan2(float(np.cross(u0, u) @ T[0]), float(np.clip(u0 @ u, -1, 1))))


def proj3(E):
    X = E - E.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:3].T


# ---------------------------------------------------------------- one loop
def round_trip(model, judge, seed, depth, identity=False):
    pairs = [IDENTITY] * depth if identity else [PAIRS[i % len(PAIRS)] for i in range(depth)]
    texts = [seed]; cur = seed
    for f, _ in pairs:
        cur = step(model, f, cur)
        if cur is None:
            return None
        texts.append(cur)
    for _, g in reversed(pairs):
        cur = step(model, g, cur)
        if cur is None:
            return None
        texts.append(cur)

    v = judge.verdict(texts[0], texts[-1])          # <-- the gate
    E = np.array([embed(t) for t in texts])
    consec = [float(E[i] @ E[i + 1]) for i in range(len(E) - 1)]
    seedsim = [float(E[0] @ e) for e in E[1:]]
    tail = min(consec[len(consec) // 2:]) > 0.95 if len(consec) > 2 else False
    th = bishop(proj3(E)); N = len(texts)
    return dict(depth=depth, N=N, texts=texts, **v,
                cos=float(E[0] @ E[-1]),            # diagnostic only
                theta=th, lam=(2 - 2 * math.cos(th / N)) if np.isfinite(th) else float("nan"),
                collapsed=bool(tail and max(seedsim[len(seedsim) // 2:]) < 0.75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm/granite-4-h-tiny")
    ap.add_argument("--depths", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--floor-only", action="store_true")
    ap.add_argument("--out", default="holonomy_v3.json")
    a = ap.parse_args()

    judge = Judge()
    seeds = SEEDS[:a.seeds]

    print("STEP 1  identity floor + depth-1 gate  (closure = mutual entailment)\n")
    floor = []
    for s in seeds:
        r = round_trip(a.model, judge, s, 1, identity=True)
        if r:
            floor.append(r)
            print(f"  {r['mode']:<10} cos {r['cos']:.3f}  fwd {r['fwd'][:4]}/rev {r['rev'][:4]}  {s[:44]}")
    if not floor:
        print("  identity arm produced nothing. Model will not follow instructions.")
        return 1
    fc = float(np.mean([r["closed"] for r in floor]))
    print(f"\n  identity closure  {fc:.2f}   (cosine>=0.75 would have said "
          f"{np.mean([r['cos'] >= 0.75 for r in floor]):.2f})")
    print(f"  failure modes     {dict(Counter(r['mode'] for r in floor))}")
    if fc < 0.80:
        print("\n  GATE FAILED at the identity arm. This model cannot restate a")
        print("  sentence without changing what it claims. Nothing deeper is")
        print("  measurable and no theta below would mean anything.")
        if not a.floor_only:
            print("  Stopping.")
        Path(a.out).write_text(json.dumps({"model": a.model, "identity_closure": fc,
                                           "gated": True}, indent=2))
        return 1
    print("  GATE PASSED.")
    if a.floor_only:
        return 0

    print("\nSTEP 2  ladder (depth 1 excluded: 3 nodes is planar, theta == 0 by construction)\n")
    xs, ys, rows = [], [], []
    for d in a.depths:
        got = [r for r in (round_trip(a.model, judge, s, d) for s in seeds) if r]
        if not got:
            continue
        ok = [r for r in got if not r["collapsed"]]
        rate = float(np.mean([r["closed"] for r in ok])) if ok else 0.0
        xs.append(d); ys.append(rate); rows += got
        print(f"  depth {d}  n={len(got)}  closure {rate:.2f}  "
              f"modes {dict(Counter(r['mode'] for r in got))}  "
              f"median|theta| {np.median([abs(r['theta']) for r in got]):.3f}")

    print("\n" + "=" * 70)
    x, y = np.array(xs, float), np.array(ys)
    m = (y > 1e-4) & (y < 1 - 1e-4)
    if m.sum() >= 3:
        z = np.log(y[m] / (1 - y[m])); A = np.vstack([x[m], np.ones(m.sum())]).T
        k, b = np.linalg.lstsq(A, z, rcond=None)[0]
        print(f"  drift tolerance depth* = {-b/k:.2f}")
        print(f"  logistic slope         = {k:.3f}")
        print(f"  implied sigma          = {(-1/k)/0.47:.4f}")
        print("  Run other models and compare depth* to trace the capacity route.")
    else:
        print(f"  only {m.sum()} interior points; ladder saturated. Widen --depths.")
    print("=" * 70)

    Path(a.out).write_text(json.dumps(
        {"model": a.model, "identity_closure": fc, "depths": xs, "closure": ys,
         "rows": [{k: v for k, v in r.items() if k != "texts"} for r in rows]}, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
