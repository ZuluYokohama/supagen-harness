"""
semantic_holonomy_v2.py -- round trips that actually close.

WHY v1 FAILED (from 48 real loops in the 2026-08-05 run)
--------------------------------------------------------
  rung 0 closed 37.5%   <- the NEAR-IDENTITY rung. median return cos 0.390.
  rung 5 closed 37.5%   <- better than rungs 2,3,4. non-monotone.
  theta flat across all rungs: 1.97 1.46 1.33 1.57 1.60 1.46
  logistic fit: midpoint -112 rungs, sigma 202  ->  no transition exists

Three causes, all mine:

1. THE LOOP NEVER CLOSED. v1 ran seed -> T -> T -> T -> CLOSE(last). CLOSE saw
   only the previous node, so at high drift it restated the *story* the model
   had wandered into, not the claim. Bishop transport on an open arc returns a
   number that is not holonomy.

2. REGISTER COLLAPSE READ AS CLOSURE. Rung 5 degenerated into a fixed frame
   ("We observe that...", "The analysis begins by..."). Self-similar text sits
   still in embedding space, so the most degraded rung scored as the most
   coherent. Same failure as an instrument returning a constant.

3. NO NOISE FLOOR. Nothing was calibrated, so no theta was interpretable.

WHAT v2 DOES INSTEAD
--------------------
The loop closes BY CONSTRUCTION, not by instruction. Each rung applies k
forward transforms and then their k semantic inverses in reverse order:

    seed -> f1 -> f2 -> ... -> fk -> gk -> ... -> g2 -> g1 -> should be seed

No step is ever told the seed, so closure is earned rather than leaked. Depth
k is the drift ladder. Loop length 2k+1, closed by design.

Then: identity arm first for the noise floor, collapse detection, meta
rejection, and an ABORT if depth 1 cannot close -- because if the easiest
round trip fails, nothing deeper means anything.

USAGE
    python semantic_holonomy_v2.py --model liquid/lfm2.5-1.2b
    python semantic_holonomy_v2.py --floor-only        # just the noise floor
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from numpy.linalg import norm

BASE = "http://127.0.0.1:1234"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
PREFIX = "clustering: "          # peers on a trajectory, not query/document

# Semantic inverse pairs. Forward moves the claim; backward is meant to undo it.
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
IDENTITY = ("Restate this sentence with identical meaning, changing at most two words.",
            "Restate this sentence with identical meaning, changing at most two words.")

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

# outputs that describe the edit instead of performing it
META = ("the sentence", "the statement", "the phrase", "the claim change", "the idea is",
        "the concept", "the context", "the situation", "the scenario", "the observation",
        "we observe", "we analyze", "we examine", "we deduce", "we notice", "we begin",
        "first, we", "the analysis", "this is like", "imagine", "consider a", "think of")


# ------------------------------------------------------------------ endpoints
def _post(path, body, timeout=180):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed(text):
    r = _post("/v1/embeddings", {"model": EMBED_MODEL, "input": PREFIX + text.strip()}, 90)
    v = np.asarray(r["data"][0]["embedding"], dtype=float)
    return v / max(norm(v), 1e-12)


def rewrite(model, instruction, text, temperature=0.0):
    # Do NOT pass context_length: LMS may try to reload the instance and OOM
    # when a second LLM is already resident (Snapdragon 16GB).
    body = {"model": model, "system_prompt": SYSTEM,
            "input": f"{instruction}\n\nSENTENCE: {text}",
            "temperature": temperature, "max_output_tokens": 120,
            "store": False}
    out = _post("/api/v1/chat", body).get("output")
    if isinstance(out, str):
        return out.strip()
    parts = []
    for blk in out or []:
        c = blk.get("content") if isinstance(blk, dict) else None
        if isinstance(c, list):
            parts += [x.get("text", "") for x in c if isinstance(x, dict)]
        elif isinstance(c, str):
            parts.append(c)
    return " ".join(parts).strip()


def is_meta(s):
    t = s.strip().strip('"').lower()
    return t.startswith(META)


def step(model, instruction, text, tries=3):
    """Rewrite, rejecting meta-narration. Returns None if it never complies."""
    for i in range(tries):
        out = rewrite(model, instruction, text, temperature=0.0 if i == 0 else 0.3)
        if out and not is_meta(out) and len(out) > 12:
            return out
    return None


# ------------------------------------------------------------------ geometry
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


# ------------------------------------------------------------------ one loop
def round_trip(model, seed, depth, identity=False):
    """seed -> k forwards -> k inverses. Closed by construction; nothing sees the seed."""
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

    E = np.array([embed(t) for t in texts])
    ret = float(E[0] @ E[-1])                       # did meaning survive the trip
    consec = [float(E[i] @ E[i + 1]) for i in range(len(E) - 1)]
    # collapse = the trajectory froze somewhere OTHER than the seed. Measured
    # against the seed, not against neighbours, because a genuinely closed loop
    # also has high consecutive similarity.
    seed_sim = [float(E[0] @ e) for e in E[1:]]
    tail_frozen = min(consec[len(consec)//2:]) > 0.95 if len(consec) > 2 else False
    collapsed = bool(tail_frozen and max(seed_sim[len(seed_sim)//2:]) < 0.75)
    th = bishop(proj3(E))
    N = len(texts)
    return dict(depth=depth, N=N, texts=texts, ret=ret, theta=th,
                lam=(2 - 2 * math.cos(th / N)) if np.isfinite(th) else float("nan"),
                max_consec=max(consec), collapsed=bool(collapsed))


def logistic(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (y > 1e-4) & (y < 1 - 1e-4)
    if m.sum() < 3:
        return float("nan"), float("nan")
    z = np.log(y[m] / (1 - y[m]))
    A = np.vstack([x[m], np.ones(m.sum())]).T
    k, b = np.linalg.lstsq(A, z, rcond=None)[0]
    return (-b / k if k else float("nan")), float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="liquid/lfm2.5-1.2b")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3, 4],
                    help="depth 1 = one forward+inverse (return cos only; |θ|≡0 on 3 nodes). "
                         "Meaningful Bishop holonomy needs depth>=2 (N>=5).")
    ap.add_argument("--floor-reps", type=int, default=8)
    ap.add_argument("--floor-only", action="store_true")
    ap.add_argument("--gate-min", type=float, default=0.90,
                    help="min depth-1 (or first depth) closure rate or abort θ*")
    ap.add_argument("--out", default="holonomy_v2.json")
    a = ap.parse_args()

    # ---- 1. noise floor: identity round trips ------------------------------
    print("STEP 1  noise floor (identity round trips, depth 1)\n")
    print("  note: identity uses same instruction both ways; |θ|~0 on near-collinear paths.")
    floor = []
    for s in SEEDS[:a.floor_reps]:
        r = round_trip(a.model, s, 1, identity=True)
        if r:
            floor.append(r)
            print(f"  ret {r['ret']:.3f}  |theta| {abs(r['theta']):.3f}  {s[:52]}")
    if not floor:
        print("  identity arm produced nothing. Model is not following instructions.")
        return 1
    f_ret = np.array([r["ret"] for r in floor])
    f_th = np.array([abs(r["theta"]) for r in floor])
    thr = float(np.percentile(f_ret, 10))
    print(f"\n  return cos : median {np.median(f_ret):.3f}  p10 {thr:.3f}  min {f_ret.min():.3f}")
    print(f"  |theta|    : median {np.median(f_th):.3f}  max {f_th.max():.3f}   <- NOISE FLOOR")
    print(f"  closure threshold set at p10 of the identity arm: {thr:.3f}")
    if np.median(f_ret) < 0.75:
        print("\n  ABORT-WORTHY: identity round trips do not return. The model cannot")
        print("  restate a sentence without changing it. Nothing deeper is measurable.")
    if a.floor_only:
        return 0

    # ---- 2. the ladder -----------------------------------------------------
    print("\nSTEP 2  drift ladder\n")
    rows, xs, ys = [], [], []
    for d in a.depths:
        got = []
        for s in SEEDS:
            r = round_trip(a.model, s, d)
            if r:
                got.append(r); rows.append(r)
        if not got:
            continue
        ok = [r for r in got if not r["collapsed"]]
        rate = float(np.mean([r["ret"] >= thr for r in ok])) if ok else 0.0
        xs.append(d); ys.append(rate)
        geom = " [ret-only; |θ|≡0 on triangle]" if d == 1 else " [Bishop N=2d+1]"
        print(f"  depth {d}  n={len(got):2d}  collapsed {len(got)-len(ok):2d}  "
              f"closure {rate:5.2f}  median ret {np.median([r['ret'] for r in got]):.3f}  "
              f"median |theta| {np.median([abs(r['theta']) for r in got]):.3f}{geom}")

    # ---- 3. gate then fit --------------------------------------------------
    # Gate uses first depth (usually 1) as RETURN check only — not as holonomy.
    # Logistic fit uses depths>=2 when present (triangles cannot carry Bishop θ).
    print("\n" + "=" * 68)
    print("  note: depth-1 is a 3-node triangle → Bishop |θ| is structurally ~0;")
    print("        gate = return cosine only. Holonomy ladder starts at depth 2.")
    gate_rate = ys[0] if ys else 0.0
    if not ys or gate_rate < a.gate_min:
        print(f"  GATE FAILED. depth-{xs[0] if xs else 1} closure = {gate_rate:.2f}, "
              f"need >= {a.gate_min:.2f}.")
        print("  Shortest forward+inverse round trip does not return past floor thr.")
        print("  No theta* reported. (v1 would have produced uninterpretable loops.)")
        mid, k, sigma = float("nan"), float("nan"), float("nan")
        gated = False
    else:
        # fit on depths with geometric holonomy (N>=5 ⇒ depth>=2)
        xs_h = [x for x in xs if x >= 2]
        ys_h = [y for x, y in zip(xs, ys) if x >= 2]
        if len(xs_h) >= 3:
            mid, k = logistic(xs_h, ys_h)
        else:
            mid, k = logistic(xs, ys)
        sigma = (-1 / k) / 0.47 if k and np.isfinite(k) else float("nan")
        print(f"  GATE PASSED. depth-1 closure {gate_rate:.2f}")
        print(f"  drift tolerance  depth* = {mid:.2f}")
        print(f"  logistic slope          = {k:.3f}")
        print(f"  implied noise    sigma  = {sigma:.4f}   (floor |θ| med "
              f"{np.median(f_th):.3f} rad)")
        print("  Run other models and compare depth* to trace the capacity route.")
        gated = True
    print("=" * 68)

    Path(a.out).write_text(json.dumps(
        {"model": a.model, "threshold": thr, "gate_min": a.gate_min, "gated": gated,
         "floor": {"median_ret": float(np.median(f_ret)), "median_theta": float(np.median(f_th)),
                   "p10_ret": thr},
         "depths": xs, "closure": ys,
         "depth1_note": "return cos only; Bishop null on 3-node triangle",
         "finding": (
             "round_trip_fail" if not gated else "ladder_ok"
         ),
         "rows": [{k: v for k, v in r.items() if k != "texts"} for r in rows]},
        indent=2))
    print(f"\nwrote {a.out}")
    return 0 if gated else 2


if __name__ == "__main__":
    raise SystemExit(main())
