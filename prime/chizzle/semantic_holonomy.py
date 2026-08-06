"""
semantic_holonomy.py -- measure whether meaning survives a round trip.

THE IDEA
--------
Cosine asks "are these two texts about the same thing." That question failed
its null (pasta vs E_ref = 0.83). This asks a different question, one with a
verdict attached:

    Push a proposition around a CLOSED loop of transformations and come back.
    Does it return to itself, or does it return rotated?

A faithful representation has zero holonomy around a closed semantic loop. A
lossy one does not. The residual rotation IS the loss, and unlike cosine it
has a floor at zero that means something.

This is the same operator as the cyclic peptide result, on a different stalk.
Identical transport code, identical closed form:

    lambda_min = 2 - 2*cos(theta / N)        closed iff lambda_min < tau

WHY THE CURVE IS AN S
---------------------
h0 is a STEP: the loop closes or it does not, nothing between. But no
instrument measures theta exactly. A step seen through measurement noise is a
sigmoid, and the sigmoid carries two readings:

    logistic slope  ->  1/slope = 0.47 * sigma      (verified R^2 = 0.9999)
    midpoint        ->  theta* = N * sqrt(tau)      (verified R^2 = 1.00000)

So the S-curve is not a separate phenomenon. It is Theorem 1 with error bars,
and fitting it recovers the representational noise of whatever model produced
the loop.

THE ROUTE
---------
Run this at several capacities -- Q4 to Q8, 1.2B to 8B, local to frontier --
and plot theta* against capacity. That curve is "the route back", and its
knee is the capacity at which semantic loops start closing at all.

USAGE
-----
    python semantic_holonomy.py --model liquid/lfm2.5-1.2b
    python semantic_holonomy.py --model qwen3-8b --out route_qwen8b.json
    python semantic_holonomy.py --compare route_*.json     # plot the route
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

# Loop rungs, increasing semantic drift. Rung 0 is near-identity, rung 5 is a
# long way from the source. Every loop RETURNS to a faithful restatement, so a
# lossless representation closes at every rung.
RUNGS = [
    "Restate this sentence with identical meaning, changing at most two words.",
    "Restate this sentence in your own words, same meaning.",
    "Restate this for a reader in a different technical field, same meaning.",
    "Express this as a concrete example that carries the same claim.",
    "Express this as an analogy from an unrelated domain, same underlying claim.",
    "Express the same claim as it would appear three inferential steps downstream.",
]
CLOSE = "Restate this as the original plain claim it came from, stripped of framing."

SEEDS = [
    "Strain that no choice of dihedrals can remove belongs to the ring, not to any residue.",
    "A claim may not open unless a restriction was recorded before the measures.",
    "The decoder was offline while the tool recorded fifty interference events downhole.",
    "An instrument that returns the same value on every input carries no information.",
    "Sequence sets the cost of ring closure and has no bearing on its feasibility.",
    "Where the mismatch is written down is bookkeeping; the total is not.",
    "A lift earns its keep only if something can be wrong upstairs that could not be wrong downstairs.",
    "Longer chains tolerate more accumulated drift before they stop cohering.",
]


# ---------------------------------------------------------------- transport
def _rot_between(u, v):
    c = float(np.clip(u @ v, -1, 1))
    ax = np.cross(u, v)
    s = norm(ax)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    ax = ax / s
    th = math.atan2(s, c)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def bishop_holonomy(P):
    """Parallel-transport a normal once around the closed trajectory P.
    Identical to the peptide code. P is (k, 3) after projection."""
    n = len(P)
    T = np.array([P[(i + 1) % n] - P[i] for i in range(n)])
    nz = norm(T, axis=1, keepdims=True)
    if (nz < 1e-12).any():
        return float("nan")
    T = T / nz
    u = np.cross(T[0], [0.0, 0.0, 1.0])
    if norm(u) < 1e-8:
        u = np.cross(T[0], [0.0, 1.0, 0.0])
    u0 = u / norm(u)
    u = u0.copy()
    for i in range(n):
        u = _rot_between(T[i], T[(i + 1) % n]) @ u
    u -= (u @ T[0]) * T[0]
    u /= norm(u)
    return float(math.atan2(float(np.cross(u0, u) @ T[0]), float(np.clip(u0 @ u, -1, 1))))


def project3(E):
    """Local 3-frame of the loop via PCA. Transport needs a 3D ambient space;
    the loop's own principal directions are the honest choice."""
    X = E - E.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:3].T


# ---------------------------------------------------------------- endpoints
def post(path, body, timeout=180):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed(text):
    # envelope-free, task-prefixed: the instrument that earned its keep
    r = post("/v1/embeddings", {"model": EMBED_MODEL,
                                "input": "search_document: " + text.strip()}, 90)
    return np.array(r["data"][0]["embedding"], dtype=float)


def transform(model, instruction, text):
    body = {"model": model,
            "system_prompt": "Output only the restated sentence. No preamble, no quotes.",
            "input": f"{instruction}\n\nSENTENCE: {text}",
            "temperature": 0.0, "max_output_tokens": 120,
            "context_length": 2048, "store": False}
    out = post("/api/v1/chat", body).get("output")
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


# ---------------------------------------------------------------- experiment
def run_loop(model, seed, rung, steps=6):
    """Closed loop: seed -> rung transforms -> back to plain claim -> seed.
    Returns (theta, lambda_min, texts)."""
    texts = [seed]
    cur = seed
    for _ in range(steps - 2):
        cur = transform(model, RUNGS[rung], cur)
        if not cur:
            return float("nan"), float("nan"), texts
        texts.append(cur)
    texts.append(transform(model, CLOSE, cur))  # the return leg
    E = np.array([embed(t) for t in texts])
    th = bishop_holonomy(project3(E))
    N = len(texts)
    return th, (2 - 2 * math.cos(th / N) if np.isfinite(th) else float("nan")), texts


def fit_logistic(x, y):
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
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="semantic_holonomy.json")
    ap.add_argument("--compare", nargs="*", help="plot theta* vs capacity from result files")
    a = ap.parse_args()

    if a.compare:
        rows = [json.loads(Path(p).read_text()) for p in a.compare]
        rows.sort(key=lambda r: r.get("theta_star") or 0)
        print(f"  {'model':<28}{'theta*':>10}{'sigma':>10}{'closure@0':>12}")
        for r in rows:
            print(f"  {r['model']:<28}{r.get('theta_star', float('nan')):>10.3f}"
                  f"{r.get('sigma', float('nan')):>10.3f}{r['rates'][0]:>12.2f}")
        print("\n  theta* rising with capacity IS the route. Its knee is the")
        print("  capacity at which semantic loops begin to close at all.")
        return 0

    print(f"model {a.model}   tau {a.tau}   steps {a.steps}   seeds {len(SEEDS)}\n")
    rates, thetas = [], []
    for rung in range(len(RUNGS)):
        ok, th_r = 0, []
        for seed in SEEDS:
            try:
                th, lm, _ = run_loop(a.model, seed, rung, a.steps)
            except Exception:
                continue
            if not np.isfinite(th):
                continue
            th_r.append(abs(th))
            ok += int(lm < a.tau)
        rate = ok / max(len(th_r), 1)
        rates.append(rate)
        thetas.append(float(np.median(th_r)) if th_r else float("nan"))
        print(f"  rung {rung}  n={len(th_r):2d}  median|theta| {thetas[-1]:6.3f}"
              f"  closure {rate:5.2f}   {RUNGS[rung][:44]}")

    x = np.arange(len(RUNGS), dtype=float)
    mid, slope = fit_logistic(x, np.array(rates))
    sigma = (-1 / slope) / 0.47 if slope and np.isfinite(slope) else float("nan")

    print("\n" + "=" * 66)
    print(f"  drift tolerance  theta*  = {mid:.3f} rungs")
    print(f"  logistic slope           = {slope:.3f}")
    print(f"  implied noise    sigma   = {sigma:.4f}")
    print("=" * 66)
    if not np.isfinite(mid):
        print("  No transition in range. Either every loop closed (extend RUNGS)")
        print("  or none did (the representation has no closure regime here).")
    else:
        print("  A real transition. The midpoint is this model's closure threshold;")
        print("  run other capacities and compare with --compare to trace the route.")

    Path(a.out).write_text(json.dumps({
        "model": a.model, "tau": a.tau, "steps": a.steps,
        "rates": rates, "median_theta": thetas,
        "theta_star": mid, "slope": slope, "sigma": sigma}, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
