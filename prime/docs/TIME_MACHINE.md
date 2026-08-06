# Semantic holonomy — the time machine (sister chizzle)

Drop: `files_chizzle.zip` → `prime/chizzle/` + installed under `scripts/`.

## What it is

Not cosine “are these about the same.” A **closed loop of transforms** in text space, embeds with **earned** nomic (`search_document:`), projects to a 3-frame, **Bishop parallel transport** — same operator as the peptide ring:

```
lambda_min = 2 - 2*cos(theta / N)     closed iff lambda_min < tau
```

- **theta** = residual rotation after one circuit  
- **S-curve** = step function of closure seen through noise σ (sigmoid.py; R² ≈ 1 on theory)  
- **theta\*** = midpoint = how much drift this model tolerates before loops open  
- **Route** = theta\* vs capacity (1.2B → frontier) = where local chart stops covering  

## Relation to our C→R→C′

| | `round_trip_holonomy.py` | `semantic_holonomy.py` |
|--|--------------------------|-------------------------|
| Loop | RESTRICT write / reconstruct | Multi-rung paraphrase → close |
| Metric | 1 − cos(C, C′) | Bishop θ on trajectory |
| Purpose | TOPICALITY | TOPICALITY + geometry |
| Verdict | drift structure | closure rate, theta\*, σ |

Same idea: **holonomy residual**. Chizzle is the full geometric instrument; RESTRICT trip is the cheap operational face.

## CLI

```bash
python prime/scripts/sigmoid_semantic.py              # offline theorem check
python prime/scripts/semantic_holonomy.py --model liquid/lfm2.5-1.2b --out prime/state/holonomy_lfm.json
python prime/scripts/semantic_holonomy.py --compare route_*.json
```

## v1 vs v2 (sister fix)

| | v1 | v2 |
|--|----|----|
| Close | `CLOSE(last)` — open arc, can restate the *story* | k forwards then k **inverses** — closed by construction, no seed leak |
| Meta | none | reject “we observe / imagine a chef…” |
| Collapse | self-similar mid-rung looked closed | frozen **and** far from seed |
| Floor | assumed τ | identity arm → p10 threshold |
| Gate | none | depth-1 closure ≥ 0.90 or **no θ reported** |

v1 full grid on LFM: non-monotone, θ\* nonsense. That was the check that couldn’t fail.

## Live v2 (LFM 1.2B) — 2026-08-05

**Identity floor (same instruction both ways):**
```
median ret 0.850  p10 thr 0.801  min 0.770
ABORT (median < 0.75): not printed — can near-restate under identity pair
```

**Drift ladder (forward + inverse pairs; thr = 0.801):**

| depth | n | collapsed | closure | med ret | med \|θ\| |
|-------|---|-----------|---------|---------|-----------|
| 1 | 6 | 0 | **0.50** | 0.82 | 0.00 |
| 2 | 6 | 0 | 0.50 | 0.79 | 2.06 |
| 3 | 7 | 0 | 0.29 | 0.80 | 1.06 |
| 4 | 6 | 0 | 0.17 | 0.77 | 2.33 |

```
GATE FAILED. depth-1 closure = 0.50, need >= 0.90.
No θ* reported. (v1 would have spent minutes on uninterpretable loops.)
```

**Finding:** identity restatement works (~0.85); **one real transform + inverse does not close** at the floor-calibrated threshold. That caps every deeper holonomy measurement on this capacity — same spirit as sister prediction, but the fail is on the **non-identity** depth-1 gate, not the identity arm.

Results: `state/holonomy_v2_lfm.json`

### Capacity probe: Granite-4-h-tiny (same v2 harness)

| arm | LFM 1.2B | Granite-4-h-tiny |
|-----|----------|------------------|
| identity median ret | 0.850 | **0.877** |
| identity p10 thr | 0.801 | 0.822 |
| depth-1 closure (fwd+inv) | **0.50** | **0.25** |
| gate (≥0.90) | FAIL | FAIL |

Granite identity is slightly better; **real round-trip is worse**, not better.  
Capacity jump did **not** clear the gate → suspect transform/inverse instructions (or model-agnostic loss), not “1.2B is just too small.”  
v2 notes: depth-1 Bishop |θ|≡0 on triangle (design); gate is return-cos only. Holonomy starts depth≥2 **if** gate passes.

`state/holonomy_v2_granite_d1.json`

## Smoke v1 (legacy)

Seed: design-law claim. Rungs 0/2/5 — see holonomy_lfm12b.json (uninterpretable S).

## Null discipline

Still required: pasta/random seeds should not share the same closure S-curve as structured claims. Extend SEEDS with pasta controls before calling theta\* a route map.

## Figure

`docs/s_curve_semantic.png` — (a) step through noise (b) 1/slope ∝ σ (c) θ\* ∝ N  
