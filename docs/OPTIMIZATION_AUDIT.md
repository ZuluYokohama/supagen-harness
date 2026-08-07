# Optimization audit — recommendations vs live `main`

**Repo:** ZuluYokohama/supagen-harness  
**Law:** GO_MEASURE · Job2 gate = CPU ORT/CE · NPU = measure fabric only · cosine never OPEN  

This document **measures** the optimization brief against the codebase and records what was implemented.

---

## Verdict table

| Recommendation | Brief claim | Live MEASURE | Action |
|----------------|-------------|--------------|--------|
| ORT session singleton | New cache needed | **Already present** (`_SESSION` + EP-class reuse in `accel_nli_ort.load_session`) | Keep; batch path reuses same singleton |
| Pre-alloc tensor buffers | Static buffers | Partial — batch tokenize allocates `B×L` once per chunk | Accept; true static pools are micro-gain |
| `glue_agreement_batch` | Missing | **Implemented** | `entailment_glue.glue_agreement_batch` + `predict_batch` |
| Batched mutual NLI | Sequential only | **Implemented** | `mutual_entailment` ORT path batches ab+ba |
| Sparse `eigsh` λ₁ in `compute_graph.py` | Dense spectral gap | **Mis-targeted** — `compute_graph.py` is RPL-C **process DAG**, not a sheaf Laplacian eigensolver | **STOP** here; see `123abc/rplc_sheaf.py` / sparse backend if needed |
| Vietoris–Rips ε-prune | Dense persistence in harness | **Not in product dual-metric path** | Defer to topology plugins; do not invent in Job1/2 |
| Jina Matryoshka `target_dim` | Full 1024 only | **Implemented (opt-in)** | `kb_index --target-dim N` prefix truncate |
| SQ8 vector index | float32 only | **Implemented (opt-in)** | `kb_index --quantize sq8` + dequant on query |
| AST cache in disposition | Re-parses AST | **False claim** — disposition is **string/substring** checks, not AST | mtime+size **text** cache added instead |
| Scrub mime/mmap | Scans all files | **Already** limited to `.json/.md/.txt` | No mmap needed for slim evidence |

---

## Implemented (this change set)

1. **`accel_nli_ort.predict_batch`** — B×L ORT runs, `force_cpu=True` product law, chunked `max_batch`  
2. **`predict`** — thin wrapper over batch of 1 (singleton session)  
3. **`mutual_entailment`** — one ORT batch for both directions when prefer auto/ort  
4. **`glue_agreement_batch`** — multi-pair one-way ORT batch; optional mutual  
5. **`verify_cr_disposition`** — mtime+size content cache for `t(rel)`  
6. **`kb_index`** — optional `target_dim`, `quantize=sq8` (aboutness only)  
7. Disposition gates: `ort_predict_batch`, `glue_agreement_batch`, `mutual_ort_batched`

---

## Explicit non-goals (residue)

| Item | Why not force |
|------|----------------|
| HTP batch as Job2 gate | E3 residual; NPU measure fabric only |
| Sparse sheaf λ₁ in `compute_graph` | Wrong module; process topology ≠ Laplacian |
| Claiming 3–5× without bench | Throughput is MEASURE; re-bench on kit after enable |
| Production OPEN from faster NLI | Still NO-GO |

---

## How to exercise

```powershell
# Product CPU ORT batch smoke (requires exported DeBERTa ORT)
python -c "from accel_nli_ort import predict_batch; print(predict_batch([('a','a'),('yes','no')], force_cpu=True))"

# Disposition still fail-closed
python prime/scripts/verify_cr_disposition.py

# Optional edge KB
python prime/scripts/kb_index.py build --target-dim 512 --quantize sq8
```

## Measured bench (author kit, post-land)

Script: `prime/scripts/bench_nli_batch.py`  
Artifact: `docs/evidence/nli_batch_bench.json`

| Arm | Result (n=24 one-way pairs, force_cpu, CPU EP) |
|-----|-----------------------------------------------|
| Sequential | ~165–185 ms/pair (varies by rep) |
| Batch | ~165–182 ms/pair |
| **Speedup seq/batch** | **~0.91–1.12×** across reps (best ~1.12×) |
| Label parity seq vs batch | **True** |
| Mutual sample | 8 pairs `batched=True`; ~400 ms/mutual pair |

**Honest takeaway:** the marketing **3–5×** band is **not** observed on this DeBERTa ORT CPU path at n=24. Batching is correct and **label-parity safe**; wall-clock gains are modest/noisy (tokenizer + max_length padding dominate). Do **not** certificate a 3–5× claim.

```powershell
python prime/scripts/bench_nli_batch.py --n 24 --reps 2
python -m unittest discover -s prime/tests -v
```

---

## NPU gate isolation (product law)

| Knob | Product behavior |
|------|------------------|
| `PRIME_ACCEL=auto` (default) | `_providers()` → **CPU only**; QNN never listed |
| `PRIME_ACCEL=qnn\|npu` | QNN may load on measure/session path — still not OPEN authority |
| `predict` / `predict_batch` default | `force_cpu=True` pins CPU even if env was non-auto |
| `prefer=htp` in glue | Requires green `nli_htp_parity_pass()`; else `htp_refused` and CPU path |
| `route_job2()` | `htp_is_gate_authority=false` always; gate order ort_cpu→CE→lfm |

Unit coverage: `prime/tests/test_nli_batch_law.py` (CI step on offline workflow).

## Sheaf ALU boundary

| Module | Role |
|--------|------|
| `prime/scripts/compute_graph.py` | RPL-C **process DAG** only — no `eigsh` / VR |
| `prime/scripts/accel_nli_ort.py` / `entailment_glue.py` | Job2 agreement — no sheaf Laplacian |
| `123abc/rplc_sheaf.py` (+ sparse backend) | Sheaf λ₁ / ALU home — **not** dual-metric Job1/2 |
