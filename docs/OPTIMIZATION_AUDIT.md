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

**Priority after this land:** re-measure ORT sequential vs batch on the author kit (truth_loop claim counts), then only then consider CE batch or SQ8 integer kernels.
