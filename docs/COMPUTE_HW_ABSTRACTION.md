# Compute ∶ HW Abstraction — Snapdragon X Plus (this kit)

**Date:** 2026-08-06  
**Host:** ASUS Zenbook A14 · Snapdragon X Plus X1P42100 · Hexagon NPU · Adreno X1-45 · Oryon 8c  
**Authoritative QNN run:** **`npu-htp-2026-08-06`** · packages `onnxruntime==1.24.4` + `onnxruntime-qnn==2.4.0` · evidence `docs/evidence/npu/`  
**Status:** HTP path **measured live** (`npu_stress`, `htp_profile.csv`). Product NLI on HTP **not** parity-green.

This is not a product pitch. It is a substrate map for *novel use* of the NPU inside the dual-metric / OPEN|STOP stack we already run.

---

## 0. What we already proved on *this* machine

| Fact | Evidence (run `npu-htp-2026-08-06`) |
|------|----------|
| Hexagon present | PnP `ComputeAccelerator` OK |
| QNN plugin EP works | `onnxruntime 1.24.4` + `onnxruntime-qnn 2.4.0` → 3 QNN devices after `register()` |
| HTP executes graphs | Profile: **HVX/HMX power-on**, **accelerator execute**, **mm0–mm7 NODE cycles** |
| Sustained load | ~**10.7k runs/s** for 45s on 8×512 QDQ MatMul chain |
| Task Manager NPU | **No `\NPU*` counters** on this Windows image — TM is not the oracle |
| Job1 jina | llama-server GGUF → **CPU** (no QNN path in llama.cpp) |
| Job2 DeBERTa QDQ on HTP | **Session loads on QNN** (~32 ms/pair); **label parity FAIL** (always ~neutral 0.51 — quant quality, not routing) |

**Implication:** We are past “does NPU exist?” and into “what *shape* of work should own it.”

---

## 1. The three brains (and what each is for)

```text
┌─────────────────────────────────────────────────────────────┐
│                    SHARED LPDDR  (one memory pool)            │
└───────────┬─────────────────┬─────────────────┬─────────────┘
            │                 │                 │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
     │   ORYON     │   │   ADRENO    │   │  HEXAGON    │
     │   CPU 8c    │   │   GPU X1-45 │   │  NPU HTP    │
     │  control,   │   │  floatish,  │   │  INT8/16    │
     │  branches,  │   │  shaders,   │   │  tile GEMM, │
     │  agents,    │   │  DML/GL,    │   │  fixed QDQ  │
     │  sheaf ALU  │   │  vision     │   │  low joules │
     └─────────────┘   └─────────────┘   └─────────────┘
```

| Unit | Loves | Hates | Our stack today |
|------|-------|-------|-----------------|
| **Oryon CPU** | Control flow, JSON, agents, sparse sheaf, LMS chat | Sustained dense GEMM at battery | SCOUT/PRESERVE, loop logic, rplc-sheaf |
| **Adreno GPU** | FP16/FP32 bulk, vision, some DML nets | Power vs NPU for INT8, app packaging (DML vs QNN ORT conflict) | Display; DML optional / separate |
| **Hexagon HTP** | **Fixed-shape QDQ**, MatMul/Conv tiles, **always-on** instruments, batch embed/NLI | Dynamic seq, heavy control flow, unquantized transformers, “one big LLM decode” | Stress path only (until Job2 QDQ) |

**Rule of three (smart routing — saturate the silicon, not the Oryon alone):**  

| Processor | Owns | Workloads | Product authority today |
|-----------|------|-----------|-------------------------|
| **Hexagon NPU** | **The Law (measure)** | QDQ DeBERTa NLI, QDQ rerank, HTP stress MatMul | **Job2 only after E3 parity green**; else refuse HTP |
| **Adreno GPU** | **The Scout (generate)** | LFM/Ministral decode via Vulkan/OpenCL llama.cpp when available | SCOUT drafting only — never OPEN authority |
| **Oryon CPU** | **The Orchestrator** | Prime MCP, Python, sheaf/compute graph, jina `:8765`, dual_enter, cert_face | Always: control plane + **CPU ORT/CE agreement authority until E3** |

Task Manager **NPU 0%** while Oryon is hot usually means: instruments still on CPU *or* host IO/quant prep only (HTP may still execute — TM is not the NPU oracle on this Windows image). Evidence: `docs/evidence/npu/`, `npu_stress` profile cycles.

**Intent alignment:** absolute precision → finish **E3 label parity** so NPU can *own* agreement measure; efficiency of SCOUT tokens is secondary (GPU llama-server hunt).  

That maps cleanly onto our law: **aboutness + agreement are measures; OPEN is a decision.**

---

## 2. HW constraints that define “possible”

### 2.1 HTP / QNN hard constraints (from ORT QNN + QAIRT)

| Constraint | Consequence for us |
|------------|-------------------|
| **Quantized only** (UINT8/UINT16 act/weight) | No FP32 DeBERTa as-is; must QDQ + calibrate |
| **Fixed shapes** | Pad/truncate to fixed `MAX_LEN` (we use 128 for NLI trial) |
| **Op subset** | MatMul, Conv, LN, Softmax, etc. OK; dynamic control / exotic DeBERTa ops may **partition** graph → CPU islands |
| **Partition tax** | Every CPU↔HTP hop kills the “free lunch”; aim for **full-graph** or **few islands** |
| **IO quant often on host** | High *python* CPU% can coexist with real HTP (we measured ~1 core host + HTP execute) |
| **Peak ~45 TOPS INT8** (X series class) | Great for **many small inferences/sec**, not magic for 70B decode |

### 2.2 What NPU is *not* (on this generation)

- Not a drop-in for **llama.cpp GGUF** (Job1 jina stays side-server CPU until a QNN embed path exists)  
- Not free **FP16 training**  
- Not Task Manager–visible telemetry on this OS build  
- Not a replacement for **external OPEN|STOP certifier** (law stays human/domain)

---

## 3. Abstraction layers we should build (compute∶hw)

Name the plane **Measure Fabric** — NPU-first instruments, CPU arbiter.

```text
                 ┌──────────────────┐
  dual_enter ──► │  ROUTE (CPU)     │  purpose_gate / fiber_mode
                 └────────┬─────────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ Job1 ABOUT │  │ Job1.5 RR  │  │ Job2 AGREE │
   │ embed      │  │ rerank     │  │ NLI/mutual │
   │ CPU today  │  │ CPU today  │  │ CPU authority│
   │ (NPU later │  │ (NPU later │  │ HTP only if │
   │  measure)  │  │  measure)  │  │ E3 parity✓) │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                 ┌──────────────────┐
                 │ cert_face (CPU)  │  contradiction→STOP; cos never OPEN
                 │ external domain  │  production OPEN ≠ Job2 alone
                 └──────────────────┘
```

### 3.1 Device capability table (runtime)

| Capability | Detect | Fallback |
|------------|--------|----------|
| `hexagon_htp` | `npu_qnn.register()` n_qnn>0 + **htp_profile cycles** | ORT CPU |
| `qdq_nli_product` | `nli_htp_parity_pass()` (held-out ORT parity ≥0.9) | **refuse HTP**; ORT/CE |
| `qdq_model(path)` | file exists + QNN session | liveness probe only |
| `adreno_dml` | `PRIME_ACCEL=dml` + Dml EP (separate ORT build) | skip GPU |
| `llama_embed` | :8765 probe | nomic LMS (degraded) |

### 3.2 Job → device affinity (desired)

| Job | Ideal HW | Why (novel for *us*) |
|-----|----------|----------------------|
| **Continuous aboutness** (KB reembed, tick, dual_enter pair cos) | **NPU** | Always-on, low joules; doesn’t thrash frankenstein RAM |
| **Batch mutual NLI** on claim lattice | **NPU** | Many fixed pairs/sec; agreement plane free while CPU agents work |
| **Adversarial twin screen** (benign vs attacks) | **NPU** | Cheap contradiction pre-filter before DeBERTa-large or LFM |
| **Rerank shortlist (top-32)** | **NPU** or GPU | Cross-encoder-like score heads quantize well |
| **Sheaf λ₁ / sparse Laplacian** | **CPU** (or GPU BLAS later) | Irregular sparsity ≠ HTP tile GEMM |
| **SCOUT / PRESERVE chat** | **CPU LMS** | Agents need decode + tools; NPU weak at full LLM decode here |
| **Identity holonomy judge** | **NPU** (same NLI QDQ) | DeBERTa judge independent of MUT fiber — perfect HTP resident |
| **Certify / OPEN** | **CPU only** | Deterministic external gate; never offload “authority” to opaque NPU |

---

## 4. Novel uses unique to *this* project (not generic RAG)

### 4.1 **Measure plane residency** (biggest idea)

Keep **Job1 + Job2 instruments always hot on HTP** while LMS loads **one** chat fiber (SCOUT or PRESERVE).

Today: instruments fight chat for Oryon + RAM.  
With NPU: instruments “live in Hexagon”; frankenstein/LFM own DRAM for identity/scout.

**Novel:** *holonomy subject and agreement judge no longer co-resident on the same ALU.*  
That is a **physical dual metric** — not just a software comment.

### 4.2 **Agreement lattice thrashing**

Mutual entailment is O(n²) on claim sets.  
NPU: fire **fixed-shape (128,128) pair batches** across a claim graph every tick.

Use cases:

- deep_loop work items × evidence stalks  
- multiplane claim bundle pre-score before external certify  
- continuous “force-OPEN” adversarial scan (attacks: twins)

**Novel:** treat NLI as a **background sheaf restriction probe** — residue = pairs that stay NEED_INFO/STOP; never force OPEN.

### 4.3 **Aboutness as cheap filtration; NLI as expensive only on survivors**

Pipeline:

```text
NPU embed (or hybrid) → top-k
NPU light NLI head on (query, hit)   → drop contradiction / neutral
CPU DeBERTa-large / mutual only if borderline
CPU cert_face
```

**Novel:** NPU is the **Vietoris–Rips style cheap radius**; CPU is the **sheaf section** when H⁰ might close.

### 4.4 **Polarity-blind cosine compensated by NPU stance head**

Bakeoff proved cosine polarity blindness (adv cos still high).  
Train or distill a **tiny stance/contradiction head** (not full DeBERTa) that:

- input: concat/pooled pair features or dual towers  
- output: 3-way NLI logits  
- fully QDQ on HTP  

**Novel:** keep jina for *aboutness geometry*; NPU head owns *polarity channel* that cosine structurally lacks (symmetry / no zero).

### 4.5 **Identity floor judge on NPU**

Holonomy already uses **independent DeBERTa** as judge vs frankenstein MUT.  
Move judge to HTP → identity ladder can run **without stealing RAM from frankenstein**.

**Novel:** identity measurement becomes a **background instrument**, not a thrash event.

### 4.6 **Power-aware OPEN|STOP ethics** (hypothesis — unmeasured)

**Unmeasured hypothesis:** NPU joules may be ≪ CPU for continuous audit; always-on law enforcement *might* be thermally cheaper.  
We have **throughput + HTP-cycle** evidence only — **no controlled CPU-vs-HTP power (W / mJ) bench** on this kit yet. Do not treat “thermally free” as an established project result.

**Still true by law (CPU authority):** never cos→OPEN; adversarial twin STOP; residue never forced — independent of NPU joules.

### 4.7 **What not to force onto NPU**

| Temptation | Why not |
|------------|---------|
| Full frankenstein decode | NPU not the genAI decode path here; LMS/CPU wins |
| Sparse sheaf Laplacian | Irregular → CPU/sparse backend |
| Dynamic agent tool loops | Branchy → CPU |
| Production OPEN | Authority must stay auditable CPU/external |

---

## 5. Research map: industry patterns → our stack

| Pattern (2025–26) | Source class | Map to PRIME/supagen |
|-------------------|--------------|----------------------|
| Smart routing NPU vs CPU | Qualcomm X series dev blogs | Job-level `purpose_gate` → device |
| On-device RAG embed on Hexagon | arXiv mobile NPU RAG (~9× embed throughput) | KB reembed + dual_enter aboutness |
| QDQ-only HTP | ORT QNN docs | All measure models must export QDQ |
| Precompiled QNN context binaries (AI Hub) | Surface / QAI Hub case studies | Ship `_ctx.onnx` for Job2 cold-start |
| Full-graph delegation | LiteRT QNN / op coverage | Simplify NLI head if DeBERTa partitions |
| Concurrent small models | 45 TOPS INT8 | Many NLI pairs + embed ticks while chat runs |

---

## 6. Experiment ladder (precision → product)

| # | Experiment | Success | Status |
|---|------------|---------|--------|
| E0 | Plugin register + tiny QDQ HTP | QNN in providers + profile HVX | **DONE** |
| E1 | Sustained stress visible / profiled | 45s burst + `htp_profile.csv` | **DONE** |
| E2 | Fix DeBERTa QDQ calib + export | quantize → 198 MB QDQ; HTP session OK | **DONE** |
| E3 | Job2 NLI on HTP with label parity | **PARTIAL** — UINT8: logits collapse ~0.51; **UINT16 act** (2026-08-06): uncollapsed but labels inverted (0/3 hits, ~32ms). Still **not** product. Next: calib / distill | **NOW** |
| E4 | Always-on NPU judge while frankenstein loaded | free_gb + identity p stable | pending |
| E5 | Batch mutual NLI lattice (n=16 claims) throughput | pairs/s vs CPU | pending |
| E6 | Distill stance head (tiny) for adv twins | adv STOP rate ≥ DeBERTa, ms << | pending |
| E7 | Optional: QAI Hub context binary for Job2 | cold start &lt; 1s | optional |

---

## 7. Software architecture sketch (`measure_fabric`)

```text
prime/scripts/
  npu_qnn.py           # register, session_qdq  (exists)
  npu_stress.py        # sustained HTP proof     (exists)
  npu_nli_qdq.py       # DeBERTa→QDQ→HTP         (blocked at calib)
  measure_fabric.py    # NEW: route Job1/1.5/2 by capability
  accel_nli_ort.py     # CPU ORT path            (exists)
```

**Routing policy (pseudocode) — E3 parity gate required for HTP ownership:**

```python
def nli(premise, hyp):
    # HTP only when explicit label-parity + calibration PASS (E3 green).
    # qdq_nli_ready() / session-loads is NOT enough — DeBERTa QDQ currently
    # collapses logits (neutral≈0.51) even when HTP runs.
    if has_htp() and nli_htp_parity_pass():   # E3 gate; currently FALSE
        r = nli_htp(premise, hyp)              # measure fabric only — not OPEN
        if r.ok: return r
    if ort_cpu_ready():
        return nli_ort_cpu(premise, hyp)       # agreement measure default
    return nli_cross_encoder(...)              # torch CE
    # never: cosine; never Job2 alone → production OPEN
    # cert_face + external domain audit own OPEN|STOP
```

---

## 8. “Novel” thesis (one paragraph)

Most people will use the Hexagon for **Windows Studio Effects and chat toys**.  
Our dual-metric law needs something else: a **physically separate measure ALU** that can hammer aboutness and agreement **without** unloading frankenstein or believing cosine. The NPU is that ALU — if we keep QDQ fixed-shape instruments on HTP and leave agents + OPEN authority on CPU. That is compute∶hw abstraction as **sheaf of devices**: restriction maps between planes (aboutness / agreement / chat) with **device-level heterophily**, and OPEN only when the CPU cert face sees no obstruction.

---

## 9. Status of next actions (2026-08-06)

| # | Action | Status |
|---|--------|--------|
| 1 | Fixed-shape export + QDQ recipes (UINT8 / UINT16) | **DONE** (parity still red) |
| 2 | Held-out pairs vs ORT CPU `label_parity_rate` | **DONE** in `npu_nli_qdq.py` |
| 3 | `measure_fabric` + `prefer=htp` refuse until cert green | **DONE** (`ort→ce→lfm` product) |
| 4 | No product HTP NLI until E3 green | **ENFORCED** (`PRIME_ACCEL=auto` → CPU only) |
| 5 | Task Manager not NPU oracle | **DONE** (htp_profile proof) |
| 6 | Distill / better calib / QAI Hub for E3 green | **OPEN engineering** (not GO_MEASURE blocker) |

---

## 10. References (working)

- ORT QNN EP: fixed shapes, QDQ-only HTP, op list (MatMul UINT8/16, LayerNorm, …)  
- Qualcomm X series “smart routing” NPU vs CPU  
- On-device RAG on Hexagon: embed-dominated speedups (~9× class results reported)  
- This kit artifacts: `prime/state/npu/htp_profile.csv`, `npu_stress_report.json`, `npu_qnn_smoke.json`

---

### Live finding (E3 partial)

DeBERTa-v3-base **does compile and execute on Hexagon HTP** (graph finalize, VTCM, QNN in providers, ~32 ms).  
But naive UINT8 QDQ **destroys NLI decision quality** (constant neutral).  

That itself is research gold:

| Layer | Status |
|-------|--------|
| **Routing / residency** | Solved (plugin + QDQ + fixed shape) |
| **Numeric fidelity for agreement** | Not solved with default UINT8 |
| **Implication** | Prefer **tiny distilled stance heads** or **UINT16 activations** for Job2; keep CPU DeBERTa as authority until parity |

**Bottom line:** We are in a real place to use Hexagon as the **continuous measure fabric** — and the novel constraint is clear: **NPU owns throughput of fixed QDQ instruments; CPU owns calibrated agreement until a parity-passing head exists.** Device-split dual metric stands; the next precision mile is a **HTP-native contradiction head**, not “force full DeBERTa UINT8 and pray.”
