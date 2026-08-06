# Holonomy identity floor (v3) — capacity sheet

**Gate:** mutual entailment both ways (DeBERTa). **Threshold:** p ≥ 0.80.  
**Cosine is diagnostic only** — never gates agreement or identity.

| Model | Quant | identity p | cos≥0.75 would say | median cos | Gate | Modes (8 seeds) |
|-------|-------|------------|--------------------|------------|------|-----------------|
| **frankenstein-2.0-i1** | Q4_K_S 7.2B | **~0.88** (chain flat through d4) | 1.00 | ~0.89 | **PASS** | mostly closed |
| **liquid/lfm2.5-1.2b** | Q8 1.2B | **0.29** | **0.00** | ~0.66 | **FAIL** | dropped×4, inverted×1, closed×2 |
| **gemma-4-12b** Q4_K_M | 12B | **0.38** | — | — | **FAIL** | channel/meta pollution |

## Reading (Claude's accounting)

- **p** = fraction of identity round-trips that **mutually entail** (same claim both ways).  
- Cosine can look “ok-ish” (LFM median ~0.66) while **p collapses** — same failure class as granite rewords scoring 0.598 cos while DeBERTa said entailed both ways… inverted here: LFM often **drops** content (one-way entailment).  
- LFM was **not** a strong holonomy subject. Use it as **fast scout fiber**, not multi-hop identity carrier.  
- Frankenstein remains the multi-hop / identity-chain subject on this kit.  
- Gemma-12B capacity did **not** buy higher p.

## Reproduce

```bash
python prime/scripts/run_lfm_identity_floor.py
python prime/scripts/run_identity_chain.py --model frankenstein-2.0-i1
# prior gemma: prime/scripts/run_gemma12b_floor.py (needs llama-server side load)
```

Artifacts: `prime/state/holonomy_v3_lfm12b_identity_floor.json`,  
`holonomy_v3_frankenstein_identity_chain.json`, `holonomy_v3_gemma12b_floor.json`.

---

## Dual metric + Tier-B instruments (2026-08-06)

| Instrument | Model | Domain measure | Gate |
|------------|-------|----------------|------|
| Job1 aboutness | **jina-v5-small Q4** (:8765, **pooling=last**, dim **1024**) | bakeoff floor **0.085**, ceil **0.735**, range **0.650**; live A/C floor **0.077** / ceil **0.875** / range **0.80** | max local retrieve |
| Job1.5 rerank | jina-reranker-v3 | prefer benign **9/9** | retrieval only |
| Job2 agreement | ORT DeBERTa-v3-base MNLI | adv contra **1.0**, block OPEN **1.0** | owns agreement |
| Cosine polarity | any encoder | neg **0.69** / adv **0.64** / worst twin **0.78** (was 0.85 nano) | **never OPEN** |
| KB manifold | 53 chunks reembedded | family **jina**, dim **1024** | retrieve aligned |

**Fiber modes:** SCOUT = LFM/Ministral (unload frankenstein). PRESERVE = frankenstein alone.  
**V&V matrix:** `docs/VV_RUN_RESULTS.md` → **GO_MEASURE** (**18/18**).  
**Push suite:** **GO_MEASURE** (5 PASS + Hexagon QNN WARN only).  
**supagen contract live:** **21/21 PASS** (floor 0.077, ceil 0.875, range 0.80, dim 1024, small model).  
**Field harness offline smoke:** **OK**.

### Accel path (Job2)

| Backend | Status | Note |
|---------|--------|------|
| torch CE DeBERTa | PASS | authority fallback |
| **ORT CPU** DeBERTa ONNX (~704MB) | **PASS** label parity; **~3× faster** batch (3.1s vs 9.6s) | `accel_nli_ort.py`; `PRIME_NLI_ORT=1` |
| ORT DML (Adreno) | FAIL runtime Reshape | skip auto; `PRIME_ACCEL=dml` forces |
| Hexagon QNN | not installed | residual silicon |

### Push suite (`vv_push_domains.py`)

| Cell | Status | Measure |
|------|--------|---------|
| P1 jina v5-small | **PASS** | Q4_K_M GGUF + last pool; floor 0.038 / ceil 0.88 / range 0.84 / dim 1024 |
| P2 PRESERVE smoke | **PASS** | frankenstein alone @16384; restore SCOUT LFM |
| P3 truth_plane enter | **PASS** | face STOP/NEED_INFO; no force-OPEN |
| P4 negative force-OPEN | **PASS** | cos 0.92 + attacks twin → STOP via ORT NLI |
| P5 ORT hot | **PASS** | ~187ms contradiction |
| P6 Hexagon QNN | **WARN** | HW present; QNN EP not in ORT providers on ARM64 |

### Hexagon NPU (2026-08-06 — LIVE + profiled)

| Fact | Detail |
|------|--------|
| HW | Snapdragon X Plus Hexagon NPU (`ComputeAccelerator` OK) |
| Stack | `onnxruntime==1.24.4` + plugin `onnxruntime-qnn==2.4.0` |
| Register | `npu_qnn.register()` → **3 QNN devices** |
| Sustained load | `python prime/scripts/npu_stress.py --seconds 45` → **~10.7k runs/s**, 45s continuous |
| **Hard proof** | `prime/state/npu/htp_profile.csv` — **HVX/HMX power-on**, **QNN accelerator execute**, **mm0–mm7 NODE cycles** on HTP |
| Task Manager | **This Windows build has NO `\NPU*` perf counters** (`typeperf` empty). Performance panel **cannot graph** third-party QNN HTP. GPU counters exist (Adreno); NPU tile missing/unwired for app QNN. |
| Why it looked “pathetic” | Micro-smoke finished in ms (0% average); Job1/2 still CPU; TM has nothing to plot for NPU |
| Job2 | DeBERTa still ORT **CPU** until full QDQ NLI lands on HTP |

**Do not use Task Manager as NPU proof on this kit. Use `htp_profile.csv`.**
