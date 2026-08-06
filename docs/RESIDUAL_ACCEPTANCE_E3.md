# Residual acceptance — E3 NPU Job2 label parity

**Status:** **ACCEPTED RESIDUAL** for dual-metric **GO_MEASURE** advertise  
**Architectural commitment:** **NPU = measure fabric only; Job2 gate = CPU ORT/CE** (design law, not temporary fallback)  
**Not accepted for:** production OPEN marketing, product HTP Job2 agreement path  
**Date:** 2026-08-06  
**Run / evidence:** `docs/evidence/npu/nli_eval_qdq_vs_cpu.json` · held-out reports · host Snapdragon X Plus Hexagon

---

## What was measured

| Backend | Domain pairs | Result |
|---------|--------------|--------|
| ORT CPU DeBERTa (product authority) | held-out | **PASS** (authoritative) |
| QDQ UINT8 @ HTP (early) | 3 | 0/3 — logits collapse ~neutral 0.51 |
| QDQ act=UINT16 w=UINT8 @ HTP (early) | 3 | 0/3 — **label-inverted** |
| QDQ UINT16/UINT8 + **expanded balanced CALIB (18)** | 4 (incl. neutral) | **1/4 (rate 0.25)** — HTP live ~35ms; still red vs ≥0.9 |
| QDQ UINT16/UINT8 + **contra-heavy CALIB (32)** | 4 | **1/4 (0.25)** — same pattern; calib mix insufficient |
| **Same QDQ graph on CPU EP** (not HTP) | 4 | **1/4 (0.25)** — **isolates residual to quant, not HTP** |

**Isolation (2026-08-06, contra-heavy calib):**  
Loading `model.qdq.aui16_wui8.onnx` with `CPUExecutionProvider` alone yields the **same** held-out miss pattern as QNN HTP.  
Therefore: Hexagon is not “corrupting” a good quant graph — static QDQ of DeBERTa-v3-base currently **destroys label geometry** before HTP.  
Product law remains correct: CPU **FP32/ORT** (or CE) owns agreement until a green parity cert.

**Logit diagnostic (UINT16, 2026-08-06):**

| Expect | HTP argmax | HTP logits (approx) |
|--------|------------|---------------------|
| contradiction | entailment | `[-1.30, 2.23, -1.30]` |
| contradiction | entailment | `[-0.03, 1.66, -2.17]` |
| entailment | contradiction | `[1.13, 0.31, -1.68]` |

ORT CPU on the same pairs: contradiction 0.9999 / 0.9999, entailment 0.9985.

**Conclusion:** quant/HTP path damages label geometry. Session-ready QNN ≠ calibrated NLI.

---

## Product law (code-enforced)

1. `glue_agreement(prefer="auto")` → **ORT → CE → LFM** only.  
2. `prefer="htp"|"hexagon"|"npu"`:  
   - **Red / missing** `nli_htp_parity_pass` → annotate `htp_refused`, normalize to `auto`, run ORT→CE→LFM.  
   - **Green** cert today still **normalizes to `auto`** and runs ORT→CE→LFM — production HTP Job2 session is **not wired** yet; green cert is a future gate, not a live HTP product path.  
3. Parity cert never sets `job2_owns_open=true`.  
4. Production OPEN remains external domain audit + cert_face.

See: `prime/scripts/measure_fabric.py`, `entailment_glue.glue_agreement`.

---

## Acceptance decision

| Question | Answer |
|----------|--------|
| Block GO_MEASURE for dual-metric instrument law? | **No** — CPU Job2 owns agreement measure |
| Block production OPEN marketing? | **Yes** (already NO-GO for other reasons too) |
| Claim product HTP NLI? | **No** until E3 parity cert green |
| Next engineering | **Do not spend on QAI Hub/distill for product gate.** Full `nli_eval_v1` (65 pairs) proves QDQ destroys the discriminative faculty: **con_high 0.952 → 0.048**. Not single-label collapse (pred ~90% entailment, not 100%). Held-out **0.25 is measured (1/4)**, not a default. Optional: HTP stays measure fabric for latency experiments only. |

**Signed residual (author measure):** accepted for GO_MEASURE instrument advertise on 2026-08-06.
