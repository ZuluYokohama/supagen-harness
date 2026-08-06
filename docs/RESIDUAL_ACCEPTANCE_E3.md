# Residual acceptance — E3 NPU Job2 label parity

**Status:** **ACCEPTED RESIDUAL** for dual-metric **GO_MEASURE** advertise  
**Not accepted for:** production OPEN marketing, product HTP Job2 agreement path  
**Date:** 2026-08-06  
**Run / evidence:** `docs/evidence/npu/npu_nli_qdq*.json` · host Snapdragon X Plus Hexagon

---

## What was measured

| Backend | Domain pairs | Result |
|---------|--------------|--------|
| ORT CPU DeBERTa (product authority) | held-out | **PASS** (authoritative) |
| QDQ UINT8 @ HTP (early) | 3 | 0/3 — logits collapse ~neutral 0.51 |
| QDQ act=UINT16 w=UINT8 @ HTP (early) | 3 | 0/3 — **label-inverted** |
| QDQ UINT16/UINT8 + **expanded balanced CALIB (18)** | 4 (incl. neutral) | **1/4 (rate 0.25)** — HTP live ~35ms; still red vs ≥0.9 |

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
2. `prefer="htp"` is refused unless `prime/state/npu/nli_htp_parity_cert.json` is green (`measure_fabric.nli_htp_parity_pass`).  
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
| Next engineering | **expanded balanced CALIB_PAIRS** (entail/contra/neutral) → re-QDQ UINT16 → held-out parity; if still red: distill stance head / QAI Hub context binary. **Not** blocked on Adreno llama-server (GPU owns SCOUT, not law). |

**Signed residual (author measure):** accepted for GO_MEASURE instrument advertise on 2026-08-06.
