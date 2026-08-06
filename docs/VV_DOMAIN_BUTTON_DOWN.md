# V&V domain button-down (PR #1)

**PR:** https://github.com/ZuluYokohama/supagen-harness/pull/1  
**Branch:** `vv/dual-metric-npu-measure-fabric`  
**Seal:** **`docs/GO_MEASURE_SEAL.md`** — dual-metric instrument law **GO_MEASURE**  
**CI:** `verify-offline` green  
**Matrix:** 18/18 PASS · live contract 21/21  

---

## Scoreboard

| Domain | Status | Evidence / residual |
|--------|--------|---------------------|
| **D0 Architecture** | **SEALED** | Hybrid LMS + off-LMS; owns_agreement / not owns_open |
| **D1 Job1 aboutness** | **SEALED** | jina floor≈0.04 ceil≈0.88 range≈0.84 |
| **D2 Job1.5 rerank** | **SEALED** | aboutness_hybrid; envelopes |
| **D3 Job2 NLI** | **SEALED** | ORT contra STOP p=0.9999 |
| **D4 Fiber modes** | **SEALED** | ensure --mode; preserve fail-closed |
| **D5 cert_face / dual_enter** | **SEALED** | force-OPEN → NEED_INFO/STOP |
| **D6 Identity holonomy** | **SEALED** | measured floors |
| **D7 Package / contract** | **SEALED** | offline+live; CI schema-only |
| **D8 NPU / Hexagon** | **SEALED + residual** | path live; E3 accepted; HTP refuse measured |
| **D9 Adversarial** | **SEALED** | lexical + NLI block |
| **D10 Truth loop** | **SEALED** | MEASURE only |
| **D11 Field harness** | **SEALED** | offline multiplane |
| **D12 KB family/dim** | **SEALED** | jina 1024 |
| **D13 Compute∶HW** | **SEALED** | measure_fabric ort→ce→lfm |
| **D14 Ops/secrets** | **SEALED** | sanitized evidence |
| **D15 Docs honesty** | **SEALED (law)** | GO_MEASURE ≠ production OPEN; buddy protocol runnable |

**Aggregate instrument law:** **GO_MEASURE (sealed)**  
**Production OPEN marketing:** **NO-GO**

---

## Hard residuals (accepted / optional)

1. **E3 NPU Job2 label parity** — accepted residual; product HTP refused (`RESIDUAL_ACCEPTANCE_E3.md`).  
2. **Independent human dual-sign** — optional; clean-clone offline L8-01…04 **PASS** (`buddy_l8_clean_clone_offline.json`).  
3. **CI golden** — schema-only by design.

---

## Protocol runners

```powershell
python prime/scripts/buddy_l8_offline.py   # L8-01…04
python prime/scripts/vv_full_matrix.py     # D0–D17 matrix
python -m supagen contract                 # live 21/21
```

See also: `CODERABBIT_RESOLUTION_MAP.md`, `GO_MEASURE_SEAL.md`.
