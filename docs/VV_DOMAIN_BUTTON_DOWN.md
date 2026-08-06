# V&V domain button-down (PR #1)

**PR:** https://github.com/ZuluYokohama/supagen-harness/pull/1  
**Branch:** `vv/dual-metric-npu-measure-fabric`  
**CodeRabbit:** ASSERTIVE cycles; majors + criticals addressed on HEAD; re-review queued.  
**CI:** `verify-offline` green  
**Author matrix (this kit):** `vv_full_matrix` → **GO_MEASURE · 18 pass / 0 fail / 0 critical fail** (regenerated `docs/VV_RUN_RESULTS.md`)

This sheet buttons down every domain of relevance. Status = author measures + static review + CodeRabbit parse.

---

## Scoreboard

| Domain | Status | Evidence / residual |
|--------|--------|---------------------|
| **D0 Architecture** | **PASS** | Hybrid LMS + off-LMS instruments; `owns_agreement` / **not** `owns_open` |
| **D1 Job1 aboutness** | **PASS** | jina floor/range; cos never OPEN; fail-closed if range unevaluated |
| **D2 Job1.5 rerank** | **PASS** | aboutness_hybrid; error envelopes |
| **D3 Job2 NLI** | **PASS** | prefer=auto ORT→CE; mutual; contra STOP; CE per-model locks |
| **D4 Fiber modes** | **PASS** | scout/preserve; `ensure --mode`; preserve unload fail-closed |
| **D5 cert_face / dual_enter** | **PASS** | high cos no NLI ≠ OPEN; force-OPEN → NEED_INFO/STOP |
| **D6 Identity holonomy** | **PASS** | LFM FAIL / frankenstein PASS floors measured |
| **D7 Package / contract** | **PASS** | offline + live author 21/21; CI golden schema-only |
| **D8 NPU / Hexagon** | **PASS w/ residual** | HTP path live (`hexagon_path_live`); **Job2 QDQ label parity residual** |
| **D9 Adversarial** | **PASS** | lexical correlation + force-OPEN blocked |
| **D10 Truth loop** | **PASS** | MEASURE only; stable=None if &lt;2 rounds |
| **D11 Field harness** | **PASS** | offline smoke; DRAFT STOP; multiplane OPEN when covered |
| **D12 KB family/dim** | **PASS** | jina dim 1024 |
| **D13 Compute∶HW** | **PASS (doc)** | CPU authority; HTP only after E3; power = hypothesis |
| **D14 Ops/secrets** | **PASS** | evidence sanitized; no user-profile paths in tracked NPU JSON |
| **D15 Docs honesty** | **PASS (law)** / **PENDING (independent buddy)** | GO_MEASURE ≠ production OPEN; buddy protocol unsigned externally |

**Aggregate for advertise of dual-metric instrument law:** **GO_MEASURE**  
**Aggregate for production OPEN marketing:** **NO-GO** until independent buddy L8 + NPU parity residual accepted or closed.

---

## Hard residuals (do not paper over)

1. **NPU Job2 label parity** — UINT8 collapse / UINT16 invert. CPU ORT/CE = agreement authority.  
2. **Independent buddy L8** — protocol `docs/BUDDY_L8_SIGNOFF.md`; author self-evidence only under `docs/evidence/buddy_l8_*.json`.  
3. **CI golden** — schema-only by design (full sandbox seal is field-local).

---

## Reproduce

```powershell
cd C:\PRIMEdEV-1
git checkout vv/dual-metric-npu-measure-fabric
python -m pip install -e ./supagen
$env:GOLDEN_SCHEMA_ONLY=1
python -m supagen verify
python prime/scripts/vv_full_matrix.py
```

PR: https://github.com/ZuluYokohama/supagen-harness/pull/1
