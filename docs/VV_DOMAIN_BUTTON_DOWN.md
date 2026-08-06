# V&V domain button-down (PR #1)

**PR:** https://github.com/ZuluYokohama/supagen-harness/pull/1  
**Branch:** `vv/dual-metric-npu-measure-fabric`  
**CodeRabbit:** ASSERTIVE full review **completed** (`CHANGES_REQUESTED`, ~60+ actionable). Critical/major quick-wins landed in follow-up commits.  
**CI:** `verify-offline` **SUCCESS** (golden schema-only + follow-up)

This sheet buttons down every domain of relevance. Status = author measures + static review + CodeRabbit parse.

---

## Scoreboard

| Domain | Status | Evidence / residual |
|--------|--------|---------------------|
| **D0 Architecture** | **PASS** | Hybrid LMS chat + jina/DeBERTa/rerank/NPU off-LMS; `default_embed_base` jina isolation |
| **D1 Job1 aboutness** | **PASS** | v5-small path, pooling=last, bakeoff floor≈0.08 range≈0.65; cos not OPEN |
| **D2 Job1.5 rerank** | **PASS** | `rerank_service` + retrieve hook; aboutness_hybrid only |
| **D3 Job2 NLI** | **PASS** | prefer=auto ORT→CE; mutual; layered_enter fixed to auto; contradiction STOP |
| **D4 Fiber modes** | **PASS** | scout unloads frankenstein; preserve alone; measured |
| **D5 cert_face / dual_enter** | **PASS** | high cos no NLI ≠ OPEN; truth_plane optional |
| **D6 Identity holonomy** | **PASS** | LFM p=0.29 FAIL; frank p=0.875 PASS; gemma 0.38 FAIL (docs+artifacts) |
| **D7 Package / contract** | **PASS** | offline+live contract 21/21 author kit; CI offline SUCCESS |
| **D8 NPU / Hexagon** | **PASS w/ residual** | HTP live + profile HVX; DeBERTa QDQ runs ~32ms **label parity FAIL** — not product default |
| **D9 Adversarial** | **PASS** | force-OPEN + cos 0.92 → STOP; lexical r≈0.85–0.93 |
| **D10 Truth loop** | **PASS** | MEASURE only; residue/STOP elevate face |
| **D11 Field harness** | **PASS** | offline smoke OK; DRAFT STOP; multiplane OPEN when covered |
| **D12 KB family/dim** | **PASS** | reembed jina dim 1024 |
| **D13 Compute∶HW** | **PASS (doc)** | `COMPUTE_HW_ABSTRACTION.md` device-split dual metric |
| **D14 Ops/secrets** | **PASS** | gguf/onnx gitignored; no secrets in PR |
| **D15 Docs honesty** | **PASS** | GO_MEASURE ≠ production OPEN; limitations explicit |

**Aggregate for advertise of dual-metric instrument law:** **GO_MEASURE**  
**Aggregate for production OPEN marketing:** **NO-GO** until buddy L8 + any CodeRabbit blockers fixed.

---

## CodeRabbit response (parse)

| Severity | Theme | Action |
|----------|--------|--------|
| Critical | `bakeoff_adv_lexical` crash on None cos | **Fixed** — skip None cos rows |
| Major | bakeoff `ok` always True | **Fixed** — derive from n_ok + metrics |
| Major | `attacks:` stopword never matched | **Fixed** — tokenize on punctuation |
| Major | duplicate PRIME_RERANK gate | **Fixed** — single gate in rerank_service |
| Major | fiber_mode env unreachable | **Fixed** — default `None` → env |
| Major | preserve unload results discarded | **Fixed** — record extra_acts |
| Major | truth_loop ignores explicit False | **Fixed** — explicit opt-out wins |
| Major | operator_summary face stale | **Fixed** — refresh from cert_face |
| Major | tier_b exit ≠ tier_b_ready | **Fixed** — exit on tier_b_ready |
| Major | trust_remote_code loose | **Fixed** — TRUSTED_REMOTE allowlist |
| Major | NPU evidence gitignored | **Fixed** — `docs/evidence/npu/` slim archive |
| Major | stale state artifacts as PASS | **Fixed** — D1 freshness helper (72h) |
| Major | HTP NLI without parity | **Documented** — not product default |
| Major | V&V plan L1-04 red vs D4 green | **Fixed** — plan text updated |

Remaining nits (markdownlint blank lines, atomic write, accel cache) tracked as non-blockers.

## Hard residuals (do not paper over)

1. **NPU Job2 label parity** — QDQ DeBERTa on HTP collapses logits (neutral≈0.51). CPU ORT/CE remains authority. Evidence: `docs/evidence/npu/npu_nli_qdq_report.json`.  
2. **Buddy lab L8-08** — external clean install not signed in this PR.  
3. **supagen ensure --mode** CLI flag still thinner than env/API (modes green via residency).

---

## CodeRabbit parse protocol (when complete)

1. Open PR #1 summary comment + review comments.  
2. Map each finding → D0–D15 row.  
3. Severity: Blocker / Major / Minor / Nit per brief.  
4. Close domain only if: no open Blocker/Major **or** residual is explicitly documented (like NPU parity).  
5. Update this file + `VV_RUN_RESULTS.md` with bot run ID.

---

## Reproduce

```powershell
cd C:\PRIMEdEV-1
git checkout vv/dual-metric-npu-measure-fabric
python -m pip install -e ./supagen
python -m supagen verify   # offline
python prime/scripts/vv_full_matrix.py
# NPU proof (optional hardware):
python prime/scripts/npu_stress.py --seconds 20
```

PR: https://github.com/ZuluYokohama/supagen-harness/pull/1
