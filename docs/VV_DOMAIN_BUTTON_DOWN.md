# V&V domain button-down (PR #1)

**PR:** https://github.com/ZuluYokohama/supagen-harness/pull/1  
**Branch:** `vv/dual-metric-npu-measure-fabric`  
**CodeRabbit:** ASSERTIVE full review cycles; majors addressed in follow-up commits.  
**CI:** `verify-offline` **SUCCESS** (golden **schema-only** on CI + full offline contract)

This sheet buttons down every domain of relevance. Status = author measures + static review + CodeRabbit parse.

---

## Scoreboard

| Domain | Status | Evidence / residual |
|--------|--------|---------------------|
| **D0 Architecture** | **PASS** | Hybrid LMS chat + jina/DeBERTa/rerank/NPU off-LMS; `default_embed_base` jina isolation |
| **D1 Job1 aboutness** | **PASS** | v5-small path, pooling=last, bakeoff floor≈0.08 range≈0.65; cos not OPEN; D1 fail-closed when range unevaluated |
| **D2 Job1.5 rerank** | **PASS** | `rerank_service` + retrieve hook; aboutness_hybrid only; error envelopes |
| **D3 Job2 NLI** | **PASS** | prefer=auto ORT→CE; mutual; layered_enter fixed to auto; contradiction STOP; CE per-model locks |
| **D4 Fiber modes** | **PASS** | scout unloads frankenstein; preserve alone; `supagen ensure --mode scout\|preserve` wired |
| **D5 cert_face / dual_enter** | **PASS** | high cos no NLI ≠ OPEN; truth_plane optional |
| **D6 Identity holonomy** | **PASS** | LFM p=0.29 FAIL; frank p=0.875 PASS; gemma 0.38 FAIL (docs+votes) |
| **D7 Package / contract** | **PASS (author live)** / **CI schema-only** | live 21/21 on author kit; CI = offline + golden schema-only (not full sandbox seal) |
| **D8 NPU / Hexagon** | **PASS w/ residual** | HTP live + profile HVX (run `npu-htp-2026-08-06`); DeBERTa QDQ **label parity FAIL** — not product default |
| **D9 Adversarial** | **PASS** | force-OPEN + cos 0.92 → STOP; lexical r≈0.85–0.93 |
| **D10 Truth loop** | **PASS** | MEASURE only; residue/STOP elevate face; stable=None if &lt;2 rounds |
| **D11 Field harness** | **PASS** | offline smoke OK; DRAFT STOP; multiplane OPEN when covered |
| **D12 KB family/dim** | **PASS** | reembed jina dim 1024 |
| **D13 Compute∶HW** | **PASS (doc)** | device-split; power ethics = **hypothesis**; E3 parity gate for HTP NLI |
| **D14 Ops/secrets** | **PASS** | gguf/onnx gitignored; NPU evidence under `docs/evidence/npu/` |
| **D15 Docs honesty** | **PASS w/ residual** | GO_MEASURE ≠ production OPEN; buddy L8 + NPU parity still open |

**Aggregate for advertise of dual-metric instrument law:** **GO_MEASURE (provisional)** — full offline golden seal not on CI; buddy L8 unsigned.  
**Aggregate for production OPEN marketing:** **NO-GO** until buddy L8 + NPU parity residual accepted or closed + CR re-review clean.

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
| Major | HTP NLI without parity | **Documented** — not product default; E3 gate |
| Major | V&V plan L1-04 red vs D4 green | **Fixed** — plan + CLI `--mode` |
| Major | dim=768 stale | **Fixed** — L2-01 dim=1024 |
| Major | golden soft-PASS without schema flag | **Fixed** — fail-closed incomplete |
| Major | CE global lock | **Fixed** — per-model locks |
| Major | rerank exceptions uncaught | **Fixed** — ok:False envelope |
| Major | preserve env accepts scout key | **Fixed** — PRESERVE_KEYS filter |
| Major | truth_loop stable=True under 2 rounds | **Fixed** — stable=None |
| Major | D1 range None still critical PASS | **Fixed** — critical=False + applied_rule |
| Major | QDQ partial write | **Fixed** — tmp + replace |
| Major | npu smoke `--bench` orphan | **Fixed** — argparse + bench path |
| Major | Job1 provenance mixed | **Fixed** — run IDs in HOLONOMY |
| Major | QNN historical vs live | **Fixed** — run `npu-htp-2026-08-06` |
| Major | power ethics as fact | **Fixed** — labeled hypothesis |

Follow-up also: atomic truth_plane JSON, accel 60s cache, jina live meta, npu register via devices.

## Hard residuals (do not paper over)

1. **NPU Job2 label parity** — QDQ DeBERTa on HTP collapses logits (neutral≈0.51). CPU ORT/CE remains authority. Evidence: `docs/evidence/npu/npu_nli_qdq_report.json`.  
2. **Buddy lab L8-08** — external clean install not signed in this PR.  
3. **CI golden** — schema-only by design; full sandbox seal is author/field only.

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
python -m supagen verify   # offline (CI sets GOLDEN_SCHEMA_ONLY=1)
python prime/scripts/vv_full_matrix.py
# NPU proof (optional hardware):
python prime/scripts/npu_stress.py --seconds 20
```

PR: https://github.com/ZuluYokohama/supagen-harness/pull/1
