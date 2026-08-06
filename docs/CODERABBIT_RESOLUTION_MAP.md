# CodeRabbit resolution map (PR #1)

**Branch:** `vv/dual-metric-npu-measure-fabric`  
**Purpose:** Map prior ASSERTIVE findings → HEAD disposition so re-review is not blocked by stale inline comments.

| Finding (theme) | Disposition on HEAD | Evidence |
|-----------------|---------------------|----------|
| Job2 owns OPEN / HTP prefer without parity | **Fixed** | `owns_open_gate=false`; diagram E3; `measure_fabric` + glue refuse HTP |
| Power ethics as fact | **Fixed** | labeled unmeasured hypothesis |
| D15 closed before buddy | **Documented** | PASS(law)/PENDING(buddy); `BUDDY_L8_SIGNOFF.md` |
| Providers list as HTP proof | **Fixed** | `qnn_ep_registered` + profile proof |
| Partial QDQ reuse | **Fixed** | atomic write + `_qdq_looks_complete` |
| Rerank uncaught predict | **Fixed** | ok:False envelope |
| Preserve env scout key | **Fixed** | PRESERVE_KEYS filter |
| Preserve unload soft-ok | **Fixed** | substrate ok=False |
| truth_loop stable under 2 rounds | **Fixed** | stable=None |
| CE global lock | **Fixed** | per-model locks |
| golden soft PASS without schema flag | **Fixed** | fail-closed incomplete |
| D1 range None critical PASS | **Fixed** | applied_rule + critical=False |
| dim=768 | **Fixed** | dim=1024 |
| D17 WARN as n_fail | **Fixed** | n_warn separate; `n_pass+n_warn+n_fail==n_cells` asserted |
| Job2 auto QNN EP | **Fixed** | `predict(force_cpu=True)` default; QNN refused on product path |
| Parity cert incomplete green | **Fixed** | fail-closed fields: held_out, cpu_fallback=false, no probe_only |
| Buddy independent_buddy free-form | **Fixed** | requires BUDDY_L8_ATTESTATION=signed:… |
| LFM in PRESERVE | **Fixed** | LFM fallback scout-only |
| Evidence user paths | **Fixed** | sanitized + logical IDs |
| nomic fallback in jina bakeoff | **Fixed** | family_mismatch reject |
| ORT probs key case | **Fixed** | normalized |
| smoke --bench orphan | **Fixed** | argparse + bench |
| smoke duplicate register | **Fixed** | delegates npu_qnn.register |
| tier_b miss contra gate | **Fixed** | nli_catches_contradiction required |
| empty adv-lexical crash | **Fixed** | fail artifact |
| NPU Job2 label parity | **Accepted residual** | `RESIDUAL_ACCEPTANCE_E3.md` + red parity cert |
| Independent buddy L8 | **Package portability proven** | clean-clone offline L8-01…04 PASS; human dual-sign optional |
| Smoke double quantize | **Fixed** | single `quantize()` pass |
| Profile full-file load | **Fixed** | stream scan `n_lines_scanned` |
| PowerShell `;` vs `,` | **Fixed** | calculated properties use commas |

**Disposition verifier:** `python prime/scripts/verify_cr_disposition.py` → `CR_DISPOSITION_VERIFY_PASS`

Stale inline comments may still map to new line numbers after re-review; this table + verifier are authoritative until CR re-submits on current HEAD.
