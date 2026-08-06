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
| D17 WARN as n_fail | **Fixed** | n_warn separate |
| Evidence user paths | **Fixed** | sanitized + logical IDs |
| nomic fallback in jina bakeoff | **Fixed** | family_mismatch reject |
| ORT probs key case | **Fixed** | normalized |
| smoke --bench orphan | **Fixed** | argparse + bench |
| smoke duplicate register | **Fixed** | delegates npu_qnn.register |
| tier_b miss contra gate | **Fixed** | nli_catches_contradiction required |
| empty adv-lexical crash | **Fixed** | fail artifact |
| NPU Job2 label parity | **Accepted residual** | `RESIDUAL_ACCEPTANCE_E3.md` + red parity cert |
| Independent buddy L8 | **Pending external** | author self-evidence only |

Stale inline comments may still map to new line numbers after re-review; this table is the authoritative disposition until CR re-submits on current HEAD.
