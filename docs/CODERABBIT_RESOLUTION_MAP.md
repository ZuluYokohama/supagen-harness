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
| D17 WARN as n_fail | **Fixed** (artifact regen) | report has n_warn; count_ok; n_pass+n_warn+n_fail==n_cells |
| Job2 auto QNN EP | **Fixed** | `predict(force_cpu=True)` default; QNN refused on product path |
| Parity cert incomplete green | **Fixed** | fail-closed fields: held_out, cpu_fallback=false, no probe_only |
| Buddy independent_buddy free-form | **Fixed** | requires BUDDY_L8_ATTESTATION=signed:… |
| LFM in PRESERVE | **Fixed** | LFM fallback scout-only |
| Evidence user paths | **Fixed** | scrub_evidence_paths + disposition host-path scan fail-closed |
| nli_eval digests / shared QDQ labels | **Fixed** | score_nli_qdq_eval eval_sha256+qdq_sha256+load_qdq_labels |
| DOMAIN cert HEAD bind | **Fixed** | lineage seal: commit ancestor of HEAD + tree match |
| README QAI Hub Next | **Fixed** | measure-only law; no product remediation path |
| Jina GGUF basename match | **Fixed** | samefile / size+mtime identity |
| Smoke QDQ direct write | **Fixed** | temp + os.replace atomic |
| D17 header vs cells | **Fixed** | header_counts_match required; published MD match |
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
| HELD_OUT overlaps CALIB | **Fixed** (`43c8ce3`) | `validate_held_out_disjoint()` + CI runtime check |
| ORT parity without force_cpu | **Fixed** (`43c8ce3`/`4110aee`) | held-out loop always `force_cpu=True` |
| Free-form contract_live_author | **Fixed** (`43c8ce3`) | structured `contract_live` object |
| D4 `preserve_ok` always true | **Fixed** | `"frankenstein" in preserve_key` only; no `or bool(key)` |
| gitignore drops NPU proofs | **Documented** | proofs under `docs/evidence/npu/` (tracked); `prime/state/npu/` scratch only |
| buddy_l8 `_run` no timeout | **Fixed** | `PRIME_BUDDY_L8_TIMEOUT` (default 300s); TimeoutExpired → failed step |
| unknown `prefer` → LFM | **Fixed** | reject unknown with NEED_INFO; LFM only auto fallback or prefer=lfm |
| E3 held_out OR rate bypass | **Fixed** | `held_out` mandatory + `label_parity_rate` present + `n>=2` |
| D17 count assert not gate | **Fixed** | artifact `count_ok` required by D17 gate + writer |
| rerank trust_remote unpinned | **Fixed** | refuse jina AutoModel path unless `PRIME_JINA_RERANK_REV` set; CE fallback |
| synthesize ort_force_cpu=True | **Fixed** | require `run.ort_force_cpu is True`; persist False if missing |
| LFM fiber env-only | **Fixed** | `glue_agreement(fiber_mode=…)`; dual_enter passes request mode |
| held-out missing neutral | **Fixed** | neutral fixture + `labels_covered` / complete gate |
| red-cert disposition on green | **Fixed** | inject red for route check; live green no longer fails CI |
| contract_live commit_hint only | **Fixed** | full `commit` + `tree` SHA on certificate |
| soft_critical wrong names | **Fixed** (`de28339`) | set uses `fn.__name__` (d1_aboutness, d8_accel_npu, …) |
| D15 no freshness | **Fixed** | `_artifact_fresh` required; stale → WARN |
| D17 mixed count basis | **Fixed** | recompute from cells; require integrity fields on artifact |
| ORT force_cpu drops cache | **Fixed** | reuse CPU session when EP class matches |
| intent_self asymmetric | **Fixed** | same stalk both sides + expect_hit |
| hexagon present hardcoded | **Fixed** | `present: null` until probe |
| single-logit threshold | **Fixed** | coupled to ONEWAY_P |
| --mode gap as open defect | **Documented closed** | VV_FINAL historical note |
| rerank rev not immutable | **Fixed** | full 40-char commit SHA only (`_immutable_rev`) |
| QDQ silent uint8 default | **Fixed** | `ValueError` on unsupported act/weight |
| MD detail invalid JSON splice | **Fixed** | `{"truncated":true,"preview":…}` wrapper |
| truth_loop doubles static NLI | **Fixed** | default `PRIME_TRUTH_ROUNDS=1` |
| prefer=htp doc incomplete | **Fixed** | red refuse + green still auto (HTP product unwired) |
| disposition string-only | **Fixed** | runtime incomplete-labels + D17 recompute checks |
| D17 empty cells PASS | **Fixed** | `cells_ok` + type check; empty/malformed fail |
| evidence absolute paths | **Fixed** (re-scrubbed) | all `docs/evidence/npu/*.json` host-path free |
| substrate_ok soft OR | **Fixed** | fiber.ok only |
| nli_engine warm probe | **Fixed** | agreement engine first |
| preserve_alone any frankenstein | **Fixed** | selected model alone |
| labels_covered optional | **Fixed** | mandatory complete set for green |
| push_domains env/scout leak | **Fixed** | finally restore env + scout fiber |
| jina pool/gguf mismatch | **Fixed** | force restart when meta differs |
| golden claims structure | **Fixed** | require sha256, required_ids, n_claims |
| ensure invalid mode | **Fixed** | reject non scout/preserve |
| QDQ SHA isolation | **Fixed** | qdq_sha256 + held_out_manifest_sha256 on evidence |
| D12 criticality split | **Fixed** | session-load + fixture + exception all critical=True |
| D17 portable is_file only | **Fixed** | runtime parse integrity JSON + count match |
| Jina unverified already_running | **Fixed** | no META/gguf/pooling → force_restart (fail closed) |
| QNN silent CPU attach | **Fixed** | strict session asserts active provider is QNN |
| QDQ default uint8 | **Fixed** | default act=uint16 for HTP measure |

**Disposition verifier:** `python prime/scripts/verify_cr_disposition.py` → `CR_DISPOSITION_VERIFY_PASS`  
Runtime CI checks: `runtime_held_out_disjoint`, `runtime_cert_structured`, `runtime_red_cert_no_htp_first` (+ D4 frankenstein-only + evidence archive note).

Stale inline comments may still map to new line numbers after re-review; this table + verifier are authoritative until CR re-submits on current HEAD.
