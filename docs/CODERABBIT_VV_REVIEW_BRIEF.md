# CodeRabbit review brief — comprehensive V&V domains

**Repo:** ZuluYokohama/supagen-harness  
**Intent:** Dual-metric OPEN|STOP harness — aboutness ≠ agreement; NPU measure fabric; SCOUT vs PRESERVE.

Paste/attach this when requesting `@coderabbitai full review`. Reviewers (human + CodeRabbit) must cover **every domain below**. Fail any domain that is unmeasured, overclaimed, or law-violating.

---

## Global law (non-negotiable)

1. **Aboutness must not promote OPEN** — cosine / hybrid / neural rerank are retrieval only.  
2. **NLI owns agreement** — contradiction → STOP; mutual entailment for strong agree.  
3. **Residue never forced** — NEED_INFO/STOP are valid; no force-OPEN.  
4. **Production OPEN** requires external domain audit / harness certifier — not local fiber prose.  
5. **SCOUT ≠ PRESERVE** — LFM identity FAIL (~0.29); frankenstein PASS (~0.88); modes must not confuse.

---

## Domain checklist for CodeRabbit

### D0 — Architecture honesty
- [ ] Hybrid LMS (chat only) vs off-LMS instruments (jina :8765, DeBERTa, rerank, NPU) is documented in code comments / README / `truth_plane.architecture_map`
- [ ] No path claims jina via LMS `:1234` embeddings (silent nomic remap risk)
- [ ] `nomic_metric.default_embed_base` isolates jina from LMS base

### D1 — Job1 aboutness (jina)
- [ ] Prefixes: jina `Query:` / `Document:`; nomic `search_*` only on nomic family
- [ ] Default pooling for v5 is **`last`** not mean (`jina_service`)
- [ ] Prefer v5-small GGUF when present; nano fallback
- [ ] Cosine diagnostic never feeds OPEN gate
- [ ] Bakeoff / null: floor usable, range usable; polarity still blind (expected)

### D2 — Job1.5 neural rerank
- [ ] `rerank_service` ladder (jina-reranker-v3 → bge → MiniLM)
- [ ] `dimensional_parse.retrieve` optional neural blend; score_kind remains aboutness*
- [ ] Not used as agreement

### D3 — Job2 NLI / agreement
- [ ] `glue_agreement` prefer auto: ORT → CE DeBERTa → LFM fallback
- [ ] `mutual_entailment` p-floor (default 0.80)
- [ ] `lms_layers` must not prefer LFM NLI over DeBERTa for production path (flag if `prefer="lfm"` remains on dual metric enter)
- [ ] Contradiction demotes OPEN_CANDIDATE

### D4 — Fiber modes / residency
- [ ] `PRIME_FIBER_MODE` / `seamless_substrate(fiber_mode=scout|preserve)`
- [ ] SCOUT unloads frankenstein (HEAVY)
- [ ] PRESERVE loads frankenstein alone; unloads scouts
- [ ] No default holonomy claims for LFM

### D5 — cert_face / dual_enter
- [ ] High cos + no NLI agree ≠ OPEN
- [ ] NLI contradiction → STOP
- [ ] OPEN only as **OPEN_CANDIDATE** + process ok
- [ ] `PRIME_TRUTH_PLANE` substrate warm path

### D6 — Identity / holonomy
- [ ] Artifacts / docs: LFM FAIL, frankenstein PASS, gemma FAIL
- [ ] Cos≥0.75 never substitutes for mutual entailment p
- [ ] Capacity sheet matches claims

### D7 — Package / supagen CLI
- [ ] Offline contract gates still valid
- [ ] Live contract: jina isolation, floor/ceil/range, dual_enter face
- [ ] ensure/status/enter/aboutness/query/reindex-kb coherence

### D8 — Accel / NPU / Hexagon
- [ ] `npu_qnn.register()` plugin EP required (2.x)
- [ ] HTP only QDQ fixed-shape; FP32 never claimed as NPU
- [ ] Task Manager is **not** V&V oracle (no NPU counters on this OS class)
- [ ] Proof = `htp_profile.csv` HVX/accelerator cycles or stress report
- [ ] DeBERTa QDQ on HTP: session may run; **label parity** must not be claimed if collapsed logits
- [ ] Product default Job2 remains CPU ORT/CE until parity PASS

### D9 — Adversarial / negative
- [ ] attacks: twins; force-OPEN with high cos → STOP
- [ ] Lexical overlap predicts cos glue (not arbitrary)
- [ ] Envelope / raw JSON not embedded unprefixed on bakeoff path

### D10 — Truth plane / loop
- [ ] `request_plane` / truth_loop: MEASURE only; residue/STOP elevate face
- [ ] Precision domains can enable loop
- [ ] No production OPEN from loop alone

### D11 — Field / multiplane harness
- [ ] External certify OPEN|STOP; DRAFT → STOP
- [ ] Offline smoke independent of LMS

### D12 — KB family / dim
- [ ] Index `embed_family` + dim match live jina (1024 for small)
- [ ] Auto reembed on mismatch; never silent cross-family cosine

### D13 — Compute∶HW abstraction
- [ ] `docs/COMPUTE_HW_ABSTRACTION.md` device-split dual metric
- [ ] NPU = measure fabric; CPU = agents + OPEN authority
- [ ] No overclaim of NPU for chat/decode/sheaf sparse

### D14 — Security / ops
- [ ] No secrets committed
- [ ] Large GGUF/ONNX not in git
- [ ] Paths portable or env-overridable (`PRIME_JINA_GGUF`, etc.)

### D15 — Docs / marketing honesty
- [ ] Known limitations: LFM scout-only; cos≠agreement; NPU QDQ/parity residual
- [ ] V&V GO_MEASURE ≠ production OPEN advertise without L8 buddy if claimed

---

## Severity guide for findings

| Severity | Example |
|----------|---------|
| **Blocker** | Cosine promotes OPEN; LFM claimed identity-safe; jina via LMS remap; NPU parity claimed while logits collapsed |
| **Major** | prefer=lfm still on dual_enter path; preserve/scout modes missing; KB family mismatch silent |
| **Minor** | Naming confusion (nomic module hosts jina); docs lag measured numbers |
| **Nit** | Style, unused imports |

---

## Suggested CodeRabbit invocation

```text
@coderabbitai full review

Review this PR against docs/CODERABBIT_VV_REVIEW_BRIEF.md domains D0–D15.
Prioritize law violations (OPEN/STOP), dual-metric separation, fiber modes,
NPU honesty (QDQ/parity), and package contract drift.
Do not approve overclaims of production OPEN or NPU NLI accuracy without evidence.
```

---

## Local re-measure commands (reviewer)

```powershell
python prime/scripts/vv_full_matrix.py
python prime/scripts/vv_push_domains.py
python prime/scripts/npu_stress.py --seconds 20
python -m supagen contract --offline
# live if LMS + jina:
python -m supagen contract
```
