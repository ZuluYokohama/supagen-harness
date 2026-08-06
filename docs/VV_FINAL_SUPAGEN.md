# Final V&V Plan — Supagen Agent Harness

**Classification:** Pre-advertisement Verification & Validation (V&V)  
**Standard of care:** Independent, falsifiable, evidence-backed go/no-go  
**Reputation rule:** If a gate is red, **do not market**. Residue never forced.

---

## 0. Why this plan exists (known defects — not vibes)

These are **already measured** on this kit. V&V must re-confirm them under controlled conditions, not hand-wave.

| Finding | Evidence | Implication for marketing |
|---------|----------|---------------------------|
| Cosine ≠ agreement | Pasta floor, negation blindness, granite 0.598 cos vs mutual entail | **Never** claim “semantic glue = cosine” |
| LFM 1.2B identity **p ≈ 0.29 FAIL** | `holonomy_v3_lfm12b_identity_floor.json` | LFM is **scout only**, not holonomy subject |
| Frankenstein identity **p ≈ 0.88 PASS** | identity chain flat d1–d4 | Only proven multi-hop / preserve fiber |
| Gemma-12B **p ≈ 0.38 FAIL** | gemma floor | Capacity ≠ preservation |
| LMS jina typed as LLM | silent nomic remap if base=:1234 | Job1 must be **side-server :8765** |
| Residency thrash | `seamless_substrate` unloads frankenstein as “heavy” | Default load ≠ holonomy mode |
| KB family mismatch | nomic vectors vs jina queries | Retrieve must re-stamp family |

**Core risk:** Advertising “the harness works” while the **default loaded model fails the identity gate** and operators still think cos is the score.

---

## 1. System under test (SUT) — define once

### 1.1 In scope

| Layer | Component | Job |
|-------|-----------|-----|
| **L0** | Install / bootstrap / `supagen verify` | Buddy can replicate offline |
| **L1** | Residency (`ctx_policy`, unload, promote) | Right model, right ctx, no thrash |
| **Job1** | jina aboutness (`:8765`, prefixes, hybrid retrieve) | Aboutness / retrieval only |
| **Job2** | NLI (LFM structured + DeBERTa holonomy judge) | Agreement / identity **p** |
| **Chat scout** | LFM 1.2B or Ministral 3B | dual_enter roles, draft, explore |
| **Preserve fiber** | frankenstein-2.0-i1 | Identity floor + multi-hop only if PASS |
| **Field** | multiplane harness packs / certify | OPEN\|STOP on claims, not on cos |
| **Law** | cert_face, purpose_gate | Cos never promotes OPEN |

### 1.2 Explicitly out of scope (do not claim)

- Production OPEN from local LLM alone  
- Cosine as entailment / “faithfulness”  
- LFM as multi-hop semantic carrier  
- Speculative decoding α until measured on free-RAM box  
- Vision / mmproj planes  
- Guarantees on models not in the matrix  

### 1.3 Modes of operation (must be tested separately)

| Mode | Resident chat | Purpose | Default? |
|------|---------------|---------|----------|
| **SCOUT** | LFM 1.2B (or Ministral) | dual_enter, draft, tools | Daily |
| **PRESERVE** | frankenstein @ policy ctx | identity floor, holonomy ladder | Explicit only |
| **RETRIEVE** | no large chat required | jina + KB hybrid | Always-on Job1 |
| **FIELD** | optional live scout | pack → certify | Offline packs always |

**Go criterion:** Docs + CLI + doctor make these modes **impossible to confuse**.

---

## 2. V&V philosophy (world-class)

1. **Separate instrument validation from subject validation**  
   - First prove jina/NLI/install work.  
   - Then score models with those instruments.

2. **Pass/fail is binary on gates; scores are continuous for ranking**  
   - Gate fail → STOP advertise for that claim.  
   - Scores go in a capacity sheet, not marketing slogans.

3. **Independence**  
   - Holonomy judge = **DeBERTa**, not the model under test.  
   - Certify gate = **external** harness certifier, not scout prose.

4. **Reproducibility**  
   - Fixed seeds, fixed seed sentences (`SEEDS`), fixed packs, recorded LMS catalog keys, free RAM snapshot, ctx loaded, quant.

5. **Negative testing**  
   - Prove the system **refuses** wrong claims (cos-OPEN, draft OPEN, LFM holonomy).

6. **Buddy as external lab**  
   - Clean machine (or clean env) runs install + verify.  
   - If buddy fails, product fails.

---

## 3. Evidence package (what must exist before “advertise”)

| Artifact | Path / command | Required |
|----------|----------------|----------|
| Offline verify log | `supagen verify` exit 0 | YES |
| Live contract | `supagen contract` 21/21 | YES (kit with LMS+jina GGUF) |
| Live e2e | `supagen e2e --live` exit 0 | YES |
| Doctor clean | `supagen doctor` ok | YES |
| Aboutness null A/B/C | jina floor &lt;0.30, range &gt;0.40, family=jina | YES |
| Identity floors | LFM + frankenstein (+ optional gemma) JSON | YES |
| Capacity sheet | `docs/HOLONOMY_CAPACITY.md` updated with this run | YES |
| Residency mode test | SCOUT vs PRESERVE load matrices | YES |
| Harness golden | offline OPEN/STOP packs | YES |
| Buddy install | second machine or clean venv + clone | YES |
| Known-limitations | public README section | YES |
| Sign-off | this plan §12 checklist signed | YES |

All runs store: timestamp, free_gb, loaded models+ctx, git SHA, OS.

---

## 4. Test levels

### L0 — Package / install (buddy lab)

| ID | Test | Procedure | Pass |
|----|------|-----------|------|
| L0-01 | Fresh clone install | `git clone` → `.\install.ps1` | exit 0 |
| L0-02 | No PYTHONPATH | New shell: `python -c "import nomic_metric"` | imports |
| L0-03 | Offline verify | `python -m supagen verify` | exit 0 |
| L0-04 | Bootstrap persistence | reboot shell: import still works | ok |
| L0-05 | Docs honesty | README states LFM scout vs frankenstein preserve | review |

### L1 — Transport & residency

| ID | Test | Procedure | Pass |
|----|------|-----------|------|
| L1-01 | LMS health | `l0_health` / doctor | LMS up when live |
| L1-02 | SCOUT load | unload all → ensure LFM | LFM loaded; frankenstein **not** required |
| L1-03 | PRESERVE load | unload all → load frankenstein only | frankenstein loaded @ policy ctx (≥8k) |
| L1-04 | No silent thrash | After PRESERVE, run dual_enter without mode flag | **Documented** behavior: either keep frankenstein or explicit switch — **must not** unload without log |
| L1-05 | ctx no downgrade | LFM@128k → promote again | stays ≥ prior ctx |
| L1-06 | ctx policy tiers | free=8 sim: LFM≥32k, Ministral≥16k | unit contract |
| L1-07 | Free RAM gate | Attempt 12B alone when free&lt;2GB | fail soft or half-ctx, no crash |

**Critical defect to close in implementation before advertise:**  
`seamless_substrate` currently treats frankenstein as **heavy** and unloads it. V&V L1-04 **fails** until modes are explicit (`--mode scout|preserve`).

### L2 — Job1 aboutness instrument

| ID | Test | Procedure | Pass |
|----|------|-----------|------|
| L2-01 | jina ensure | kill :8765 → `ensure_jina` | auto-restart, dim=1024 |
| L2-02 | jina base isolation | `embed(..., base=LMS:1234)` | family still **jina**, not nomic |
| L2-03 | A paraphrase ceiling | stripped+prefix | cos &gt; 0.70 |
| L2-04 | C pasta floor | stripped+prefix | cos &lt; 0.30 |
| L2-05 | A−C range | | &gt; 0.40 |
| L2-06 | B negation | | cos may stay high; **NLI** must see contradiction on held-out pairs |
| L2-07 | KB family stamp | doctor | `stored_family == live_family` |
| L2-08 | Hybrid retrieve | query dual_enter/aboutness | top hit domain-relevant (METRIC_JOBS-class), cos_q_doc ≳ 0.30 |
| L2-09 | Re-rank FILL | synthetic FILL vs domain | domain ranks #1 |
| L2-10 | Determinism | same string embed 2× | cos(self)=1.0 |

### L3 — Job2 agreement / identity instrument

| ID | Test | Procedure | Pass |
|----|------|-----------|------|
| L3-01 | DeBERTa loads | Judge() | ok |
| L3-02 | Negation pair | premise vs explicit negation | label contradiction (held-out) |
| L3-03 | Paraphrase pair | mutual entail both ways | closed |
| L3-04 | Identity floor protocol | fixed `SEEDS`×8, depth=1 identity | report p, modes, median cos |
| L3-05 | Cosine not gate | for same run, report cos≥0.75 rate separately | cos rate **not** used for PASS |
| L3-06 | Subject ≠ judge | model under test never scores its own entailment | design review |

### L4 — Model matrix (subjects)

**Protocol:** unload all LLMs → load **only** MUT → identity floor L3-04 → record free_gb, ctx, quant.

| ID | Model under test (MUT) | Role claim | Pass if |
|----|------------------------|------------|---------|
| L4-LFM | liquid/lfm2.5-1.2b | scout | p **may fail** identity; must complete dual_enter smoke; **must not** be labeled holonomy-capable in docs |
| L4-FRK | frankenstein-2.0-i1 | preserve | **p ≥ 0.80** |
| L4-MIN | mistralai/ministral-3-3b | scout/structured | dual_enter JSON ok; identity p **recorded** (pass if ≥0.80 **or** labeled scout-only) |
| L4-GEM | gemma-12B (optional) | capacity null | record only; prior fail expected |

**Optional extension (same protocol):** identity chain depths 2–4 on any model with L4-FRK PASS only advertised for multi-hop.

### L5 — dual_enter (integrated SCOUT path)

| ID | Test | Procedure | Pass |
|----|------|-----------|------|
| L5-01 | Substrate | dual_enter ensure | jina ok + fiber loaded |
| L5-02 | Roles | SCOUT/FALSIFY/GLUE/VERDICT | ≥2 non-empty payloads |
| L5-03 | cert_face | | ∈ {OPEN_CANDIDATE, STOP, NEED_INFO} |
| L5-04 | Cos never OPEN | inject high cos, weak NLI | face ≠ production OPEN; `not_open_authority` |
| L5-05 | Contradiction demote | force NLI contradiction | OPEN_CANDIDATE → STOP |
| L5-06 | KB pack | retrieve_kb=True | n_hits≥1 if KB present; top scores logged |
| L5-07 | Fence sanitize | Ministral-style fenced JSON | parses |
| L5-08 | Latency budget | dual_enter cold/warm | record only (informational) |

### L6 — Field harness

| ID | Test | Procedure | Pass |
|----|------|-----------|------|
| L6-01 | Offline smoke | `supagen harness smoke` | exit 0 |
| L6-02 | Pack filmore | pipeline `--pack filmore_magpi` | OPEN on golden multiplane claim |
| L6-03 | Draft STOP | draft single-plane claim | STOP |
| L6-04 | Live scout optional | LFM scout → certify | scout **never** issues OPEN; certifier does |
| L6-05 | Seal | certificate sha present | yes |

### L7 — Negative / adversarial

| ID | Attack | Pass if system |
|----|--------|----------------|
| L7-01 | Market “LFM preserves meaning” | Docs + capacity sheet **forbid** it while p&lt;0.80 |
| L7-02 | Gate OPEN on mean_cos | code path + dual_enter test L5-04 |
| L7-03 | Use nomic query on jina KB | L2-02 + L2-07 |
| L7-04 | Co-load frankenstein + 12B | load fails soft or OOM handled |
| L7-05 | Kill jina mid-session | auto-ensure restores Job1 |
| L7-06 | Empty / filler role JSON | validators reject; NEED_INFO/STOP |

### L8 — Acceptance (reputation gate)

| ID | Criterion | Pass |
|----|-----------|------|
| L8-01 | All L0 offline green on clean machine | yes |
| L8-02 | L2 aboutness instrument green | yes |
| L8-03 | L4-FRK p≥0.80 | yes |
| L8-04 | L4-LFM results published as FAIL identity | yes (honest) |
| L8-05 | Modes SCOUT vs PRESERVE in CLI + README | yes |
| L8-06 | Known limitations section public | yes |
| L8-07 | No claim contradicts capacity sheet | review |
| L8-08 | Buddy (non-author) run signed | yes |

**Advertise only if L8 all PASS.**

---

## 4. Detailed protocols

### 4.1 Identity floor (canonical)

```
unload all LLMs
load MUT only @ ctx_policy(purpose=floor|chat)
DeBERTa Judge independent
for seed in SEEDS[0:8]:
    identity rewrite forward + reverse (depth=1)
    closed = (seed ⊨ final) ∧ (final ⊨ seed)
p = mean(closed)
PASS if p ≥ 0.80
Record: modes, median_cos, cos≥0.75 rate, free_gb, ctx, quant, git SHA
```

**Do not** pass on cos≥0.75 alone.

### 4.2 Aboutness null (canonical)

```
ensure jina :8765
pairs A paraphrase / B negation / C pasta
score stripped + family prefixes only
PASS: C < 0.30, A > 0.70, A−C > 0.40, family=jina
```

### 4.3 Residency mode matrix (authoritative)

| Action | Expected resident |
|--------|-------------------|
| `supagen ensure` (default) | SCOUT fiber (LFM/Ministral); jina side; heavies unloaded |
| `supagen ensure --mode scout` | same as default; explicit |
| `supagen ensure --mode preserve` | frankenstein alone @ policy ctx; scouts unloaded |
| `PRIME_FIBER_MODE=preserve` / `=scout` | env equivalent when CLI flag omitted |
| API | `residency.seamless_substrate(fiber_mode=…)` · `truth_plane.ensure_substrate(mode=…)` · `ensure_all(fiber_mode=…)` |

**L1-04 (modes):** **green** — CLI `--mode`, env, and API all wired (P2/D4 measured).

### 4.4 Buddy lab script (external)

```powershell
git clone https://github.com/ZuluYokohama/supagen-harness.git
cd supagen-harness
.\install.ps1
python -m supagen verify
# install LMS + jina GGUF + load LFM
python -m supagen ensure
python -m supagen verify --live
python -m supagen doctor
# PRESERVE
# unload LFM in LMS UI or via API; load frankenstein
python prime/scripts/run_lfm_identity_floor.py   # expect FAIL ~0.29
python prime/scripts/run_identity_chain.py --model frankenstein-2.0-i1 --depths 1
# expect p ≥ 0.80
```

Capture full console logs as evidence zip.

---

## 5. Implementation gaps that V&V will fail today

These are **not optional polish** — they block L8.

| Gap | Why V&V fails | Fix before advertise |
|-----|---------------|----------------------|
| No `--mode scout\|preserve` | frankenstein unloaded by default ensure | CLI + residency keep list |
| Docs still imply LFM is “the” fiber | Reputation: LFM p=0.29 | Capacity sheet + README roles |
| Harness golden may need sandbox data | Offline buddy without 123abc dumps | Ship minimal golden fixtures in-repo (done partially) or document optional |
| Live V&V needs jina GGUF | Job1 offline without GGUF is partial | Document PRIME_JINA_GGUF; ship download pointer |
| Single-machine bias | Author kit only | Buddy lab L8-08 |

---

## 6. Execution schedule (suggested)

| Phase | Duration | Owner | Exit |
|-------|----------|-------|------|
| **P0** Close residency modes + docs honesty | 0.5–1 day | eng | L1-04 green |
| **P1** Instrument re-baseline (L2+L3) | 0.5 day | eng | L2/L3 green |
| **P2** Model matrix L4 (LFM, FRK, MIN) | 0.5–1 day | eng | sheet updated |
| **P3** dual_enter + harness L5–L6 | 0.5 day | eng | logs archived |
| **P4** Negative L7 | 0.5 day | eng | red-team notes |
| **P5** Buddy lab L0+L8 | 0.5 day | **independent** | sign-off |
| **P6** Go/No-Go review | 1 meeting | eng + you | advertise or STOP |

**Do not skip P5.** Reputation is external validation.

---

## 7. Go / No-Go decision board

### GO (all required)

- [ ] Offline + live verify green on author kit  
- [ ] Buddy install green  
- [ ] jina instrument green (L2)  
- [ ] frankenstein p ≥ 0.80 (L4-FRK)  
- [ ] LFM p published; labeled **scout-only**  
- [ ] SCOUT vs PRESERVE modes ship and tested  
- [ ] Public “Known limitations” matches data  
- [ ] No marketing sentence contradicts capacity sheet  

### NO-GO (any one)

- Identity gate advertised for LFM  
- Cosine described as agreement / faithfulness  
- Default ensure unloads frankenstein without user intent while claiming holonomy  
- Buddy cannot install offline  
- Job1 silently uses nomic against jina KB  

---

## 8. Reporting template (per run)

```text
RUN_ID:
GIT_SHA:
HOST: OS / RAM total / free_gb before / after
LMS: version / models catalog keys
MUT: key / quant / loaded_ctx
PROTOCOL: identity_floor_v3 | aboutness_abc | dual_enter | harness_pack
RESULT: p= / cos.75= / modes= / face= / pack_verdict=
PASS/FAIL:
ARTIFACTS: paths to JSON logs
OPERATOR:
```

Aggregate into `docs/HOLONOMY_CAPACITY.md` + `supagen/state/MEASURED.json` + run archive under `supagen/state/vv/<RUN_ID>/`.

---

## 9. Marketing claims whitelist (post-GO only)

**Allowed (if green):**

- “Dual metric: aboutness (jina) ≠ agreement (NLI).”  
- “Install + offline harness packs verify.”  
- “On this kit, frankenstein 7.2B Q4 passes identity floor p≥0.8; LFM 1.2B does not — use LFM for scout.”  
- “OPEN|STOP on field claims via external certifier.”

**Forbidden until re-measured:**

- “Preserves meaning across rewrites” (without MUT + p).  
- “Embeddings verify claims.”  
- “1.2B is enough for multi-hop fidelity.”  
- “12B is better at preservation” (contradicted).

---

## 10. Immediate next engineering (ordered)

1. **Ship `--mode scout|preserve`** (or ensure flags) so frankenstein stays loaded when intended.  
2. **Freeze capacity sheet** with LFM p=0.29, FRK p=0.88, gemma 0.38.  
3. **Run P1–P3** under this plan; archive logs.  
4. **Independent buddy clone** of `https://github.com/ZuluYokohama/supagen-harness`.  
5. **Go/No-Go meeting** with this checklist — no soft launch.

---

## 11. One-sentence thesis

**The harness is not “LFM + cosine.”**  
It is **jina (aboutness) + NLI (agreement) + explicit SCOUT vs PRESERVE fibers + external certify** — and it is only worth anyone’s time after **frankenstein still PASSes identity, LFM is labeled FAIL for that job, and a stranger can clone and verify offline.**

---

## 12. Sign-off

| Role | Name | Date | Verdict (GO / NO-GO) | Notes |
|------|------|------|----------------------|-------|
| Engineering | | | | |
| Product / reputation owner | | | | |
| Independent buddy | | | | |

**Default until signed:** **NO-GO for advertisement.**
