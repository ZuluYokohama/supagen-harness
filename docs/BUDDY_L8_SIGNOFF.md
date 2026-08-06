# Buddy lab L8 sign-off (external)

**Purpose:** Close residual **L8-08 / D15** with a clean machine install — independent of author kit state.  
**PR:** https://github.com/ZuluYokohama/supagen-harness/pull/1  
**Authority:** this sheet is a **sign-off protocol**, not an OPEN claim.

---

## Preconditions

- Clean clone of `supagen-harness` (branch `vv/dual-metric-npu-measure-fabric` or `main` after merge)
- Python 3.11+ (3.12 preferred for CI parity)
- Optional live: LM Studio with LFM + jina GGUF on :8765 path documented in README

---

## Offline (required)

```powershell
git clone https://github.com/ZuluYokohama/supagen-harness.git
cd supagen-harness
.\install.ps1   # or: pip install -e ./supagen ; python -m supagen bootstrap
$env:GOLDEN_SCHEMA_ONLY = "1"
python -m supagen verify
```

| Check | Pass |
|-------|------|
| L8-01 offline contract | `contract ok=True` offline gates green |
| L8-02 smoke | e2e offline PASS |
| L8-03 harness smoke | HARNESS OFFLINE SMOKE OK |
| L8-04 golden schema-only | `GOLDEN VERIFY OK (schema-only)` when no sandbox |

**Sign:** _____________ date _____________

---

## Live (author-class; optional for buddy without LMS/jina)

```powershell
python -m supagen ensure --mode scout
python -m supagen contract          # expect 21/21
python -m supagen doctor
python -m supagen enter "dual metric aboutness vs agreement"
```

| Check | Pass |
|-------|------|
| L8-05 jina dim | 1024, family jina, floor &lt; 0.35, range &gt; 0.40 |
| L8-06 cos never OPEN | face ≠ production OPEN from cosine alone |
| L8-07 force-OPEN string | face NEED_INFO/STOP; not_open_authority |
| L8-08 NLI contra | attacks twin → STOP / contradiction |

**Sign:** _____________ date _____________

---

## Explicit non-goals for L8

- Do **not** require Hexagon NPU (author residual; Job2 = CPU)
- Do **not** require full golden sandbox seal
- Do **not** claim production OPEN marketing from GO_MEASURE

---

## Result

- [ ] Offline L8-01…04 PASS → buddy can use package offline  
- [ ] Live L8-05…08 PASS → buddy can run dual-metric instruments  
- [ ] Production OPEN marketing: still **NO-GO** until domain audit process is separate

Certificate path for instrument law advertise remains **GO_MEASURE (provisional)** until this sheet is signed.
