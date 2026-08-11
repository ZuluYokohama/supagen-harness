# GeoSteerN — measurement-respecting feature system

## 0. The finding this design exists to honour

The corpus presents ~5,300 rows per well at 1 ft spacing. Measurement says
otherwise:

| measurement | result | meaning |
|---|---|---|
| course length (arc-fit phase contrast, 60 wells) | **95–96 ft**, contrast 0.65 vs 0.02 null | real survey stations, ~30× above a no-station null |
| within-course fit vs perfect arc | in-plane 0.00298 ft, floor 0.00285 | **pure min-curvature interpolation** |
| within-course out-of-plane | 0.00298 ft = 0.01/√12 | quantization only, no wander |
| per-foot direction signal | 57.15 vs 57.30 °/100 ft predicted | **100% rounding artifact** |
| survey accuracy (reference pack) | ISCWSA Rev 0, 95%, ±0.5° MWD | tens of feet, vs 0.01 ft stored |

**One station per ~96 ft carries information. Everything between is derived.**
A 5,555 ft lateral is ~58 observations, not 5,555. The system below makes that
structurally impossible to forget.

---

## 1. Requirements

**Functional**
- Reconstruct the station grid per well; label every row `measured` or `derived`
- Emit canonical survey records (signed inclination, referenced azimuth)
- Produce features only at course resolution
- Attach an uncertainty to every value that feeds a model

**Non-functional**
- Determinism and byte-reproducibility (the repo is hash-sealed; see `SEAL_STATE.md`)
- Full corpus reprocessing in minutes on CPU
- Effective-N–aware evaluation: no interval computed as if 1-ft rows were independent

**Constraints**
- 773 wells; no plans, no as-drilled station data, no covariances, and **no
  telemetry channels** — `aTFA`, `CInc` and `CAzm` are absent, so the entire
  continuous steering record is missing (see `DATA_REQUEST.md` §1.0)
- BHA sensor offsets unknown, and bit-to-gamma differs from bit-to-directional,
  so GR and geometry are misregistered by an unmeasurable amount (§1.0b)
- Measurement floor ~0.2 ft: effects below it are unverifiable on this corpus
- Formation tops are train-only leakage; test wells duplicated in `train/`

---

## 2. High-level design

```text
 raw 1-ft CSV
      |
      v
[L0] survey_primitives ............ IMPLEMENTED: windowed_survey,
      |                             derive_section_azimuth, vertical_section,
      |                             azimuth_error_multiplier. Window band
      |                             (96-128 ft) swept, not assumed.
      v
[L0b] StationReconstructor ........ fit (course_length, phase) per well
      |                             emit station index + provenance flag
      v
[L1] CanonicalSurvey .............. MD, Inc(signed), Azi(ref stated), TVD,
      |                             VSEC, DLS  -- AT STATIONS ONLY
      v
[L2] UncertaintyModel ............. per-station covariance (ISCWSA when
      |                             available; documented surrogate until then)
      v
[L3] CourseFeatures ............... course-indexed only; sub-course is
      |                             unrepresentable by construction
      v
      +----------------------+----------------------+
      |                      |                      |
 [L4a] CPU              [L4b] GPU             [L4c] GPU
 GBDT on course        sequence model         GP w/ derivative
 features              over course tokens     observations
      |                      |                      |
      +----------------------+----------------------+
                             |
                             v
                 [L5] Evaluation — grouped by well,
                      effective-N intervals, 0.2 ft gate
```

**The load-bearing property is L3's type.** Features are indexed by
`(well_id, course_idx)`. There is no API that accepts a foot index. A
sub-course feature cannot be expressed, so the class of bug that invalidated
the survey-geometry result cannot recur.

---

## 3. Contracts

### L0 → L1

```text
StationIndex:
  well_id            str
  course_length_ft   float     # fitted, ~95-96
  phase_ft           float
  station_md         float[]   # reconstructed station MDs
  fit_contrast       float     # accept only if >> null (~0.02)
  provenance         enum{FITTED, DELIVERED}
```
`DELIVERED` when real station data arrives and reconstruction is retired.

### L1 → L3

```text
SurveyStation:
  md, inc_deg, azi_deg, tvd, vsec, ns, ew, dls
  north_reference    enum{GRID, TRUE, MAGNETIC}   # required, no default
  inc_signed         bool                          # must be True
```

**Two invariants, enforced at construction:**
1. `inc_deg` may exceed 90 — 60.6% of lateral samples do. Any transform
   applying `abs()` to the vertical component is rejected in review.
2. `north_reference` has no default. Grid convergence −1.5441° and
   Mag→Grid 8.7414° in the reference pack mean an unstated convention
   is ~9° of silent error.

---

## 4. Dimensioning — value by intent

The "context of merit" per field, dimensioned by what it can actually support:

| field | resolution | independent N / well | supports |
|---|---|---|---|
| station MD/Inc/Azi | ~96 ft | ~58 in lateral | steering features, control error |
| position X/Y/Z | derived | 0 beyond stations | geometry **only** at ≥96 ft baselines |
| TVT_input prefix | 1 ft, labelled | prefix length | anchor, hold-last baseline |
| GR | measured; effective res **unknown, ≥1 ft** | **unknown, ≤ rows** | densest real channel, but not per-foot independent |
| formation tops | per well | train-only | **nothing** — leakage |

**Correction to an earlier draft of this table.** GR was previously listed as
"real per-foot" without qualification. It *is* a real counting measurement —
values are exact integer multiples of a per-well calibration quantum, lattice
residual ~2e-8, with all 80 sampled wells carrying a distinct quantum over a
~200× range. But its high-frequency power sits ~10× below raw 1 ft Poisson
statistics while the lattice stays exact, which only holds if counts are
**accumulated in-tool** and the integer total transmitted. Its effective depth
resolution is the accumulation window, not the 1 ft grid it is presented on.
Real-time gamma is 8-bit, ~110× coarser than the stored values.

That per-well quantum is a free **tool fingerprint** — a well-level covariate
proxying MWD configuration, and therefore vendor, crew and period. For a target
that is a human control output, that is a legitimate feature.

It also raises an *observation-leakage* question distinct from label leakage:
the geosteerer acted on 8-bit, accumulation-smeared, telemetry-delayed gamma,
while the model trains on a finely-quantized post-processed log. Modelling a
human decision against information the human never had is worth testing
explicitly — degrade GR to the controller's information set and check whether
it predicts the decision *better*.

**GR remains the densest real channel, but its independent N is unknown.** It
is a measurement rather than an interpolation, so it is categorically unlike
geometry — designs that treat X/Y/Z and GR as equally dense are wrong. But
"denser than geometry" is not "one independent observation per foot", and the
accumulation finding above forbids that stronger claim.

Until the accumulation window is disclosed, the contract is:

- GR effective depth resolution: **unknown, ≥ 1 ft**
- GR independent N per well: **unknown, ≤ row count**
- aggregation for course-indexed features: mean over the course, and any
  estimator that assumes per-foot independence must be flagged, not shipped

Requesting that window is cheap and it is the only thing standing between this
row and a real number — see `DATA_REQUEST.md` §1.0, which asks for the raw
`gama` channel alongside the decode.

---

## 5. CPU / GPU allocation

**CPU — everything that is exact or small.**
- L0 reconstruction: 60 wells scanned in ~2 min; full corpus minutes
- L1 min-curvature forward/inverse: closed form, vectorised
- L2 covariance propagation: small per-station matrix ops
- L4a GBDT on ~58 features/well: the incumbent, unchanged in cost
- Rationale: nothing here is compute-bound, and GPU histogram builds would
  add nondeterminism against a hash-sealed protocol

**GPU — only where it changes an answer, not the runtime.**
- **L4c GP with derivative observations** — highest value. Directional-derivative
  observations with heteroscedastic ISCWSA noise; addresses the anisotropy
  directly (measured Gram condition median 87, never rank-deficient)
- **L4b sequence model over course tokens** — ~58 tokens/well instead of ~5,500.
  This is the dilated-CNN revisit as a different model class, not a retune;
  it is the most plausible fix for overfitting at 510 wells
- **Finer inducing grids** than the current 292 nodes at 5,000 ft cells

**Explicitly not GPU:** LightGBM (small, determinism risk), GR registration
(refuted structurally, not by compute), anything targeting <0.2 ft.

---

## 6. Trade-offs

| decision | buys | costs | revisit when |
|---|---|---|---|
| Course-indexed features only | makes the invalidating bug unrepresentable | discards any real sub-course signal | never — measured to be zero |
| Fitted stations (L0) vs delivered | usable now | fit is inference, not ground truth | station data arrives → `DELIVERED` |
| Drop sub-96 ft geometry | removes 57°/100 ft of artifact | loses features that "worked" on one split | never |
| Quantization as noise floor | simple | wrong — true floor is ISCWSA, ~1000× larger | covariances arrive |
| Per-well grouped evaluation | honest intervals | wider intervals, fewer "wins" | never |

**The uncertainty row is the one I would flag hardest.** Until real
covariances arrive, any uncertainty this system reports is a surrogate. It
should be labelled as such at the API boundary, not quietly treated as
calibrated.

---

## 7. What I would revisit as this grows

1. **When plans arrive**, L3 gains control-error features and the model class
   changes from "estimate a surface" to "estimate residual of a controlled
   process." That is the largest single upgrade available and it is a data
   request, not an engineering task.
2. **If station data arrives**, L0 becomes a validator instead of an estimator
   — compare fitted vs delivered stations as an accuracy check on this design.
3. **Effective-N audit** across the existing CV machinery: confirm no interval
   is computed per-point. Grouping by well should protect the headline numbers,
   but that is an assumption until checked.
4. **The 0.2 ft floor is a property of 773 wells**, not of the method. It moves
   only with more wells — which the learning curve says are still helping at 580.
