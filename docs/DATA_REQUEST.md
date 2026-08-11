# GeoSteerN — data request

**Purpose.** The current corpus (773 wells: `MD, X, Y, Z, GR, TVT_input`, resampled
to 1 ft) cannot support the modelling it is being asked to support. This document
states what is missing, what each item unlocks, and how to verify a delivery.

Every claim below is measured against the existing corpus, not assumed. The
reference for what a complete deliverable looks like is a standard DD pack —
Proposal Geodetic Report plus wall plot (example inspected: Noble Energy
Calamity Jane 2133H REV0, Schlumberger `ExchangeServices 2.9.365.0`).

---

## What is wrong with the corpus as delivered

**1. It was resampled to 1 ft, and that destroyed the survey stations.**
Every MD in every well is an exact integer on a 1 ft grid. A periodicity scan
over 80 wells (P = 20–200 ft) found no station structure at any period —
strongest peak 1.24× baseline, and 94.5 ft came in *below* baseline. The
original station MDs are unrecoverable.

**2. Sub-100-ft geometry is 100% quantization artifact.**
X/Y/Z are stored to 2 decimals. A 0.01 ft rounding across a 1 ft step is
0.573° of angular noise = 57.3°/100 ft. Measured apparent dogleg at 1 ft
baseline across 40 wells: **57.15°/100 ft**, against a predicted pure-artifact
value of **57.30**. There is no geometry in the per-foot signal at all.

**3. Stored precision is ~1000× finer than survey accuracy.**
The reference pack specifies `ISCWSA Rev 0, 3-D 95% confidence, 2.7955 sigma`
with an `NAL_MWD_PLUS_0.5_DEG` tool and `EOU Freq 1/100 ft`. Real positional
uncertainty at ~18,400 ft MD is tens of feet laterally. Storing 0.01 ft is
precision theatre and it invites features that fit noise.

**4. There is no setpoint, so control error cannot be computed.**
The project's core thesis is that TVT is a *human control output*. The
reference plan lands at 90.4° inclination and holds it, DLS 0, for 7,484 ft.
The corpus's as-drilled laterals scatter p5 86.5° to p95 93.6° around that.
**That scatter is the geosteering.** Without the plan there is no reference to
difference against, and the control framing cannot be operationalised.

**5. Inclination sign is lost in practice.**
60.6% of lateral samples in the corpus exceed 90° (toe-up); 60.8% of wells run
toe-up for more than half the lateral. Any pipeline using `abs()` on the
vertical component folds 92° onto 88° and destroys the steering state. Delivered
inclination must be signed and unfolded past 90°.

**6. Formation tops are train-only, which is leakage by construction.**
`train/*horizontal_well*.csv` carries ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA;
`test/*` does not. `TVT = ANCC − Z + c` holds to ~0.005 ft on all 773 wells.
The identity is exact and unusable.

**7. The three test wells are duplicated in `train/` with full labels.**
All 14,151 submission rows are recoverable by index lookup, which makes the
public test set useless as a scoreboard.

---

## Tier 1 — changes the model class

### 1.1 Directional plan per well
The Proposal Geodetic Report equivalent. Specifically the **control-point table**,
which is the setpoint program:

```
Comments | MD | Incl | Azim(Grid) | TVD | VSEC | NS | EW | DLS
```

where `Comments` carries the directional intent verbatim — `KOP - Build 10°/100'
DLS to Land`, `Landing Point @ 90.4° Inc`, `Hold 8° Inc`, `PBHL`. The instruction
text is not decoration; it is the control program, and it is the single most
valuable field in the pack.

*Unlocks:* control error = as-drilled − plan. Turns TVT prediction from
"estimate a surface" into "estimate the residual of a controlled process,"
which is what the evidence already says it is.

### 1.2 As-drilled definitive survey at station resolution
Not resampled. Station spacing as acquired (100 ft in the reference pack).

```
MD | Incl (signed, >90 permitted) | Azim | TVD | VSEC | NS | EW | DLS
```

*Unlocks:* real geometry. The existing "survey geometry refuted by 773-well CV
(better on 49.9% of wells, p=0.71)" result was computed on quantization
artifact and does not stand — geometry has never actually been tested.

---

## Tier 2 — makes uncertainty honest

### 2.1 Survey program and error model
Tool type, EOU frequency, ISCWSA revision, sigma / confidence level, and the
MD intervals each tool covers. Per-station covariance if it can be exported;
otherwise the inputs to compute it.

*Unlocks:* per-observation noise. Enables heteroscedastic weighting and honest
uncertainty instead of treating 0.01 ft as truth.

### 2.2 Geodetic header
CRS and zone; grid convergence; grid scale factor; magnetic declination with
model and date; TVD reference datum and elevation; vertical section azimuth
and origin; explicit north reference (grid / true / magnetic).

*Unlocks:* correct cross-well geometry. Azimuths are meaningless without a
stated north reference — the reference pack shows `Grid Conv −1.5441°` and
`Total Corr Mag→Grid 8.7414°`, so mixing conventions introduces ~9° of error.

---

## Tier 3 — context features

### 3.1 Pad and offset relationships
Which wells share a pad and drilling program; offset well identifiers;
leaseline and hard-line geometry (the reference wall plot names seven offsets
plus `Leaseline` and `330'x200' HL`).

*Unlocks:* a valid retest of neighbour-well features, previously rejected at
median 9.21 → 9.55 but built blind to pad membership and lease constraints.
Also explains the observed heading concentration — 75% of corpus wells fall in
a ~26° window, which drives the azimuth Gram condition (median 87).

### 3.2 Formation tops for test wells, or an explicit statement they are withheld
Current asymmetry is silently exploitable.

### 3.3 Steering records, if they exist
Slide/rotate intervals, toolface, motor yield, bit depth vs hole depth.

*Unlocks:* the literal control input alongside the control output. This is the
highest-value item in the entire request if it exists.

---

## Delivery requirements

- **Do not resample.** Native station data. Resampling to any grid destroys
  station identity and manufactures interpolation artifacts.
- **Full precision, or state the true precision.** Do not pad to 2 decimals.
- **Signed inclination**, unfolded past 90°.
- **State the north reference and units** explicitly per file.
- **Do not "improve" the data by densifying it.** Interpolating the same
  surveys to a finer grid adds zero information and actively misleads.

## Verifying a delivery before trusting it

Run these on arrival. Each corresponds to a defect found in the current corpus:

1. **MDs are not on a uniform integer grid.** `np.diff(md)` should vary.
2. **Station spacing matches the stated EOU frequency.**
3. **Inclination exceeds 90° somewhere** in toe-up laterals. If nothing exceeds
   90°, the sign has been folded — reject.
4. **Decimal precision is greater than 2** on X/Y/Z, or precision is documented.
5. **Apparent DLS at native station spacing is physical** (< ~15°/100 ft).
   Compare against `degrees(quantum / baseline) × 100` — if observed tracks
   that, you are looking at rounding.
6. **Plan and as-drilled resolve to the same well** and share a north reference.

## Not worth requesting

- Higher-frequency resampling of existing surveys.
- Anything justified by an expected improvement below ~0.2 ft: the corpus
  cannot resolve effects that small. The same model scores 8.81 or 8.97 median
  depending only on evaluation stride and fold construction, so effects under
  that threshold are unverifiable regardless of how they are produced.
