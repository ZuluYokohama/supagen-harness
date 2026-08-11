# geosteern — TVT prediction from steering-policy dynamics

Predicts True Vertical Thickness (TVT) along the unlogged tail of a horizontal
well, given the logged prefix plus geometry and gamma ray over the whole well.

## The idea

TVT is **not a geological surface to be inverted from logs** — it is a *human
control output*. A geosteerer actively holds the bit in zone, so TVT stays near
a target and its excursions have learnable dynamics. Every approach that treated
TVT as geology failed; treating it as control worked immediately.

The model predicts a correction to the hold-last-TVT baseline:

```
TVT(i) = t_last + k · f(x_i)
```

`k` is shrinkage calibrated on grouped-by-well out-of-fold predictions.

Its strongest single signal is that `TVT = S − Z`, where `Z` (wellbore TVD) is
**measured over the prediction horizon**. The model learns what fraction of that
known vertical motion the structural surface absorbs — information a constant
predictor structurally cannot use.

## Validated performance

Holdout of 263 wells, never trained on (deterministic hash split, `data.split_wells`):

| method | median | mean | p90 |
|---|---:|---:|---:|
| **policy model** | **9.21** | **11.44** | **20.70** |
| hold last TVT | 11.35 | 13.23 | 24.21 |
| oracle constant *(uses labels)* | 6.96 | 8.04 | 13.66 |

- **+1.79 ft** mean improvement over hold, 95% CI **[1.29, 2.29]**, P(gain>0) = 100%
- beats hold on **69.2%** of wells; beats the per-well oracle constant on **28.5%**
- captures **34.5%** of the hold → oracle-constant headroom

### Stronger estimate: full-corpus cross-validation

The table above uses one 263-well holdout with 510 training wells. Repeating it as
4-fold grouped CV over **all 773 wells** — each predicted exactly once by a model
that never saw it, ~3x the evaluation data, shrinkage calibrated inside each fold —
gives a better-supported figure for the same model:

| method | median | mean | p90 |
|---|---:|---:|---:|
| **policy model** | **8.81** | **10.96** | **18.78** |
| hold last TVT | 10.66 | 12.81 | 22.97 |
| oracle constant *(uses labels)* | 6.92 | 7.86 | 13.11 |

Beats hold on **71.0%** of wells. Protocol: 4-fold `GroupKFold` over well id, each
fold fitting via `model.fit` and calibrating shrinkage on its own grouped OOF, then
scoring with `per_well_rmse`. The driver was a scratch script and is not in the
repo; `cli.evaluate` reproduces only the single-holdout table above.

**Treat this as ~8.8–9.0 median, not 8.81.** Re-running the same model with a
finer evaluation stride and different fold construction gives 8.97 / 10.93 /
19.13. That 0.16 ft wobble is pure protocol sensitivity — worth knowing, because
it is larger than several candidate improvements that have been tested, and any
effect below it is unmeasurable here.

On the three shipped test wells specifically (model retrained with them excluded,
scored against their labelled copies in `train/`): model 9.84 vs hold 10.21 mean,
winning on 2 of 3. **n=3 — treat the actual score as close to a coin flip.**

## Usage

```bash
python -m geosteern.cli evaluate                       # reproduce the table above
python -m geosteern.cli train   --out tvt_model.json   # fit on all labelled wells
python -m geosteern.cli predict --model tvt_model.json --out submission.csv

# --data-dir is a root-level option, so it goes BEFORE the subcommand:
python -m geosteern.cli --data-dir /path/to/data evaluate
```

`--data-dir` defaults to `$GEOSTEERN_DATA`, else `C:/PRIMEdEV-1/GeoSteerN-Codex`.
It must contain `train/`, `test/`, and `sample_submission.csv`.
Requires pandas, numpy, scikit-learn, lightgbm. `evaluate` takes ~2 min on CPU.

## Two data facts you need to know

**1. Formation tops are train-only.** `train/*horizontal_well*.csv` carries
ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA; `test/*` does not. The identity
`TVT = ANCC − Z + c` holds to ~0.005 ft on all 773 wells but is **leakage** and
cannot be evaluated at inference. This model never touches those columns.

**2. The three test wells are duplicated in `train/` with full labels.** All
14,151 `sample_submission` rows are recoverable by index lookup. This model does
*not* do that — it predicts them genuinely. Flagging it because it makes the
public test set unusable as a scoreboard; validate on held-out train wells.

## What was tried and rejected

Each was implemented, measured on the same holdout, and dropped:

| approach | result |
|---|---|
| structural / datum surface extrapolation | plane fit 40 ft median — worse than doing nothing |
| GR-to-typewell registration (6 variants) | objective is exact at truth, uninformative elsewhere |
| persistence-based marker matching | no wider basin than plain L2; slightly worse |
| multi-scale template blurring | monotonically worse with blur radius |
| neighbour-well features | median 9.21 → 9.55; CI straddles zero |
| within-well control features | heavily used (ranks 1, 2, 5, 7, 8, 9 of 69) but redundant with `dmd` |
| quantile-0.5 objective | significantly *worse* — the metric is RMSE, minimised by the conditional mean |
| heteroscedastic shrinkage | spread correlates +0.32 with error but per-bin shrinks are non-monotonic |
| learned sequence encoder (dilated CNN) | 10.21 median; overfits past epoch 50 on 510 wells |
| survey geometry (DLS, build/turn rate, azimuth) | **retired on a valid test — see below.** The original CV result was uninformative: those features were differenced at 1 ft, where the signal is 100% coordinate-rounding artifact. Rebuilt correctly and retested, geometry still adds nothing |
| split-point augmentation (3 prefix cuts/well, 3x training rows) | screened well (mean 11.44 → 11.05), **not confirmed**: CV mean +0.096 ft CI [-0.151, +0.354], better on 49.9% of wells, Wilcoxon p=0.86. Does improve calibration — optimal shrink rises 0.85 → ~1.0 — but not accuracy |

### Survey geometry, retested

The first attempt reported "+0.31 median on one holdout, refuted by 773-well CV:
better on 49.9% of wells, p=0.71." That refutation happened to reach the right
conclusion for the wrong reason, and it is worth recording why.

X/Y/Z are stored to 0.01 ft. Differencing them at 1 ft yields `0.01/1` rad =
**57.3°/100 ft of pure rounding**; measured apparent dogleg at that baseline is
**57.15** against the 57.30 prediction. The original DLS/build/turn features
contained no geometry at all. A 49.9% win rate is precisely what a pure-noise
feature produces — so that number described the *inputs*, not survey geometry.

Rebuilt through `research/survey_primitives` (signed inclination unfolded past
90°, azimuth relative to each well's own derived section azimuth, dogleg and
build/turn rates at course scale) and rerun on the same 4-fold grouped CV with
shrinkage calibrated inside each fold:

| set | window | median gain | better on | Wilcoxon p |
|---|---:|---:|---:|---:|
| N/S subset, n=130 | 96 ft | +0.041 | 43.8% | 0.255 |
| N/S subset, n=130 | 128 ft | −0.006 | 50.8% | 0.832 |
| corpus, n=773 | 96 ft | +0.067 | 50.5% | 0.606 |
| corpus, n=773 | 128 ft | +0.167 | 51.5% | 0.126 |

Null on four counts. Every gain sits under the ~0.2 ft resolution floor. Win
rates are coin flips. The two windows — both inside the swept usable band —
disagree by 2.5×, so protocol sensitivity exceeds the effect. And decisively,
the effect is *weakest* where it should be strongest: the N/S subset is the
~130 wells whose azimuth is least corrupted by axial magnetic interference
(`sin(I)·sin(A_mag) < 0.34`), and geometry there is worthless at both windows,
with a negative mean at 96 ft.

The +0.167 ft at 128 ft is the largest number in the experiment and the one to
distrust, for the same reason the original +0.31 was: a positive point estimate
with a ~50% per-well win rate is the signature of noise, not signal. This
project has now made that error twice and caught it twice.

Geometry is retired — this time on evidence that could have shown otherwise.

## Learning curve — still improving at 580 wells

Measured **inside each of the 4 CV folds**, so both the training subset and the
evaluation set vary (shrinkage held fixed at 0.85 so only data volume differs).
Mean ± sd across folds:

| wells | median | mean | beats hold |
|---:|---:|---:|---:|
| 60 | 10.60 ± 0.77 | 12.89 ± 1.12 | 54.8% |
| 120 | 10.07 ± 0.39 | 12.48 ± 0.13 | 58.2% |
| 200 | 9.50 ± 0.29 | 11.96 ± 0.37 | 62.7% |
| 300 | 9.25 ± 0.54 | 11.41 ± 0.58 | 66.5% |
| 400 | 9.14 ± 0.44 | 11.15 ± 0.33 | 67.4% |
| 580 | 8.78 ± 0.50 | 10.89 ± 0.57 | 72.5% |

**The curve is still descending at 580 wells.** The fold-mean median falls from
9.25 at 300 wells to 8.78 at 580 — a **+0.47 ft** gain — and every size step is
monotone in both median and beats-hold (54.8% → 72.5%).

Treat that magnitude cautiously. It is comparable to the per-size spread across
folds (sd 0.50 at 580, 0.54 at 300), and the per-fold deltas recorded at the
time (0.45, 0.31, 1.03) average 0.60, which cannot be reconciled with the
table's 0.47 — a fourth fold's delta went unrecorded, and the driver script no
longer exists to recover it. The monotone trend across six sizes is the durable
claim; the size of the gain is not pinned down.

More wells should therefore help, and the learned sequence encoder — which was
still improving when it overfit at 510 — becomes the more interesting revisit.

> **This section previously claimed the opposite.** An earlier curve used one dev
> split against one fixed holdout, single run per size, and read 300 → 510 as flat
> (9.25 → 9.23). Repeating it across folds shows that was noise: the 400-well point
> had bounced *up* to 9.37 on that split, flattening a curve that is actually
> monotone. The driver was a scratch script and is not in the repo; the protocol
> is as described above — subsample the training wells inside each CV fold at
> each size, shrinkage pinned at 0.85, and score the fold's own holdout.

## A note on method, learned twice

Single-split results misled this project in **both** directions:

- survey-geometry features looked like a +0.31 ft median gain with P(helps)=68%;
  full-corpus CV showed noise (better on 49.9% of wells, p=0.71) — an over-claim;
- the learning curve looked saturated at ~300 wells; repeating it across folds
  showed steady improvement to 580 — an under-claim that nearly closed off the
  most promising direction.

One split is one sample. Repeat across folds before believing a difference in
either direction; it costs ~10 minutes and both errors above were caught that way.

There is also a floor on what can be measured at all. The same model scores 8.81
or 8.97 median depending only on evaluation stride and fold construction, so
**effects below ~0.2 ft are not resolvable on 773 wells** by any of these
protocols. Two candidates (geometry +0.05, augmentation +0.11) sat under that
floor and both showed a ~50% per-well win rate — the giveaway that a positive
point estimate is noise.

## Layout

```
data.py      loading; works identically on train and test wells
features.py  well-level context + per-point design matrix
model.py     fit / OOF shrink calibration / predict
cli.py       evaluate | train | predict
```

Calibrate shrinkage on **grouped** OOF — in-sample calibration picks k≈1.0 and
costs ~0.3 ft. Model hyperparameters are tuned; dropping to 600 trees / lr 0.05 /
3 folds costs ~0.4 ft.
