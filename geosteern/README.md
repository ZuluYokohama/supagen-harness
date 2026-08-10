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

Beats hold on **71.0%** of wells. Reproduce with `scratchpad/confirm.py`.

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
| survey geometry (DLS, build/turn rate, azimuth) | promising on one holdout (+0.31 median), **refuted** by 773-well CV: better on 49.9% of wells, Wilcoxon p=0.71 |
| split-point augmentation (3 prefix cuts/well, 3x training rows) | screened well (mean 11.44 → 11.05), **not confirmed**: CV mean +0.096 ft CI [-0.151, +0.354], better on 49.9% of wells, Wilcoxon p=0.86. Does improve calibration — optimal shrink rises 0.85 → ~1.0 — but not accuracy |

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

**The curve is still descending at 580 wells.** Going 300 → 580 improves the
median by **+0.60 ft** (per fold: 0.45, 0.31, 1.03), against a fold-to-fold sd of
0.38 — so the gain clears the noise, and every size step is monotone in both
median and beats-hold (54.8% → 72.5%).

More wells should therefore help, and the learned sequence encoder — which was
still improving when it overfit at 510 — becomes the more interesting revisit.

> **This section previously claimed the opposite.** An earlier curve used one dev
> split against one fixed holdout, single run per size, and read 300 → 510 as flat
> (9.25 → 9.23). Repeating it across folds shows that was noise: the 400-well point
> had bounced *up* to 9.37 on that split, flattening a curve that is actually
> monotone. Reproduce with `scratchpad/curve_cv.py`.

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
