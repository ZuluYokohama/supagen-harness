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

On the three shipped test wells specifically (model retrained with them excluded,
scored against their labelled copies in `train/`): model 9.84 vs hold 10.21 mean,
winning on 2 of 3. **n=3 — treat the actual score as close to a coin flip.**

## Usage

```bash
python -m geosteern.cli evaluate                                  # reproduce the table above
python -m geosteern.cli train   --out tvt_model.pkl               # fit on all labelled wells
python -m geosteern.cli predict --model tvt_model.pkl --out submission.csv
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

The sequence encoder is the one worth revisiting: it lost by ~1 ft while starved
of data. Feature engineering and objective changes have both plateaued, so
further gains most likely need **more wells**, not more modelling.

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
