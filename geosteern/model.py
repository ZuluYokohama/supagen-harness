"""Training, calibration and prediction for the TVT steering-policy model.

Predicts the correction to the hold-last-TVT baseline:

    TVT(i) = t_last + k * f(x_i)

`k` is a shrinkage factor calibrated on GROUPED-by-well out-of-fold predictions.
Calibrating it in-sample instead picks k~1.0 and costs ~0.3 ft, because
in-sample predictions look more reliable than they are.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Tuned on the dev split. Reducing to 600 trees / lr 0.05 / 3 folds costs ~0.4 ft.
PARAMS = dict(
    n_estimators=900,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=20.0,
    verbose=-1,
)
TRAIN_STRIDE = 8          # subsample points for training; ~313k rows over 510 wells
N_FOLDS = 4


def calibrate_shrink(oof: np.ndarray, y: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.2, 61)
    return float(grid[int(np.argmin([((s * oof - y) ** 2).mean() for s in grid]))])


def fit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int = N_FOLDS):
    """Fit the model and calibrate shrinkage honestly via grouped OOF."""
    oof = np.zeros_like(y)
    for tr, va in GroupKFold(folds).split(X, y, groups=groups):
        m = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    shrink = calibrate_shrink(oof, y)
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y)
    oof_r2 = 1.0 - ((shrink * oof - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(model=model, shrink=shrink, columns=list(X.columns),
                oof_r2=float(oof_r2), n_points=int(len(X)),
                n_wells=int(len(set(groups))))


def predict_well(bundle: dict, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (tail_indices, predicted TVT) for one well."""
    from .features import point_frame

    X, idx, _ = point_frame(w, stride=1)
    X = X[bundle["columns"]]
    t_last = w["tvt_prefix"][w["known"]][-1]
    delta = bundle["shrink"] * bundle["model"].predict(X)
    return idx, t_last + delta


def per_well_rmse(pred: np.ndarray, y: np.ndarray, wells: np.ndarray) -> pd.Series:
    return (pd.DataFrame({"w": wells, "e": (pred - y) ** 2})
            .groupby("w")["e"].mean().pow(0.5).sort_index())


def save(bundle: dict, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)


def load(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
