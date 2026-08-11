"""Re-test survey geometry with correctly-constructed features.

The README records survey geometry as refuted by 773-well CV (better on 49.9%
of wells, Wilcoxon p=0.71). Those features were differenced at 1 ft, where the
direction signal is 100% coordinate-rounding artifact (57.15 measured against a
57.30 prediction). The CV was valid; the inputs were not.

This rebuilds them through research.survey_primitives at both ends of the
measured operating band and reruns the same protocol.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

sys.path.insert(0, ".")
from geosteern.data import list_wells, load_well, well_id            # noqa: E402
from geosteern.features import point_frame                            # noqa: E402
from geosteern.model import PARAMS, TRAIN_STRIDE, calibrate_shrink, per_well_rmse  # noqa: E402
from research.survey_primitives import (                              # noqa: E402
    azimuth_error_multiplier, derive_section_azimuth, unwrap_azimuth, windowed_survey,
)

DATA = r"C:/PRIMEdEV-1/GeoSteerN-Codex"
MAG_TO_GRID = 8.7414
EVAL_STRIDE = 4
FOLDS = 4


def geometry_block(w: dict, idx: np.ndarray, window: int) -> pd.DataFrame | None:
    """Geometry features on the tail. Uses only MD/X/Y/Z -- inference-safe."""
    df = w["df"]
    md = df["MD"].to_numpy(float)
    try:
        s = windowed_survey(md, df["X"].to_numpy(float), df["Y"].to_numpy(float),
                            df["Z"].to_numpy(float), window_ft=window)
    except Exception:
        return None
    if len(s.md) < 8:
        return None

    azi_u = unwrap_azimuth(s.azimuth_deg)
    try:
        sect = derive_section_azimuth(md, df["X"].to_numpy(float),
                                      df["Y"].to_numpy(float),
                                      df["Z"].to_numpy(float), survey=s)
    except Exception:
        return None

    target = md[idx]
    inc = np.interp(target, s.md, s.inclination_deg)
    azi = np.interp(target, s.md, azi_u)
    dls = np.interp(target, s.md, s.dls_deg_per_100ft)

    # rates along the hole -- the steering actions themselves
    d_inc = np.gradient(s.inclination_deg, s.md) * 100.0
    d_azi = np.gradient(azi_u, s.md) * 100.0
    build = np.interp(target, s.md, d_inc)
    turn = np.interp(target, s.md, d_azi)

    # state at the anchor (last labelled point), and drift from it
    anchor_md = md[np.where(w["known"])[0][-1]]
    inc0 = float(np.interp(anchor_md, s.md, s.inclination_deg))
    azi0 = float(np.interp(anchor_md, s.md, azi_u))

    azi_sensitivity = azimuth_error_multiplier(np.interp(target, s.md, s.azimuth_deg), inc, MAG_TO_GRID)
    return pd.DataFrame({
        "g_inc": inc,
        "g_inc_dev": inc - 90.0,          # toe-up positive, toe-down negative
        "g_inc_drift": inc - inc0,
        "g_azi_rel": ((azi - sect + 90.0) % 180.0) - 90.0,
        "g_azi_drift": azi - azi0,
        "g_dls": dls,
        "g_build": build,
        "g_turn": turn,
        "g_inc_anchor": np.full(len(idx), inc0),
        "g_azi_sensitivity": azi_sensitivity,
    })


def build(files, window: int | None):
    Xs, ys, ws = [], [], []
    for f in files:
        w = load_well(f)
        if w is None or w.get("truth") is None:
            continue
        try:
            X, idx, y = point_frame(w, stride=TRAIN_STRIDE)
        except Exception:
            continue
        if y is None or len(X) == 0:
            continue
        if window is not None:
            g = geometry_block(w, idx, window)
            if g is None:
                continue
            X = pd.concat([X.reset_index(drop=True), g.reset_index(drop=True)], axis=1)
        Xs.append(X); ys.append(y); ws.append(np.full(len(X), well_id(f)))
    if not Xs:
        return None, None, None
    return (pd.concat(Xs, ignore_index=True).replace([np.inf, -np.inf], 0.0).fillna(0.0),
            np.concatenate(ys), np.concatenate(ws))


def cross_val(X, y, wells):
    """4-fold grouped CV; shrinkage calibrated inside each fold."""
    pred = np.zeros_like(y)
    for tr, va in GroupKFold(FOLDS).split(X, y, groups=wells):
        inner = np.zeros(len(tr), dtype=float)
        gtr = wells[tr]
        for itr, iva in GroupKFold(FOLDS).split(X.iloc[tr], y[tr], groups=gtr):
            m = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[tr].iloc[itr], y[tr][itr])
            inner[iva] = m.predict(X.iloc[tr].iloc[iva])
        shrink = calibrate_shrink(inner, y[tr])
        model = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[tr], y[tr])
        pred[va] = shrink * model.predict(X.iloc[va])
    return pred


def report(name, files):
    print(f"\n{'='*74}\n{name}  ({len(files)} wells)\n{'='*74}")
    base_X, base_y, base_w = build(files, None)
    if base_X is None:
        print("  no usable wells"); return
    base_pred = cross_val(base_X, base_y, base_w)
    base_rmse = per_well_rmse(base_pred, base_y, base_w)
    hold_rmse = per_well_rmse(np.zeros_like(base_y), base_y, base_w)
    print(f"  baseline      median {base_rmse.median():6.3f}  mean {base_rmse.mean():6.3f} "
          f"| hold {hold_rmse.median():6.3f}  | beats hold {100*(base_rmse<hold_rmse).mean():5.1f}%")

    for window in (96, 128):
        gX, gy, gw = build(files, window)
        if gX is None:
            print(f"  +geom@{window}: no usable wells"); continue
        gp = cross_val(gX, gy, gw)
        g_rmse = per_well_rmse(gp, gy, gw)
        common = base_rmse.index.intersection(g_rmse.index)
        a, b = base_rmse.loc[common], g_rmse.loc[common]
        delta = a.mean() - b.mean()
        med = a.median() - b.median()
        win = 100.0 * (b < a).mean()
        try:
            p = wilcoxon(a.values, b.values).pvalue
        except Exception:
            p = float("nan")
        print(f"  +geom@{window:3d}  median {b.median():6.3f}  mean {b.mean():6.3f} "
              f"| gain med {med:+.3f} mean {delta:+.3f} | better on {win:5.1f}% | Wilcoxon p={p:.3f}")


if __name__ == "__main__":
    files = list_wells(DATA, "train")
    # trust weight per well -> the N/S subset where azimuth is reliable
    scored = []
    for f in files:
        w = load_well(f)
        if w is None or w.get("truth") is None:
            continue
        d = w["df"]
        try:
            s = windowed_survey(d["MD"].to_numpy(float), d["X"].to_numpy(float),
                                d["Y"].to_numpy(float), d["Z"].to_numpy(float))
        except Exception:
            continue
        lat = s.inclination_deg > 85
        if lat.sum() < 50:
            continue
        t = float(np.median(azimuth_error_multiplier(
            s.azimuth_deg[lat], s.inclination_deg[lat], MAG_TO_GRID)))
        scored.append((f, t))
    scored.sort(key=lambda r: r[1])
    ns = [f for f, t in scored if t < 0.34]
    print(f"corpus {len(scored)} usable wells; N/S subset |sin|<0.34: {len(ns)}")
    report("N/S SUBSET (azimuth trustworthy)", ns)
    report("CORPUS-WIDE", [f for f, _ in scored])
