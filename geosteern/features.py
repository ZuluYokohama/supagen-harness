"""Features for the TVT steering-policy model.

The model predicts TVT(i) - t_last on the unlogged tail, where t_last is the
last known TVT. Everything below is computable at inference time: the known
prefix, plus geometry (MD/X/Y/Z) and GR which are measured over the WHOLE well
including the prediction horizon. Z in particular is a genuine known-future
covariate -- TVT = S - Z, so the model can learn how much of the known vertical
motion the structural surface absorbs.

Neighbour-well and explicit control-loop features were tested and ablated: both
were redundant with these and did not improve holdout error.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PREFIX_WINDOWS = (100, 300, 1000, 3000)


def well_features(w: dict) -> dict:
    """Well-level context: steering history, known-future geometry, typewell."""
    df, tw = w["df"], w["tw"]
    known, tail = w["known"], w["tail"]
    MD = df["MD"].to_numpy()
    Z = df["Z"].to_numpy()
    GR = df["GR"].to_numpy()
    T = w["tvt_prefix"]

    tk = T[known]
    mdk = MD[known]
    zk = Z[known]
    t_last = tk[-1]

    f = {"t_last": t_last, "n_known": float(known.sum()), "n_tail": float(tail.sum())}

    # steering history at several scales: deviation from recent mean (reversion /
    # momentum), local spread, and trend
    for n in PREFIX_WINDOWS:
        seg = tk[-n:] if len(tk) >= n else tk
        f[f"dev_mean_{n}"] = t_last - seg.mean()
        f[f"std_{n}"] = seg.std()
        f[f"rng_{n}"] = seg.max() - seg.min()
        m = mdk[-len(seg):]
        f[f"slope_{n}"] = np.polyfit(m - m[0], seg, 1)[0] if len(seg) > 10 else 0.0
    f["dev_min"] = t_last - tk.min()
    f["dev_max"] = t_last - tk.max()
    f["prefix_mid"] = t_last - 0.5 * (tk.min() + tk.max())

    # restrict to the steered lateral (drop the build section, where |dZ/dMD| is large)
    dz = np.gradient(zk, mdk)
    lat = np.abs(dz) < 0.15
    if lat.sum() > 50:
        tl = tk[lat]
        f["lat_dev_mean"] = t_last - tl.mean()
        f["lat_std"] = tl.std()
        f["lat_frac"] = lat.mean()
    else:
        f["lat_dev_mean"] = f["lat_std"] = f["lat_frac"] = 0.0

    # known-future geometry over the prediction horizon
    zt, mdt = Z[tail], MD[tail]
    f["tail_md_len"] = mdt[-1] - mdt[0]
    f["z_drift_tail"] = zt.mean() - zk[-1]
    f["z_end_drift"] = zt[-1] - zk[-1]
    f["z_slope_tail"] = np.polyfit(mdt - mdt[0], zt, 1)[0]
    f["z_std_tail"] = zt.std()
    f["z_slope_pref"] = (np.polyfit(mdk[-1000:] - mdk[-1000:][0], zk[-1000:], 1)[0]
                         if len(zk) > 1000 else 0.0)
    sk = tk + zk                                   # structural surface on the prefix
    f["s_slope_pref"] = (np.polyfit(mdk[-1500:] - mdk[-1500:][0], sk[-1500:], 1)[0]
                         if len(sk) > 1500 else 0.0)

    gk, gt = GR[known], GR[tail]
    f["gr_pref_mean"] = float(np.nanmean(gk)) if np.isfinite(gk).any() else 0.0
    f["gr_tail_mean"] = float(np.nanmean(gt)) if np.isfinite(gt).any() else 0.0
    f["gr_shift"] = f["gr_tail_mean"] - f["gr_pref_mean"]
    f["gr_tail_std"] = float(np.nanstd(gt)) if np.isfinite(gt).any() else 0.0

    twT = tw["TVT"].to_numpy()
    f["tw_lo"] = t_last - twT.min()
    f["tw_hi"] = twT.max() - t_last
    f["tw_span"] = twT.max() - twT.min()
    f["tw_pos"] = (t_last - twT.min()) / max(twT.max() - twT.min(), 1e-6)
    return f


def point_frame(w: dict, stride: int = 1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-point design matrix over the tail.

    Returns (X, tail_indices, y) where y is None when the well has no labels.
    """
    wf = well_features(w)
    if not all(np.isfinite(v) for v in wf.values()):
        raise ValueError(f"non-finite well feature for {w['well']}")

    df = w["df"]
    MD = df["MD"].to_numpy()
    Z = df["Z"].to_numpy()
    GR = df["GR"].to_numpy()
    known, tail = w["known"], w["tail"]

    t_last = wf["t_last"]
    z_last = Z[known][-1]
    idx = np.where(tail)[0][::stride]
    md0 = MD[tail][0]
    tail_len = max(MD[tail][-1] - md0, 1.0)

    dz = Z[idx] - z_last
    dmd = MD[idx] - md0
    g = GR[idx]
    g = np.where(np.isfinite(g), g, wf["gr_pref_mean"])

    blk = {
        "dz": dz,
        "dmd": dmd,
        "pos": dmd / tail_len,
        "gr": g,
        "gr_rel": g - wf["gr_pref_mean"],
        "dz_rate": dz / np.maximum(dmd, 1.0),
    }
    for k, v in wf.items():
        blk[k] = np.full(idx.size, float(v))

    X = pd.DataFrame(blk).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = (w["truth"][idx] - t_last) if w["truth"] is not None else None
    return X, idx, y


def build_dataset(files, loader, stride: int, report: bool = True):
    """Stack per-point frames across wells.

    Excluded wells are recorded with a reason and reported, rather than dropped
    silently -- a well vanishing from training and from the evaluation
    denominator without explanation is how a metric quietly stops meaning what
    it says. The prediction path (cli.predict) never reaches here and fails
    loudly instead, so no well is skipped without notice anywhere.
    """
    Xs, ys, ws = [], [], []
    skipped: dict[str, list[str]] = {}

    def note(reason: str, name: str) -> None:
        skipped.setdefault(reason, []).append(name)

    for f in files:
        name = os.path.basename(f)
        w = loader(f)
        if w is None:
            note("unloadable (schema/typewell/contiguity/too-short)", name)
            continue
        if w["truth"] is None:
            note("no TVT labels", name)
            continue
        try:
            X, _, y = point_frame(w, stride)
        except Exception as exc:
            note(f"feature error: {type(exc).__name__}", name)
            continue
        if y is None or not np.isfinite(y).all():
            note("non-finite target", name)
            continue
        Xs.append(X)
        ys.append(y)
        ws.append(np.full(len(X), w["well"]))

    if report and skipped:
        total = sum(len(v) for v in skipped.values())
        print(f"  excluded {total} of {len(files)} wells:")
        for reason, names in sorted(skipped.items()):
            shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
            print(f"    {len(names):>4}  {reason}  [{shown}]")
    if not Xs:
        raise RuntimeError("no usable wells")
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)
