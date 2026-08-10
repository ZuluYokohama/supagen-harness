"""Command line entry points.

    python -m geosteern.cli evaluate --data-dir <dir>
    python -m geosteern.cli train    --data-dir <dir> --out model.pkl
    python -m geosteern.cli predict  --data-dir <dir> --model model.pkl --out submission.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .data import list_wells, load_well, split_wells, well_id
from .features import build_dataset
from .model import (TRAIN_STRIDE, fit, load, per_well_rmse, predict_well, save)

DEFAULT_DATA = os.environ.get("GEOSTEERN_DATA", r"C:/PRIMEdEV-1/GeoSteerN-Codex")


def _report(pred, y, wells, label):
    model = per_well_rmse(pred, y, wells).to_numpy()
    hold = per_well_rmse(np.zeros_like(y), y, wells).to_numpy()
    frame = pd.DataFrame({"w": wells, "y": y})
    orc = per_well_rmse(frame.groupby("w")["y"].transform("mean").to_numpy(),
                        y, wells).to_numpy()
    print(f"\n=== {label}  ({len(model)} wells) ===")
    print(f"  {'method':<22} {'median':>8} {'mean':>8} {'p90':>8}")
    for name, v in [("policy model", model), ("hold last TVT", hold),
                    ("oracle constant", orc)]:
        print(f"  {name:<22} {np.median(v):8.2f} {v.mean():8.2f} {np.percentile(v, 90):8.2f}")
    print(f"  beats hold on            {100 * np.mean(model < hold):5.1f}% of wells")
    print(f"  beats oracle constant on {100 * np.mean(model < orc):5.1f}% of wells")

    rng = np.random.default_rng(0)
    n = len(model)
    d = np.array([hold[k].mean() - model[k].mean()
                  for k in (rng.integers(0, n, n) for _ in range(4000))])
    print(f"  mean improvement vs hold {hold.mean() - model.mean():+.2f} ft "
          f"95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")
    return model


def cmd_evaluate(a):
    files = list_wells(a.data_dir, "train")
    dev, hold = split_wells(files)
    print(f"train wells {len(files)}   dev {len(dev)}   holdout {len(hold)}")
    Xd, yd, wd = build_dataset(dev, load_well, TRAIN_STRIDE)
    Xh, yh, wh = build_dataset(hold, load_well, 4)
    print(f"dev points {len(Xd):,}   holdout points {len(Xh):,}")
    bundle = fit(Xd, yd, wd)
    print(f"shrink {bundle['shrink']:.2f}   OOF point R2 {bundle['oof_r2']:+.4f}")
    pred = bundle["shrink"] * bundle["model"].predict(Xh[bundle["columns"]])
    _report(pred, yh, wh, "HOLDOUT (never trained on)")


def cmd_train(a):
    files = list_wells(a.data_dir, "train")
    X, y, w = build_dataset(files, load_well, TRAIN_STRIDE)
    print(f"training on {len(set(w))} wells / {len(X):,} points")
    bundle = fit(X, y, w)
    save(bundle, a.out)
    print(f"shrink {bundle['shrink']:.2f}   OOF point R2 {bundle['oof_r2']:+.4f}")
    print(f"wrote {a.out}")


def cmd_predict(a):
    bundle = load(a.model)
    preds = {}
    for f in list_wells(a.data_dir, "test"):
        w = load_well(f)
        if w is None:
            print(f"  SKIP {f} (unusable)")
            continue
        idx, tvt = predict_well(bundle, w)
        preds[well_id(f)] = dict(zip(idx.tolist(), tvt.tolist()))
        print(f"  {well_id(f)}: {len(idx)} points, "
              f"TVT {tvt.min():.1f}..{tvt.max():.1f}")

    sub_path = os.path.join(a.data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    wells = sub["id"].str.rsplit("_", n=1).str[0]
    rows = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
    out, missing = [], 0
    for wid, r in zip(wells, rows):
        v = preds.get(wid, {}).get(r)
        if v is None:
            missing += 1
            v = 0.0
        out.append(v)
    sub["tvt"] = out
    sub.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(sub):,} rows, {missing} unmatched)")


def main():
    p = argparse.ArgumentParser(prog="geosteern.cli")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    t = sub.add_parser("train"); t.add_argument("--out", default="tvt_model.pkl")
    t.set_defaults(fn=cmd_train)
    q = sub.add_parser("predict")
    q.add_argument("--model", default="tvt_model.pkl")
    q.add_argument("--out", default="submission.csv")
    q.set_defaults(fn=cmd_predict)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
