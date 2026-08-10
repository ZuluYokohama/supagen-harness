"""Data loading for the TVT steering-policy model.

Works identically on TRAIN and TEST wells. TEST files carry only
MD, X, Y, Z, GR, TVT_input -- no formation tops and no TVT column -- so the
prefix TVT is always read from TVT_input (verified identical to TVT wherever
TVT_input is present). The TVT column is used ONLY as a training target.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

REQUIRED = ("MD", "X", "Y", "Z", "GR", "TVT_input")


def well_id(path: str) -> str:
    return os.path.basename(path).split("__")[0]


def find_typewell(horizontal_csv: str) -> str | None:
    """Locate a well's typewell. Filenames are inconsistent across the corpus
    ('typewell', 'typewelll', with/without _TRAIN/_TEST), so match on well id."""
    d = os.path.dirname(horizontal_csv)
    hits = [p for p in glob.glob(os.path.join(d, f"{well_id(horizontal_csv)}__typewel*"))
            if "horizontal" not in os.path.basename(p)]
    return hits[0] if hits else None


def load_well(horizontal_csv: str) -> dict | None:
    """Return a well record, or None if it is unusable.

    keys: well, df, tw, known, tail, tvt_prefix, truth (None at inference)
    """
    df = pd.read_csv(horizontal_csv)
    if not set(REQUIRED) <= set(df.columns):
        return None
    tw_path = find_typewell(horizontal_csv)
    if tw_path is None:
        return None
    tw = pd.read_csv(tw_path)
    if not {"TVT", "GR"} <= set(tw.columns):
        return None
    tw = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    if len(tw) < 50:
        return None

    # pandas>=3 returns read-only arrays from .to_numpy(); copy before mutating
    known = df["TVT_input"].notna().to_numpy().copy()
    tail = ~known
    geom_ok = df[["MD", "X", "Y", "Z"]].notna().all(axis=1).to_numpy()
    known &= geom_ok
    tail &= geom_ok
    if known.sum() < 50 or tail.sum() < 5:
        return None

    truth = df["TVT"].to_numpy() if "TVT" in df.columns else None
    return dict(well=well_id(horizontal_csv), df=df, tw=tw, known=known, tail=tail,
                tvt_prefix=df["TVT_input"].to_numpy(), truth=truth)


def list_wells(data_dir: str, subset: str) -> list[str]:
    pat = os.path.join(data_dir, subset, "*horizontal_well*.csv")
    return sorted(glob.glob(pat))


def split_wells(files: list[str], frac: float = 0.35) -> tuple[list[str], list[str]]:
    """Deterministic hash split -> (dev, holdout). Stable across runs and
    independent of file ordering, so evaluation numbers are reproducible."""
    dev, hold = [], []
    for f in files:
        (hold if (int(well_id(f)[:6], 16) % 100) < frac * 100 else dev).append(f)
    return dev, hold
