"""Leakage-safe measurement gate for interval-valued gamma evidence.

This is deliberately research code, not a production prediction path.  It asks
one narrow question: after an inference-safe baseline proposes TVT, can gamma
ray evidence aggregated by *stratigraphic cell* estimate a useful correction?

The experiment gives every occupied TVT cell equal weight.  A well therefore
cannot dominate the match merely because it dwelled in one bed for hundreds of
measured-depth samples.  Typewell and known-prefix atlases carry empirical GR
quartiles; formation surfaces, suffix TVT, PNGs, and test/train ID overlap are
forbidden from feature construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geosteern.data import find_typewell, list_wells, load_well, well_id
from geosteern.features import build_dataset, point_frame
from geosteern.model import PARAMS, calibrate_shrink
from research.ordered_transport import ordered_reversible_interval_transport


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TEST_OVERLAP = {"000d7d20", "00bbac68", "00e12e8b"}
FORBIDDEN_FEATURE_COLUMNS = {
    "TVT", "Geology", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"
}
OFFSETS = np.arange(-30.0, 30.0001, 0.5)
BIN_WIDTHS = (1.0, 2.0, 4.0)


@dataclass
class PredictionRecord:
    well: str
    path: str
    idx: np.ndarray
    prediction: np.ndarray
    truth: np.ndarray

    @property
    def oracle_shift(self) -> float:
        """Best constant additive correction for squared error."""
        return float(np.mean(self.truth - self.prediction))


@dataclass
class Atlas:
    centers: np.ndarray
    quartiles: np.ndarray
    counts: np.ndarray

    @property
    def empty(self) -> bool:
        return self.centers.size == 0


def _research_params(confirm: bool) -> tuple[dict, int, int]:
    """Return model parameters, training stride, and evaluation stride.

    Screen mode is intentionally cheaper.  Confirm mode reproduces the inherited
    baseline settings before a candidate may be treated as more than a screen.
    """
    params = dict(PARAMS)
    params["n_jobs"] = 4
    if confirm:
        return params, 8, 1
    params.update(n_estimators=300, learning_rate=0.05, num_leaves=47)
    return params, 16, 6


def _typewell_hash(path: str) -> str:
    """Hash the visible reference log so exact copies never cross a split."""
    tw_path = find_typewell(path)
    if tw_path is None:
        raise RuntimeError(f"missing typewell for {path}")
    tw = pd.read_csv(tw_path, usecols=["TVT", "GR"])
    values = pd.util.hash_pandas_object(tw, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _filtered_files(data_dir: str) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    files = [f for f in list_wells(data_dir, "train")
             if well_id(f) not in EXCLUDED_TEST_OVERLAP]
    hashes = {well_id(f): _typewell_hash(f) for f in files}
    # Split on the reference-log hash, not well ID, so an exact copied typewell
    # is entirely on one side of the outer boundary.
    hold_hashes = {h for h in set(hashes.values()) if int(h[:8], 16) % 100 < 35}
    dev = [f for f in files if hashes[well_id(f)] not in hold_hashes]
    hold = [f for f in files if hashes[well_id(f)] in hold_hashes]
    return files, dev, hold, hashes


def _assert_inference_safe_feature_surface() -> None:
    """Static audit of the inherited point-frame source columns.

    The inherited loader retains labels for scoring, so the stronger invariant is
    that feature code references none of the answer-derived column names.  The
    experiment below only reads MD/Z/GR/TVT_input and typewell TVT/GR.
    """
    import inspect
    import geosteern.features as inherited_features

    source = inspect.getsource(inherited_features)
    # Typewell TVT is an inference input; horizontal ``df[\"TVT\"]`` is not.
    # Restrict the textual audit to the horizontal dataframe access pattern.
    hits = sorted(c for c in FORBIDDEN_FEATURE_COLUMNS
                  if f'df["{c}"]' in source or f"df['{c}']" in source)
    if hits:
        raise RuntimeError(f"forbidden columns referenced by feature code: {hits}")


def _runtime_mutation_audit(path: str) -> None:
    """Prove model features ignore answer-derived horizontal columns at runtime."""
    w = load_well(path)
    if w is None:
        raise RuntimeError(f"could not load mutation-audit well {path}")
    before, _, _ = point_frame(w, stride=37)
    for column in FORBIDDEN_FEATURE_COLUMNS:
        if column in w["df"].columns:
            w["df"][column] = np.linspace(-1e6, 1e6, len(w["df"]))
    if w["truth"] is not None:
        w["truth"] = np.linspace(1e6, -1e6, len(w["df"]))
    after, _, _ = point_frame(w, stride=37)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def _make_atlas(tvt: np.ndarray, gr: np.ndarray, width: float) -> Atlas:
    tvt = np.asarray(tvt, dtype=float)
    gr = np.asarray(gr, dtype=float)
    ok = np.isfinite(tvt) & np.isfinite(gr)
    if ok.sum() < 3:
        return Atlas(np.array([]), np.empty((0, 3)), np.array([]))
    bins = np.rint(tvt[ok] / width).astype(np.int64)
    frame = pd.DataFrame({"bin": bins, "gr": gr[ok]})
    grouped = frame.groupby("bin", sort=True)["gr"]
    q = grouped.quantile([0.25, 0.50, 0.75]).unstack()
    counts = grouped.size().reindex(q.index).to_numpy(dtype=float)
    return Atlas(
        centers=q.index.to_numpy(dtype=float) * width,
        quartiles=q.to_numpy(dtype=float),
        counts=counts,
    )


def _interp_atlas(atlas: Atlas, positions: np.ndarray, width: float) -> np.ndarray:
    """Interpolate without bridging gaps larger than two atlas cells."""
    positions = np.asarray(positions, dtype=float)
    out = np.full((len(positions), 3), np.nan)
    if atlas.empty:
        return out
    for j in range(3):
        out[:, j] = np.interp(
            positions, atlas.centers, atlas.quartiles[:, j],
            left=np.nan, right=np.nan,
        )
    insert = np.searchsorted(atlas.centers, positions)
    left = np.clip(insert - 1, 0, len(atlas.centers) - 1)
    right = np.clip(insert, 0, len(atlas.centers) - 1)
    nearest = np.minimum(
        np.abs(positions - atlas.centers[left]),
        np.abs(positions - atlas.centers[right]),
    )
    out[nearest > 2.01 * width] = np.nan
    internal = (insert > 0) & (insert < len(atlas.centers))
    bracket_gap = atlas.centers[right] - atlas.centers[left]
    out[internal & (bracket_gap > 2.01 * width)] = np.nan
    return out


def _robust_scale(values: np.ndarray, floor: float = 5.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return floor
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float(max(q75 - q25, floor))


def _calibrated_typewell_atlas(w: dict, width: float) -> tuple[Atlas, dict]:
    """Map typewell GR onto horizontal-tool scale using the known prefix only."""
    df, tw = w["df"], w["tw"]
    known = w["known"]
    prefix = _make_atlas(
        w["tvt_prefix"][known], df["GR"].to_numpy()[known], width
    )
    ref = _make_atlas(tw["TVT"].to_numpy(), tw["GR"].to_numpy(), width)
    paired = _interp_atlas(ref, prefix.centers, width)
    ok = np.isfinite(paired[:, 1]) if len(paired) else np.array([], dtype=bool)
    if ok.sum() >= 6:
        x = paired[ok, 1]
        y = prefix.quartiles[ok, 1]
        x_scale = _robust_scale(x, 1.0)
        y_scale = _robust_scale(y, 1.0)
        gain = float(np.clip(y_scale / x_scale, 0.25, 4.0))
        bias = float(np.median(y) - gain * np.median(x))
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    else:
        h = df["GR"].to_numpy()[known]
        h = h[np.isfinite(h)]
        x = ref.quartiles[:, 1] if not ref.empty else np.array([])
        gain = float(np.clip(_robust_scale(h, 1.0) / _robust_scale(x, 1.0), 0.25, 4.0))
        bias = float(np.nanmedian(h) - gain * np.nanmedian(x)) if h.size and x.size else 0.0
        corr = 0.0
    calibrated = Atlas(ref.centers, ref.quartiles * gain + bias, ref.counts)
    return calibrated, {
        "gain": gain,
        "bias": bias,
        "prefix_cell_corr": corr,
        "prefix_calibration_cells": int(ok.sum()),
    }


def _landscape(query: Atlas, reference: Atlas, width: float,
               scale: float, min_cells: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Return interval-measure mismatch and overlap for every candidate shift."""
    costs = np.full(len(OFFSETS), np.inf)
    supports = np.zeros(len(OFFSETS), dtype=int)
    if query.empty or reference.empty:
        return costs, supports

    for j, offset in enumerate(OFFSETS):
        rq = _interp_atlas(reference, query.centers + offset, width)
        ok = np.isfinite(rq).all(axis=1) & np.isfinite(query.quartiles).all(axis=1)
        if ok.sum() < min_cells:
            continue
        supports[j] = int(ok.sum())
        hq = query.quartiles[ok]
        tq = rq[ok]
        location = float(np.median(np.mean(np.abs(hq - tq), axis=1)) / scale)

        centers = query.centers[ok]
        hm = gaussian_filter1d(hq[:, 1], sigma=1.0, mode="nearest")
        tm = gaussian_filter1d(tq[:, 1], sigma=1.0, mode="nearest")
        if len(hm) >= 6 and np.std(hm) > 1e-8 and np.std(tm) > 1e-8:
            correlation = float(np.clip(np.corrcoef(hm, tm)[0, 1], -1.0, 1.0))
            corr_cost = 1.0 - correlation
            gaps = np.diff(centers)
            good_gap = gaps <= 2.01 * width
            if good_gap.sum() >= 3:
                dh = np.diff(hm)[good_gap] / gaps[good_gap]
                dt = np.diff(tm)[good_gap] / gaps[good_gap]
                gradient = float(np.median(np.abs(dh - dt)) / max(scale / width, 1e-6))
            else:
                gradient = 1.0
        else:
            corr_cost, gradient = 1.0, 1.0
        costs[j] = location + 0.30 * corr_cost + 0.20 * gradient
    return costs, supports


def _landscape_features(costs: np.ndarray, supports: np.ndarray,
                        prefix: str) -> dict[str, float]:
    finite = np.isfinite(costs)
    if finite.sum() < 5:
        return {
            f"{prefix}_available": 0.0,
            f"{prefix}_delta_min": np.nan,
            f"{prefix}_delta_soft": np.nan,
            f"{prefix}_sharpness": 0.0,
            f"{prefix}_near_width": 60.0,
            f"{prefix}_support": 0.0,
            f"{prefix}_edge": 1.0,
            f"{prefix}_cost0": np.nan,
        }
    vals = costs[finite]
    offs = OFFSETS[finite]
    supp = supports[finite]
    best = int(np.argmin(vals))
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    temperature = max(mad, 0.05)
    weights = np.exp(-np.clip((vals - vals[best]) / temperature, 0, 50))
    delta_soft = float(np.sum(weights * offs) / np.sum(weights))
    threshold = vals[best] + max(0.25 * mad, 0.03)
    near = offs[vals <= threshold]
    zero_idx = int(np.argmin(np.abs(offs)))
    return {
        f"{prefix}_available": 1.0,
        f"{prefix}_delta_min": float(offs[best]),
        f"{prefix}_delta_soft": delta_soft,
        f"{prefix}_sharpness": float((median - vals[best]) / max(mad, 1e-6)),
        f"{prefix}_near_width": float(near.max() - near.min()) if len(near) else 60.0,
        f"{prefix}_support": float(supp[best]),
        f"{prefix}_edge": float(abs(offs[best]) >= OFFSETS.max() - 0.01),
        f"{prefix}_cost0": float(vals[zero_idx]),
    }


def interval_features(record: PredictionRecord) -> dict[str, float]:
    """Build one inference-safe interval-evidence row for a predicted well."""
    w = load_well(record.path)
    if w is None:
        raise RuntimeError(f"could not reload {record.path}")
    df = w["df"]
    gr = df["GR"].to_numpy()
    known = w["known"]
    out: dict[str, float] = {
        "gr_valid_fraction": float(np.isfinite(gr[record.idx]).mean()),
        "base_tvt_range": float(np.ptp(record.prediction)),
        "n_evaluation_rows": float(len(record.idx)),
    }
    tw_deltas, prefix_deltas = [], []

    for width in BIN_WIDTHS:
        tag = str(width).replace(".", "p")
        query = _make_atlas(record.prediction, gr[record.idx], width)
        prefix = _make_atlas(w["tvt_prefix"][known], gr[known], width)
        typewell, calibration = _calibrated_typewell_atlas(w, width)
        scale = _robust_scale(prefix.quartiles[:, 1] if not prefix.empty
                              else query.quartiles[:, 1])

        tw_cost, tw_support = _landscape(query, typewell, width, scale)
        pref_cost, pref_support = _landscape(query, prefix, width, scale)
        twf = _landscape_features(tw_cost, tw_support, f"tw_{tag}")
        pff = _landscape_features(pref_cost, pref_support, f"prefix_{tag}")
        out.update(twf)
        out.update(pff)
        out[f"tw_{tag}_cal_gain"] = calibration["gain"]
        out[f"tw_{tag}_prefix_corr"] = calibration["prefix_cell_corr"]
        out[f"tw_{tag}_cal_cells"] = float(calibration["prefix_calibration_cells"])
        out[f"query_{tag}_cells"] = float(len(query.centers))
        if np.isfinite(twf[f"tw_{tag}_delta_soft"]):
            tw_deltas.append(twf[f"tw_{tag}_delta_soft"])
        if np.isfinite(pff[f"prefix_{tag}_delta_soft"]):
            prefix_deltas.append(pff[f"prefix_{tag}_delta_soft"])

    out["tw_consensus_delta"] = float(np.median(tw_deltas)) if tw_deltas else np.nan
    out["prefix_consensus_delta"] = (
        float(np.median(prefix_deltas)) if prefix_deltas else np.nan
    )
    out["atlas_disagreement"] = (
        abs(out["tw_consensus_delta"] - out["prefix_consensus_delta"])
        if np.isfinite(out["tw_consensus_delta"])
        and np.isfinite(out["prefix_consensus_delta"])
        else np.nan
    )
    return out


def _fit_base_predictions(dev_files: list[str], hold_files: list[str],
                          params: dict, train_stride: int,
                          eval_stride: int,
                          typewell_hashes: dict[str, str]) -> tuple[list[PredictionRecord],
                                                     list[PredictionRecord], dict]:
    print("building inherited baseline matrix...", flush=True)
    X, y, groups = build_dataset(dev_files, load_well, train_stride)
    files_by_id = {well_id(f): f for f in dev_files}
    oof = np.zeros_like(y)
    raw_records: dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    splitter = GroupKFold(4)
    cv_groups = np.array([typewell_hashes[wid] for wid in groups])
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups=cv_groups), 1):
        print(f"  base OOF fold {fold}/4", flush=True)
        model = lgb.LGBMRegressor(**params).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict(X.iloc[va])
        for wid in sorted(set(groups[va])):
            path = files_by_id[wid]
            w = load_well(path)
            if w is None:
                continue
            frame, idx, delta_truth = point_frame(w, stride=eval_stride)
            raw = model.predict(frame[X.columns])
            t_last = float(w["tvt_prefix"][w["known"]][-1])
            raw_records[wid] = (path, idx, raw, delta_truth, t_last)

    shrink = calibrate_shrink(oof, y)
    dev_records = [
        PredictionRecord(
            well=wid, path=v[0], idx=v[1],
            prediction=v[4] + shrink * v[2], truth=v[4] + v[3],
        )
        for wid, v in sorted(raw_records.items())
    ]

    print("  fitting final base model for held-out wells", flush=True)
    final_model = lgb.LGBMRegressor(**params).fit(X, y)
    hold_records = []
    for path in hold_files:
        w = load_well(path)
        if w is None:
            continue
        frame, idx, delta_truth = point_frame(w, stride=eval_stride)
        raw = final_model.predict(frame[X.columns])
        t_last = float(w["tvt_prefix"][w["known"]][-1])
        hold_records.append(PredictionRecord(
            well=w["well"], path=path, idx=idx,
            prediction=t_last + shrink * raw, truth=t_last + delta_truth,
        ))
    meta = {
        "shrink": float(shrink),
        "oof_point_r2": float(1.0 - np.sum((shrink * oof - y) ** 2)
                              / np.sum((y - y.mean()) ** 2)),
        "training_points": int(len(X)),
        "training_wells": int(len(set(groups))),
        "training_typewell_groups": int(len(set(cv_groups))),
    }
    return dev_records, hold_records, meta


def _metric_block(records: list[PredictionRecord],
                  corrections: np.ndarray | list[np.ndarray] | None = None) -> dict:
    if corrections is None:
        corrections = np.zeros(len(records))
    well_rmse, counts, squared = [], [], 0.0
    for record, correction in zip(records, corrections, strict=True):
        err = record.prediction + correction - record.truth
        well_rmse.append(float(np.sqrt(np.mean(err ** 2))))
        counts.append(len(err))
        squared += float(np.sum(err ** 2))
    values = np.asarray(well_rmse)
    return {
        "median_well_rmse": float(np.median(values)),
        "mean_well_rmse": float(np.mean(values)),
        "p90_well_rmse": float(np.quantile(values, 0.9)),
        "pooled_row_rmse": float(np.sqrt(squared / np.sum(counts))),
        "n_wells": int(len(records)),
        "n_rows": int(np.sum(counts)),
        "per_well": dict(
            zip((r.well for r in records), values.tolist(), strict=True)
        ),
    }


def _bootstrap_gain(records: list[PredictionRecord],
                    corrections: np.ndarray | list[np.ndarray]) -> dict:
    base_rmse, candidate_rmse = [], []
    base_sse, candidate_sse, counts = [], [], []
    for record, correction in zip(records, corrections, strict=True):
        base_error = record.prediction - record.truth
        candidate_error = base_error + correction
        base_sse.append(float(np.sum(base_error ** 2)))
        candidate_sse.append(float(np.sum(candidate_error ** 2)))
        counts.append(len(base_error))
        base_rmse.append(float(np.sqrt(np.mean(base_error ** 2))))
        candidate_rmse.append(float(np.sqrt(np.mean(candidate_error ** 2))))
    base_sse = np.asarray(base_sse)
    candidate_sse = np.asarray(candidate_sse)
    counts = np.asarray(counts)
    base_rmse = np.asarray(base_rmse)
    candidate_rmse = np.asarray(candidate_rmse)
    rng = np.random.default_rng(20260810)
    draws = np.empty(4000)
    for i in range(len(draws)):
        sample = rng.integers(0, len(records), len(records))
        base = np.sqrt(base_sse[sample].sum() / counts[sample].sum())
        candidate = np.sqrt(candidate_sse[sample].sum() / counts[sample].sum())
        draws[i] = base - candidate
    pooled_gain = (
        np.sqrt(base_sse.sum() / counts.sum())
        - np.sqrt(candidate_sse.sum() / counts.sum())
    )
    return {
        "pooled_rmse_gain_ft": float(pooled_gain),
        "ci95_low_ft": float(np.quantile(draws, 0.025)),
        "ci95_high_ft": float(np.quantile(draws, 0.975)),
        "mean_well_rmse_gain_ft": float(np.mean(base_rmse - candidate_rmse)),
        "win_rate": float(np.mean(candidate_rmse < base_rmse)),
    }


def _fit_interval_correction(dev_features: pd.DataFrame, dev_target: np.ndarray,
                             hold_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, object]:
    """Well-level stacked correction; OOF dev predictions are diagnostic only."""
    columns = [c for c in dev_features.columns
               if c != "well" and ("tw_" in c or "prefix_" in c
                                    or c in {"gr_valid_fraction", "atlas_disagreement"})]
    X = dev_features[columns].replace([np.inf, -np.inf], np.nan)
    Xh = hold_features[columns].replace([np.inf, -np.inf], np.nan)
    oof = np.zeros(len(X))
    for tr, va in KFold(5, shuffle=True, random_state=20260810).split(X):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(), Ridge(alpha=20.0),
        )
        model.fit(X.iloc[tr], dev_target[tr])
        oof[va] = model.predict(X.iloc[va])
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(), Ridge(alpha=20.0),
    )
    model.fit(X, dev_target)
    hold = model.predict(Xh)
    return np.clip(oof, -25.0, 25.0), np.clip(hold, -25.0, 25.0), model


def _scalar_correction(dev_estimate: np.ndarray, dev_target: np.ndarray,
                       hold_estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ok = np.isfinite(dev_estimate)
    if ok.sum() < 20 or np.sum(dev_estimate[ok] ** 2) < 1e-8:
        return np.zeros_like(dev_target), np.zeros_like(hold_estimate), 0.0
    shrink = float(np.clip(
        np.sum(dev_estimate[ok] * dev_target[ok])
        / np.sum(dev_estimate[ok] ** 2), 0.0, 1.5,
    ))
    dev = np.where(np.isfinite(dev_estimate), shrink * dev_estimate, 0.0)
    hold = np.where(np.isfinite(hold_estimate), shrink * hold_estimate, 0.0)
    return np.clip(dev, -25, 25), np.clip(hold, -25, 25), shrink


def _evaluate_candidate(records: list[PredictionRecord],
                        corrections: np.ndarray | list[np.ndarray]) -> tuple[dict, dict]:
    metrics = _metric_block(records, corrections)
    return metrics, _bootstrap_gain(records, corrections)


def _ordered_corrections(
    records: list[PredictionRecord], label: str
) -> tuple[list[np.ndarray], dict[str, dict]]:
    """Run the frozen ordered stay/cross/reverse solver on complete suffixes."""
    corrections: list[np.ndarray] = []
    diagnostics: dict[str, dict] = {}
    for i, record in enumerate(records, 1):
        w = load_well(record.path)
        if w is None:
            raise RuntimeError(f"could not reload {record.path}")
        if len(record.idx) != int(w["tail"].sum()):
            raise RuntimeError("ordered confirmation requires evaluation_stride=1")
        df, tw = w["df"], w["tw"]
        proposal = w["tvt_prefix"].copy()
        proposal[record.idx] = record.prediction

        # The solver requires finite MD to place nodes and a finite proposal
        # over the evaluated tail. (GR may have gaps -- observed_gr_fraction
        # reports them.) A well violating either cannot be confirmed, so record
        # why and leave its prediction uncorrected instead of raising and
        # discarding the whole pass.
        md_values = df["MD"].to_numpy()
        tail_proposal = proposal[record.idx]
        non_finite_md = int((~np.isfinite(md_values)).sum())
        non_finite_tail = int((~np.isfinite(tail_proposal)).sum())
        if non_finite_md or non_finite_tail:
            corrections.append(np.zeros(len(record.idx), dtype=float))
            diagnostics[record.well] = {
                "status": "skipped_non_finite_solver_inputs",
                "non_finite_md_rows": non_finite_md,
                "non_finite_proposal_tail_rows": non_finite_tail,
            }
            print(
                f"  ordered {label}: SKIP {record.well} "
                f"({non_finite_md} non-finite MD, "
                f"{non_finite_tail} non-finite tail proposal)",
                flush=True,
            )
            continue

        result = ordered_reversible_interval_transport(
            horizontal_md=df["MD"].to_numpy(),
            horizontal_gr=df["GR"].to_numpy(),
            typewell_tvt=tw["TVT"].to_numpy(),
            typewell_gr=tw["GR"].to_numpy(),
            known_prefix_tvt=w["tvt_prefix"],
            proposed_tvt_path=proposal,
        )
        corrections.append(result.corrected_tvt[record.idx] - record.prediction)
        diag = result.diagnostics
        diagnostics[record.well] = {
            "status": diag.status,
            "observed_gr_fraction": diag.observed_gr_fraction,
            "viterbi_nodes": diag.viterbi_nodes,
            "states": diag.states,
            "forward_orientation_nodes": diag.forward_orientation_nodes,
            "reverse_orientation_nodes": diag.reverse_orientation_nodes,
            "reversal_count": diag.reversal_count,
            "boundary_state_nodes": diag.boundary_state_nodes,
            "mean_abs_correction_tvt": diag.mean_abs_correction_tvt,
            "max_abs_correction_tvt": diag.max_abs_correction_tvt,
        }
        if i % 25 == 0 or i == len(records):
            print(f"  ordered {label}: {i}/{len(records)} wells", flush=True)
    return corrections, diagnostics


def _calibrate_vector_shrink(records: list[PredictionRecord],
                             corrections: list[np.ndarray]) -> float:
    numerator, denominator = 0.0, 0.0
    for record, correction in zip(records, corrections, strict=True):
        residual = record.truth - record.prediction
        numerator += float(np.sum(correction * residual))
        denominator += float(np.sum(correction ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.5))


def _fit_joint_correction(
    records: list[PredictionRecord],
    first: list[np.ndarray],
    second: list[np.ndarray],
) -> np.ndarray:
    """Fit two global coefficients on OOF development residuals."""
    gram = np.zeros((2, 2), dtype=float)
    rhs = np.zeros(2, dtype=float)
    for record, a, b in zip(records, first, second, strict=True):
        design = np.column_stack((a, b))
        target = record.truth - record.prediction
        gram += design.T @ design
        rhs += design.T @ target
    penalty = 1e-4 * max(float(np.trace(gram)), 1.0)
    coefficients = np.linalg.solve(gram + penalty * np.eye(2), rhs)
    return np.clip(coefficients, -1.0, 2.0)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    _assert_inference_safe_feature_surface()
    params, train_stride, eval_stride = _research_params(args.confirm)
    files, dev_files, hold_files, typewell_hashes = _filtered_files(args.data_dir)
    if not dev_files:
        raise RuntimeError(
            f"development split is empty: no eligible wells under {args.data_dir!r} "
            f"({len(files)} eligible in total). Check --data-dir points at a corpus "
            "containing train/."
        )
    _runtime_mutation_audit(dev_files[0])
    print(
        f"eligible wells {len(files)}  dev {len(dev_files)}  hold {len(hold_files)}  "
        f"excluded overlap {sorted(EXCLUDED_TEST_OVERLAP)}",
        flush=True,
    )

    dev_records, hold_records, base_meta = _fit_base_predictions(
        dev_files, hold_files, params, train_stride, eval_stride, typewell_hashes
    )
    print("building equal-cell interval landscapes...", flush=True)
    dev_rows, hold_rows = [], []
    for label, records, destination in (
        ("dev", dev_records, dev_rows), ("hold", hold_records, hold_rows)
    ):
        for i, record in enumerate(records, 1):
            row = {"well": record.well, **interval_features(record)}
            destination.append(row)
            if i % 50 == 0 or i == len(records):
                print(f"  {label}: {i}/{len(records)} wells", flush=True)
    dev_frame = pd.DataFrame(dev_rows).sort_values("well").reset_index(drop=True)
    hold_frame = pd.DataFrame(hold_rows).sort_values("well").reset_index(drop=True)
    dev_records = sorted(dev_records, key=lambda r: r.well)
    hold_records = sorted(hold_records, key=lambda r: r.well)
    dev_target = np.array([r.oracle_shift for r in dev_records])

    base_dev = _metric_block(dev_records)
    base_hold = _metric_block(hold_records)
    candidates: dict[str, dict] = {}
    scalar_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, column in (
        ("typewell_equal_cell", "tw_consensus_delta"),
        ("prefix_equal_cell", "prefix_consensus_delta"),
    ):
        dev_corr, hold_corr, shrink = _scalar_correction(
            dev_frame[column].to_numpy(dtype=float), dev_target,
            hold_frame[column].to_numpy(dtype=float),
        )
        hold_metric, hold_gain = _evaluate_candidate(hold_records, hold_corr)
        dev_metric, _ = _evaluate_candidate(dev_records, dev_corr)
        candidates[name] = {
            "calibrated_shift_shrink": shrink,
            "dev_oof_base_plus_correction": dev_metric,
            "holdout": hold_metric,
            "holdout_gain": hold_gain,
            "holdout_corrections": dict(zip(hold_frame["well"], hold_corr.tolist())),
        }
        scalar_corrections[name] = (dev_corr, hold_corr)

    dev_ridge, hold_ridge, _ = _fit_interval_correction(
        dev_frame, dev_target, hold_frame
    )
    ridge_hold, ridge_gain = _evaluate_candidate(hold_records, hold_ridge)
    ridge_dev, _ = _evaluate_candidate(dev_records, dev_ridge)
    candidates["fused_interval_ridge"] = {
        "dev_crossfit": ridge_dev,
        "holdout": ridge_hold,
        "holdout_gain": ridge_gain,
        "holdout_corrections": dict(zip(hold_frame["well"], hold_ridge.tolist())),
    }

    ordered_hold_corrections: list[np.ndarray] | None = None
    if args.confirm:
        print("running frozen ordered reversible interval transport...", flush=True)
        ordered_dev, ordered_dev_diag = _ordered_corrections(dev_records, "dev")
        ordered_hold, ordered_hold_diag = _ordered_corrections(hold_records, "hold")
        raw_hold, raw_gain = _evaluate_candidate(hold_records, ordered_hold)
        raw_dev, _ = _evaluate_candidate(dev_records, ordered_dev)
        candidates["ordered_transport_raw"] = {
            "frozen_settings": "research.ordered_transport.FROZEN_SETTINGS",
            "dev_crossfit": raw_dev,
            "holdout": raw_hold,
            "holdout_gain": raw_gain,
            "holdout_diagnostics": ordered_hold_diag,
        }
        ordered_shrink = _calibrate_vector_shrink(dev_records, ordered_dev)
        ordered_dev_scaled = [ordered_shrink * c for c in ordered_dev]
        ordered_hold_scaled = [ordered_shrink * c for c in ordered_hold]
        scaled_hold, scaled_gain = _evaluate_candidate(
            hold_records, ordered_hold_scaled
        )
        scaled_dev, _ = _evaluate_candidate(dev_records, ordered_dev_scaled)
        candidates["ordered_transport_calibrated"] = {
            "correction_shrink_from_dev_oof": ordered_shrink,
            "dev_crossfit": scaled_dev,
            "holdout": scaled_hold,
            "holdout_gain": scaled_gain,
            "dev_diagnostics": ordered_dev_diag,
            "holdout_diagnostics": ordered_hold_diag,
        }
        ordered_hold_corrections = ordered_hold_scaled

        tw_dev_scalar, tw_hold_scalar = scalar_corrections["typewell_equal_cell"]
        tw_dev_arrays = [np.full(len(r.idx), c) for r, c in zip(dev_records, tw_dev_scalar)]
        tw_hold_arrays = [np.full(len(r.idx), c) for r, c in zip(hold_records, tw_hold_scalar)]
        average_dev = [0.5 * (a + b) for a, b in zip(tw_dev_arrays, ordered_dev_scaled)]
        average_hold = [0.5 * (a + b) for a, b in zip(tw_hold_arrays, ordered_hold_scaled)]
        average_hold_metric, average_gain = _evaluate_candidate(hold_records, average_hold)
        average_dev_metric, _ = _evaluate_candidate(dev_records, average_dev)
        candidates["equal_ordered_average"] = {
            "weights": {"typewell_equal_cell": 0.5, "ordered_transport": 0.5},
            "dev_crossfit": average_dev_metric,
            "holdout": average_hold_metric,
            "holdout_gain": average_gain,
        }

        joint_coef = _fit_joint_correction(
            dev_records, tw_dev_arrays, ordered_dev_scaled
        )
        joint_dev = [
            joint_coef[0] * a + joint_coef[1] * b
            for a, b in zip(tw_dev_arrays, ordered_dev_scaled)
        ]
        joint_hold = [
            joint_coef[0] * a + joint_coef[1] * b
            for a, b in zip(tw_hold_arrays, ordered_hold_scaled)
        ]
        joint_hold = [np.clip(c, -25.0, 25.0) for c in joint_hold]
        joint_dev = [np.clip(c, -25.0, 25.0) for c in joint_dev]
        joint_hold_metric, joint_gain = _evaluate_candidate(hold_records, joint_hold)
        joint_dev_metric, _ = _evaluate_candidate(dev_records, joint_dev)
        candidates["equal_ordered_joint"] = {
            "coefficients_from_dev_oof": {
                "typewell_equal_cell": float(joint_coef[0]),
                "ordered_transport": float(joint_coef[1]),
            },
            "dev_crossfit": joint_dev_metric,
            "holdout": joint_hold_metric,
            "holdout_gain": joint_gain,
        }

    per_well = pd.DataFrame({
        "well": [r.well for r in hold_records],
        "oracle_shift": [r.oracle_shift for r in hold_records],
        "base_rmse": [base_hold["per_well"][r.well] for r in hold_records],
        "tw_delta": hold_frame["tw_consensus_delta"],
        "prefix_delta": hold_frame["prefix_consensus_delta"],
        "ridge_correction": hold_ridge,
        "ridge_rmse": [ridge_hold["per_well"][r.well] for r in hold_records],
    })
    if ordered_hold_corrections is not None:
        ordered_metrics = candidates["ordered_transport_calibrated"]["holdout"]
        per_well["ordered_rmse"] = [
            ordered_metrics["per_well"][r.well] for r in hold_records
        ]
        joint_metrics = candidates["equal_ordered_joint"]["holdout"]
        per_well["joint_rmse"] = [
            joint_metrics["per_well"][r.well] for r in hold_records
        ]

    result = {
        "status": "MEASURE_ONLY",
        "question": "Does equal-cell interval GR evidence improve an inference-safe TVT baseline?",
        "mode": "confirm" if args.confirm else "screen",
        "data_dir": str(Path(args.data_dir).resolve()),
        "eligible_train_wells": len(files),
        "dev_wells": len(dev_records),
        "holdout_wells": len(hold_records),
        "excluded_test_overlap_ids": sorted(EXCLUDED_TEST_OVERLAP),
        "split_unit": "exact typewell TVT/GR profile hash",
        "unique_typewell_groups": int(len(set(typewell_hashes.values()))),
        "evaluation_stride": eval_stride,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "bin_widths_ft": list(BIN_WIDTHS),
        "candidate_offsets_ft": [float(OFFSETS.min()), float(OFFSETS.max()), 0.5],
        "base_model": {**base_meta, "dev_oof": base_dev, "holdout": base_hold},
        "candidates": candidates,
        "runtime_seconds": float(time.time() - started),
        "interpretation_gate": (
            "Candidate only; repeat grouped CV and audit before OPEN. "
            "Effects below 0.2 ft median are treated as unresolved protocol noise."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    per_well_path = output.with_name(output.stem + "_per_well.csv")
    per_well.to_csv(per_well_path, index=False)
    print(f"wrote {output}", flush=True)
    print(f"wrote {per_well_path}", flush=True)
    print(json.dumps({
        "base": base_hold | {"per_well": "omitted"},
        "candidates": {
            k: v["holdout"] | {"per_well": "omitted"}
            for k, v in candidates.items()
        },
        "gains": {k: v["holdout_gain"] for k, v in candidates.items()},
    }, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=os.environ.get("GEOSTEERN_DATA_DIR"),
        required="GEOSTEERN_DATA_DIR" not in os.environ,
        help="directory containing train/, test/, and sample_submission.csv; "
             "defaults to $GEOSTEERN_DATA_DIR",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="use inherited 900-tree/stride-8 baseline settings",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "research" / "results" / "interval_gate_screen.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
