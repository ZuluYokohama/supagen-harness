"""Frozen repeated outer-group gate for the v2 interval-transport candidate.

The command has four deliberately separate phases:

``freeze``
    Persist the complete 2x5 exact-typewell-profile fold manifest and a
    write-once protocol, then hash both before any scoring occurs.
``audit``
    Fail closed if the protocol, manifest, source, package environment, fold
    membership, or input CSV bytes have drifted.
``run``
    Produce resumable fold shards without printing interim metrics. Every
    learned quantity is fitted from the outer-training fold only.
``aggregate``
    Require all ten shards, reveal metrics once, and calculate a paired
    exact-typewell-group bootstrap.

This remains MEASURE_ONLY research code.  Suffix TVT is used solely to fit on
outer-training wells and to score outer-test wells.  Formation surfaces,
Geology, PNGs, and the three train/test-overlap IDs are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from geosteern.data import list_wells, load_well, well_id
from geosteern.features import build_dataset, point_frame
from geosteern.model import PARAMS, calibrate_shrink
from research.interval_gate import (
    BIN_WIDTHS,
    EXCLUDED_TEST_OVERLAP,
    FORBIDDEN_FEATURE_COLUMNS,
    OFFSETS,
    PredictionRecord,
    _assert_inference_safe_feature_surface,
    _calibrate_vector_shrink,
    _calibrated_typewell_atlas,
    _fit_joint_correction,
    _landscape,
    _landscape_features,
    _make_atlas,
    _robust_scale,
    _scalar_correction,
)
from research.ordered_transport import (
    FROZEN_SETTINGS,
    ordered_reversible_interval_transport,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "geosteern-repeated-typewell-gate/1"
METHOD = "equal_ordered_joint_v2"

OUTER_REPEATS = 2
OUTER_SPLITS = 5
OUTER_SEEDS = (20260810, 20260811)
INNER_SPLITS = 4
TRAIN_STRIDE = 8
EVALUATION_STRIDE = 1
EXPECTED_ELIGIBLE_WELLS = 770
EXPECTED_TYPEWELL_GROUPS = 749

BOOTSTRAP_SEED = 20260810
BOOTSTRAP_DRAWS = 4000
UNRESOLVED_EFFECT_FT = 0.2
TOP_POSITIVE_SSE_REMOVAL_WELLS = 10
COEFFICIENT_BOUND_TOLERANCE = 1e-9

FROZEN_BASE_MODEL_PARAM_ITEMS = (
    ("n_estimators", 900),
    ("learning_rate", 0.04),
    ("num_leaves", 63),
    ("min_child_samples", 200),
    ("subsample", 0.8),
    ("subsample_freq", 1),
    ("colsample_bytree", 0.7),
    ("reg_lambda", 20.0),
    ("verbose", -1),
)
FROZEN_ORDERED_SETTING_ITEMS = (
    ("window_half_width_md", 90.0),
    ("window_samples", 13),
    ("emission_weight", 4.0),
    ("smoothing_samples", 7),
    ("state_step_tvt", 0.5),
    ("state_half_width_tvt", 150.0),
    ("node_step_md", 15.0),
    ("max_transition_bins", 10),
    ("transition_weight", 0.15),
    ("transition_scale_tvt", 1.2),
    ("anchor_weight", 10.0),
    ("geometry_sigma_initial", 1.5),
    ("geometry_sigma_growth_per_md", 0.015),
    ("huber_cap", 4.0),
    ("anchor_huber_cap", 100.0),
)

SOURCE_FILES = (
    "geosteern/data.py",
    "geosteern/features.py",
    "geosteern/model.py",
    "research/interval_gate.py",
    "research/ordered_transport.py",
    "research/repeated_group_gate.py",
    "research/test_interval_gate.py",
    "research/test_ordered_transport.py",
    "research/test_repeated_group_gate.py",
)
PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")

MANIFEST_COLUMNS = (
    "repeat",
    "outer_fold",
    "seed",
    "well",
    "typewell_profile_hash",
    "rows",
    "prefix_rows",
    "suffix_rows",
    "gr_valid_fraction",
    "horizontal_sha256",
    "typewell_sha256",
    "horizontal_file",
    "typewell_file",
)
SCORED_SSE_COLUMNS = (
    "repeat",
    "outer_fold",
    "well",
    "typewell_profile_hash",
    "n_rows",
    "base_sse",
    "typewell_sse",
    "ordered_sse",
    "joint_sse",
)


class ProtocolError(RuntimeError):
    """Raised when a frozen protocol or fold artifact fails closed."""


@dataclass(frozen=True)
class AuditBundle:
    protocol: dict
    manifest: pd.DataFrame
    protocol_sha256: str
    protocol_path: Path
    manifest_path: Path


def frozen_research_params() -> dict:
    """Return v2 confirm parameters plus explicit deterministic LightGBM flags."""

    params = dict(FROZEN_BASE_MODEL_PARAM_ITEMS)
    if dict(PARAMS) != params:
        raise ProtocolError(
            "imported production model parameters drifted from frozen values"
        )
    params.update(
        n_jobs=4,
        random_state=20260810,
        deterministic=True,
        force_col_wise=True,
    )
    return params


def _frozen_ordered_settings() -> dict:
    settings = dict(FROZEN_ORDERED_SETTING_ITEMS)
    if dict(FROZEN_SETTINGS) != settings:
        raise ProtocolError(
            "imported ordered-transport settings drifted from frozen values"
        )
    return settings


def _frozen_evaluation_protocol() -> dict:
    """Return the complete, predeclared scoring contract."""

    return {
        "primary_metric": "mean of repeat-wise pooled suffix-row RMSE gains versus base",
        "repeat_aggregation": "compute each repeat gain independently, then arithmetic mean",
        "bootstrap_unit": "exact typewell profile group",
        "bootstrap_repeat_coupling": (
            "one sampled group multiplicity is applied to every well in both repeats; "
            "each draw averages the two repeat-wise pooled RMSE gains"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "unresolved_effect_ft": UNRESOLVED_EFFECT_FT,
        "interim_metrics": "forbidden",
        "prediction_artifact": {
            "format": "compressed NPZ plus sealed JSON metadata",
            "truth_columns": "forbidden",
            "arrays": [
                "well_index",
                "row_index",
                "base_prediction",
                "typewell_prediction",
                "ordered_prediction",
                "joint_prediction",
            ],
        },
        "aggregate_artifacts": {
            "fold_lineage": (
                "complete repeat/fold inventory of shard JSON and prediction NPZ "
                "paths, names, byte hashes, and NPZ logical-array digests"
            ),
            "scored_rows": "per-well/per-repeat SSE CSV plus SHA-256 sidecar",
            "result": "final JSON plus SHA-256 sidecar",
            "overwrite": "forbidden",
        },
        "support_criteria": {
            "all_repeat_pooled_gains_positive": True,
            "primary_mean_repeat_pooled_gain_at_least_ft": UNRESOLVED_EFFECT_FT,
            "exact_group_bootstrap_ci95_low_positive": True,
            "paired_median_well_gain_at_least_ft": UNRESOLVED_EFFECT_FT,
            "joint_vs_resample_best_component_group_bootstrap_ci95_low_positive": True,
            "coefficient_stability": {
                "no_sign_flips_across_outer_folds": True,
                "repeated_bound_hit_definition": "two or more folds at one bound",
                "repeated_bound_hits_allowed": False,
                "shrink_bounds": [0.0, 1.5],
                "joint_coefficient_bounds": [-1.0, 2.0],
                "absolute_tolerance": COEFFICIENT_BOUND_TOLERANCE,
            },
            "top_positive_sse_removal": {
                "unit": "well",
                "remove": TOP_POSITIVE_SSE_REMOVAL_WELLS,
                "ranking": (
                    "descending sum over both repeats of base_sse minus joint_sse, "
                    "restricted to positive values; well ID breaks ties"
                ),
                "remaining_mean_repeat_pooled_gain_positive": True,
            },
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.stem + "_fold_manifest.csv")


def _scored_sse_path(path: Path) -> Path:
    return path.with_name(path.stem + "_scored_sse.csv")


def _fold_shard_name(repeat: int, fold: int) -> str:
    return f"repeat_{repeat}_fold_{fold}.json"


def _prediction_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".npz")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ProtocolError(f"stale temporary artifact exists: {temporary}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ProtocolError(f"stale temporary artifact exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_hash_sidecar(path: Path) -> Path:
    sidecar = _protocol_sidecar(path)
    _atomic_write_bytes(
        sidecar,
        f"{sha256_file(path)}  {path.name}\n".encode("ascii"),
    )
    return sidecar


def _packages() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ProtocolError(f"required source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _assign_outer_folds(base: pd.DataFrame) -> pd.DataFrame:
    """Create the frozen 2x5 shuffled GroupKFold manifest."""

    required = {
        "well",
        "typewell_profile_hash",
        "rows",
        "prefix_rows",
        "suffix_rows",
        "gr_valid_fraction",
        "horizontal_sha256",
        "typewell_sha256",
        "horizontal_file",
        "typewell_file",
    }
    missing = required - set(base.columns)
    if missing:
        raise ProtocolError(f"base manifest is missing columns: {sorted(missing)}")
    base = base.sort_values("well").reset_index(drop=True)
    if base["well"].duplicated().any():
        raise ProtocolError("base manifest contains duplicate well IDs")
    groups = base["typewell_profile_hash"].to_numpy()
    if len(set(groups)) < OUTER_SPLITS:
        raise ProtocolError("not enough exact-typewell groups for five outer folds")

    frames = []
    for repeat, seed in enumerate(OUTER_SEEDS):
        assignment = np.full(len(base), -1, dtype=int)
        splitter = GroupKFold(
            n_splits=OUTER_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        for fold, (_, test_index) in enumerate(splitter.split(base, groups=groups)):
            assignment[test_index] = fold
        if np.any(assignment < 0):
            raise ProtocolError("outer splitter left unassigned wells")
        frame = base.copy()
        frame.insert(0, "seed", seed)
        frame.insert(0, "outer_fold", assignment)
        frame.insert(0, "repeat", repeat)
        frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True)
    return (
        manifest.loc[:, MANIFEST_COLUMNS]
        .sort_values(["repeat", "well"])
        .reset_index(drop=True)
    )


def _validate_manifest(manifest: pd.DataFrame, protocol: dict | None = None) -> None:
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ProtocolError(f"fold manifest is missing columns: {sorted(missing)}")
    if manifest[list(MANIFEST_COLUMNS)].isna().any().any():
        raise ProtocolError("fold manifest contains missing values")
    repeats = sorted(manifest["repeat"].astype(int).unique().tolist())
    if repeats != list(range(OUTER_REPEATS)):
        raise ProtocolError(f"expected repeats 0..{OUTER_REPEATS - 1}, got {repeats}")

    reference_wells: set[str] | None = None
    for repeat in repeats:
        frame = manifest.loc[manifest["repeat"] == repeat]
        if frame["well"].duplicated().any():
            raise ProtocolError(f"repeat {repeat} contains duplicate wells")
        wells = set(frame["well"].astype(str))
        if reference_wells is None:
            reference_wells = wells
        elif wells != reference_wells:
            raise ProtocolError("repeats do not contain the same wells")
        folds = sorted(frame["outer_fold"].astype(int).unique().tolist())
        if folds != list(range(OUTER_SPLITS)):
            raise ProtocolError(f"repeat {repeat} does not contain all outer folds")
        split_counts = frame.groupby("typewell_profile_hash")["outer_fold"].nunique()
        if not (split_counts == 1).all():
            raise ProtocolError(f"repeat {repeat} splits an exact typewell profile")
        expected_seed = OUTER_SEEDS[repeat]
        if set(frame["seed"].astype(int)) != {expected_seed}:
            raise ProtocolError(f"repeat {repeat} has the wrong seed")
        if set(frame["well"].astype(str)) & EXCLUDED_TEST_OVERLAP:
            raise ProtocolError("excluded train/test-overlap well entered the manifest")

    base = manifest.loc[manifest["repeat"] == 0].drop(
        columns=["repeat", "outer_fold", "seed"]
    )
    expected = _assign_outer_folds(base)
    compare_columns = ["repeat", "well", "outer_fold", "seed"]
    actual_assignment = (
        manifest[compare_columns].sort_values(["repeat", "well"]).reset_index(drop=True)
    )
    expected_assignment = (
        expected[compare_columns].sort_values(["repeat", "well"]).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        actual_assignment,
        expected_assignment,
        check_dtype=False,
        obj="frozen outer-fold assignment",
    )
    if protocol is not None:
        expected_rows = int(protocol["data"]["eligible_wells"]) * OUTER_REPEATS
        if len(manifest) != expected_rows:
            raise ProtocolError(
                f"manifest has {len(manifest)} rows, expected {expected_rows}"
            )
        if manifest["well"].nunique() != int(protocol["data"]["eligible_wells"]):
            raise ProtocolError("protocol eligible-well count does not match manifest")
        if manifest["typewell_profile_hash"].nunique() != int(
            protocol["data"]["unique_typewell_groups"]
        ):
            raise ProtocolError("protocol typewell-group count does not match manifest")


def _base_manifest(data_dir: Path) -> pd.DataFrame:
    files = [
        path
        for path in list_wells(str(data_dir), "train")
        if well_id(path) not in EXCLUDED_TEST_OVERLAP
    ]
    rows = []
    for number, horizontal in enumerate(files, 1):
        horizontal_path = Path(horizontal)
        typewell_path = _resolve_typewell_path(horizontal_path)
        inference_well = _load_inference_well(horizontal)
        frame = inference_well["df"]
        try:
            _, inference_indices, inference_target = point_frame(
                inference_well, stride=EVALUATION_STRIDE
            )
        except Exception as exc:
            raise ProtocolError(
                f"frozen inference population contains an unusable well: "
                f"{well_id(horizontal)}"
            ) from exc
        if inference_target is not None:
            raise ProtocolError(
                "freeze inference view unexpectedly exposed suffix truth"
            )
        if not np.array_equal(
            inference_indices, np.flatnonzero(inference_well["tail"])
        ):
            raise ProtocolError(
                f"freeze inference row alignment failed for {well_id(horizontal)}"
            )
        typewell_frame = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        typewell_values = pd.util.hash_pandas_object(
            typewell_frame, index=False
        ).to_numpy(dtype=np.uint64)
        typewell_profile_hash = hashlib.sha256(typewell_values.tobytes()).hexdigest()
        wid = well_id(horizontal)
        rows.append(
            {
                "well": wid,
                "typewell_profile_hash": typewell_profile_hash,
                "rows": int(len(frame)),
                "prefix_rows": int(frame["TVT_input"].notna().sum()),
                "suffix_rows": int(frame["TVT_input"].isna().sum()),
                "gr_valid_fraction": float(frame["GR"].notna().mean()),
                "horizontal_sha256": sha256_file(horizontal_path),
                "typewell_sha256": sha256_file(typewell_path),
                "horizontal_file": horizontal_path.name,
                "typewell_file": typewell_path.name,
            }
        )
        if number % 100 == 0:
            print(f"freeze inventory: {number}/{len(files)} wells", flush=True)
    if not rows:
        raise ProtocolError("no eligible wells found")
    return pd.DataFrame(rows).sort_values("well").reset_index(drop=True)


def _resolve_typewell_path(horizontal_path: Path) -> Path:
    candidates = sorted(
        path
        for path in horizontal_path.parent.glob(
            f"{well_id(str(horizontal_path))}__typewel*"
        )
        if "horizontal" not in path.name and path.is_file()
    )
    if len(candidates) != 1:
        raise ProtocolError(
            f"expected one unambiguous typewell for {horizontal_path}, got {len(candidates)}"
        )
    return candidates[0]


def _load_inference_well(horizontal_csv: str) -> dict:
    """Load only fields present at competition inference; never read horizontal TVT."""

    horizontal_path = Path(horizontal_csv)
    safe_columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    frame = pd.read_csv(horizontal_path, usecols=safe_columns)
    typewell = pd.read_csv(
        _resolve_typewell_path(horizontal_path), usecols=["TVT", "GR"]
    )
    typewell = typewell.dropna().sort_values("TVT")
    if len(typewell) < 50:
        raise ProtocolError(f"inference typewell is too short: {horizontal_path}")
    known = frame["TVT_input"].notna().to_numpy().copy()
    first_tail = np.flatnonzero(~known)
    if first_tail.size and known[first_tail[0] :].any():
        raise ProtocolError(f"non-contiguous known prefix: {horizontal_path}")
    geometry_ok = frame[["MD", "X", "Y", "Z"]].notna().all(axis=1).to_numpy()
    known &= geometry_ok
    tail = (~known) & geometry_ok
    if known.sum() < 50 or tail.sum() < 5:
        raise ProtocolError(f"inference well is too short: {horizontal_path}")
    return {
        "well": well_id(horizontal_csv),
        "df": frame,
        "tw": typewell,
        "known": known,
        "tail": tail,
        "tvt_prefix": frame["TVT_input"].to_numpy(),
        "truth": None,
    }


def freeze_protocol(data_dir: Path, protocol_path: Path) -> tuple[Path, Path, Path]:
    """Write a complete manifest, protocol, and protocol-hash sidecar once."""

    data_dir = data_dir.resolve()
    protocol_path = protocol_path.resolve()
    manifest_path = _manifest_path(protocol_path)
    sidecar_path = _protocol_sidecar(protocol_path)
    for path in (protocol_path, manifest_path, sidecar_path):
        if path.exists():
            raise ProtocolError(
                f"freeze is write-once; artifact already exists: {path}"
            )

    # Resolve and hash every executable dependency (including tests) before
    # writing any freeze artifact.  A missing source or package therefore
    # cannot strand an orphan manifest that blocks a clean retry.
    source_hashes = _source_hashes()
    packages = _packages()
    _assert_inference_safe_feature_surface()
    base = _base_manifest(data_dir)
    if len(base) != EXPECTED_ELIGIBLE_WELLS:
        raise ProtocolError(
            f"expected {EXPECTED_ELIGIBLE_WELLS} eligible wells, got {len(base)}"
        )
    observed_groups = int(base["typewell_profile_hash"].nunique())
    if observed_groups != EXPECTED_TYPEWELL_GROUPS:
        raise ProtocolError(
            f"expected {EXPECTED_TYPEWELL_GROUPS} exact typewell groups, "
            f"got {observed_groups}"
        )
    manifest = _assign_outer_folds(base)
    _validate_manifest(manifest)
    manifest_bytes = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_SCORING_MEASURE_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "data": {
            "data_dir": str(data_dir),
            "eligible_wells": int(base["well"].nunique()),
            "unique_typewell_groups": int(base["typewell_profile_hash"].nunique()),
            "excluded_test_overlap_ids": sorted(EXCLUDED_TEST_OVERLAP),
            "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        },
        "outer_cv": {
            "repeats": OUTER_REPEATS,
            "splits": OUTER_SPLITS,
            "seeds": list(OUTER_SEEDS),
            "splitter": "sklearn.model_selection.GroupKFold(shuffle=True)",
            "group": "exact typewell TVT/GR profile hash",
        },
        "inner_fit": {
            "splits": INNER_SPLITS,
            "group": "exact typewell TVT/GR profile hash",
            "train_stride": TRAIN_STRIDE,
            "evaluation_stride": EVALUATION_STRIDE,
            "research_params": frozen_research_params(),
        },
        "interval_method": {
            "bin_widths_ft": list(BIN_WIDTHS),
            "offsets_ft": [float(OFFSETS.min()), float(OFFSETS.max()), 0.5],
            "landscape_min_cells": 6,
            "landscape_correlation_weight": 0.30,
            "landscape_gradient_weight": 0.20,
            "typewell_scalar_shrink_clip": [0.0, 1.5],
            "ordered_vector_shrink_clip": [0.0, 1.5],
            "joint_penalty_trace_multiplier": 1e-4,
            "joint_coefficient_clip": [-1.0, 2.0],
            "final_correction_clip_ft": [-25.0, 25.0],
            "ordered_transport": _frozen_ordered_settings(),
        },
        "evaluation": _frozen_evaluation_protocol(),
        "manifest": {
            "file": manifest_path.name,
            "sha256": manifest_hash,
            "rows": int(len(manifest)),
        },
        "source_sha256": source_hashes,
        "packages": packages,
        "notes": [
            "Only the equal_ordered_joint_v2 procedure is scored; no candidate selection is allowed.",
            "All model weights and all shrink/joint coefficients are refit inside each outer train fold.",
            "Freeze inventories inference columns only and never opens horizontal suffix TVT.",
            "Run opens suffix TVT for outer-training fits only; every outer-test suffix is scoring-only in aggregate.",
            "Held-out own-well prefix calibration is allowed because TVT_input is available at inference.",
            "Formation surfaces, Geology, PNGs, and ID lookup are forbidden features.",
            "Repeated CV on this previously studied corpus measures stability, not pristine external confirmation.",
        ],
    }
    # Serialize the entire protocol before committing either file, so all
    # validation and JSON conversion failures happen before the first write.
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    _atomic_write_bytes(manifest_path, manifest_bytes)
    _atomic_write_bytes(protocol_path, protocol_bytes)
    _atomic_write_bytes(
        sidecar_path,
        f"{protocol_hash}  {protocol_path.name}\n".encode("ascii"),
    )
    return protocol_path, manifest_path, sidecar_path


def _read_sidecar(protocol_path: Path) -> str:
    sidecar = _protocol_sidecar(protocol_path)
    if not sidecar.is_file():
        raise ProtocolError(f"protocol hash sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != protocol_path.name or len(fields[0]) != 64:
        raise ProtocolError("malformed protocol hash sidecar")
    return fields[0].lower()


def _safe_train_file(data_dir: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ProtocolError(f"manifest contains unsafe filename: {name!r}")
    path = (data_dir / "train" / name).resolve()
    train_root = (data_dir / "train").resolve()
    if path.parent != train_root:
        raise ProtocolError(f"manifest path escapes train directory: {path}")
    return path


def _audit_dataset(manifest: pd.DataFrame, data_dir: Path) -> None:
    inventory = manifest.loc[manifest["repeat"] == 0].sort_values("well")
    for number, row in enumerate(inventory.itertuples(index=False), 1):
        horizontal = _safe_train_file(data_dir, str(row.horizontal_file))
        typewell = _safe_train_file(data_dir, str(row.typewell_file))
        if not horizontal.is_file() or not typewell.is_file():
            raise ProtocolError(f"frozen dataset file is missing for well {row.well}")
        if sha256_file(horizontal) != str(row.horizontal_sha256):
            raise ProtocolError(f"horizontal dataset hash drift for well {row.well}")
        if sha256_file(typewell) != str(row.typewell_sha256):
            raise ProtocolError(f"typewell dataset hash drift for well {row.well}")
        if number % 200 == 0:
            print(f"audit inventory: {number}/{len(inventory)} wells", flush=True)


def audit_protocol(protocol_path: Path, verify_data: bool = True) -> AuditBundle:
    """Verify the frozen protocol and fail closed on any drift."""

    protocol_path = protocol_path.resolve()
    expected_protocol_hash = _read_sidecar(protocol_path)
    actual_protocol_hash = sha256_file(protocol_path)
    if actual_protocol_hash != expected_protocol_hash:
        raise ProtocolError("protocol hash drift")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    if protocol.get("method") != METHOD:
        raise ProtocolError("frozen method identity changed")
    if protocol.get("data", {}).get("eligible_wells") != EXPECTED_ELIGIBLE_WELLS:
        raise ProtocolError("frozen eligible-well population drifted")
    if (
        protocol.get("data", {}).get("unique_typewell_groups")
        != EXPECTED_TYPEWELL_GROUPS
    ):
        raise ProtocolError("frozen exact-typewell-group population drifted")
    expected_outer = {
        "repeats": OUTER_REPEATS,
        "splits": OUTER_SPLITS,
        "seeds": list(OUTER_SEEDS),
        "splitter": "sklearn.model_selection.GroupKFold(shuffle=True)",
        "group": "exact typewell TVT/GR profile hash",
    }
    if protocol.get("outer_cv") != expected_outer:
        raise ProtocolError("outer-fold protocol drift")
    if protocol.get("inner_fit", {}).get("research_params") != frozen_research_params():
        raise ProtocolError("frozen deterministic research parameters drifted")
    if (
        protocol.get("interval_method", {}).get("ordered_transport")
        != _frozen_ordered_settings()
    ):
        raise ProtocolError("ordered-transport frozen settings drifted")
    if protocol.get("evaluation") != _frozen_evaluation_protocol():
        raise ProtocolError("frozen evaluation protocol drifted")
    if protocol.get("source_sha256") != _source_hashes():
        raise ProtocolError("source hash drift")
    if protocol.get("packages") != _packages():
        raise ProtocolError("package-version drift")

    manifest_name = str(protocol.get("manifest", {}).get("file", ""))
    if Path(manifest_name).name != manifest_name:
        raise ProtocolError("unsafe manifest filename in protocol")
    manifest_path = protocol_path.with_name(manifest_name)
    if not manifest_path.is_file():
        raise ProtocolError(f"fold manifest is missing: {manifest_path}")
    if sha256_file(manifest_path) != protocol["manifest"]["sha256"]:
        raise ProtocolError("fold manifest hash drift")
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(manifest, protocol)

    data_dir = Path(protocol["data"]["data_dir"]).resolve()
    if verify_data:
        _audit_dataset(manifest, data_dir)
    return AuditBundle(
        protocol=protocol,
        manifest=manifest,
        protocol_sha256=actual_protocol_hash,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )


def _build_static_training_matrix(
    files: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build the immutable stride-8 matrix once; outer masks remain mandatory."""

    return build_dataset(files, load_well, TRAIN_STRIDE)


def _fit_base_fold(
    train_files: Sequence[str],
    test_files: Sequence[str],
    typewell_hashes: dict[str, str],
    static_matrix: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> tuple[list[PredictionRecord], list[PredictionRecord], dict]:
    """Nested base fit with no outer-test row entering any estimator."""

    X_all, y_all, row_wells_all = static_matrix
    train_ids = {well_id(path) for path in train_files}
    test_ids = {well_id(path) for path in test_files}
    if train_ids & test_ids:
        raise ProtocolError("outer train/test well overlap")
    train_mask = np.fromiter(
        (wid in train_ids for wid in row_wells_all),
        dtype=bool,
        count=len(row_wells_all),
    )
    if not train_mask.any():
        raise ProtocolError("outer train fold has no static training rows")
    if set(row_wells_all[train_mask]) & test_ids:
        raise ProtocolError("outer-test rows entered the training matrix")
    unexpected_static_wells = set(row_wells_all) - train_ids
    if unexpected_static_wells:
        raise ProtocolError(
            "fold-static matrix contains non-training wells: "
            f"{sorted(unexpected_static_wells)[:5]}"
        )

    X = X_all.loc[train_mask].reset_index(drop=True)
    y = y_all[train_mask]
    row_wells = row_wells_all[train_mask]
    files_by_id = {well_id(path): path for path in train_files}
    oof = np.zeros_like(y)
    raw_records: dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    inner_groups = np.array([typewell_hashes[wid] for wid in row_wells])
    splitter = GroupKFold(INNER_SPLITS)
    params = frozen_research_params()
    for train_index, validation_index in splitter.split(X, y, groups=inner_groups):
        model = lgb.LGBMRegressor(**params).fit(X.iloc[train_index], y[train_index])
        oof[validation_index] = model.predict(X.iloc[validation_index])
        validation_wells = sorted(set(row_wells[validation_index]))
        for wid in validation_wells:
            path = files_by_id[wid]
            well = load_well(path)
            if well is None:
                raise ProtocolError(f"could not reload outer-train well {wid}")
            frame, indices, delta_truth = point_frame(well, stride=EVALUATION_STRIDE)
            raw = model.predict(frame[X.columns])
            t_last = float(well["tvt_prefix"][well["known"]][-1])
            raw_records[wid] = (path, indices, raw, delta_truth, t_last)

    if set(raw_records) != train_ids:
        missing = sorted(train_ids - set(raw_records))
        raise ProtocolError(
            f"inner OOF did not predict all outer-train wells: {missing[:5]}"
        )
    base_shrink = calibrate_shrink(oof, y)
    train_records = [
        PredictionRecord(
            well=wid,
            path=value[0],
            idx=value[1],
            prediction=value[4] + base_shrink * value[2],
            truth=value[4] + value[3],
        )
        for wid, value in sorted(raw_records.items())
    ]

    final_model = lgb.LGBMRegressor(**params).fit(X, y)
    test_records = []
    for path in test_files:
        inference_well = _load_inference_well(path)
        frame, indices, delta_truth = point_frame(
            inference_well, stride=EVALUATION_STRIDE
        )
        if delta_truth is not None:
            raise ProtocolError("outer-test inference view unexpectedly exposed truth")
        raw = final_model.predict(frame[X.columns])
        t_last = float(inference_well["tvt_prefix"][inference_well["known"]][-1])
        test_records.append(
            PredictionRecord(
                well=inference_well["well"],
                path=path,
                idx=indices,
                prediction=t_last + base_shrink * raw,
                truth=np.full(len(indices), np.nan),
            )
        )
    if {record.well for record in test_records} != test_ids:
        raise ProtocolError("base fit did not predict every outer-test well")
    denominator = float(np.sum((y - y.mean()) ** 2))
    oof_r2 = 1.0 - float(np.sum((base_shrink * oof - y) ** 2)) / denominator
    return (
        train_records,
        test_records,
        {
            "base_shrink": float(base_shrink),
            "inner_oof_point_r2": float(oof_r2),
            "training_points": int(len(X)),
            "training_wells": int(len(train_ids)),
            "training_typewell_groups": int(len(set(inner_groups))),
        },
    )


def _typewell_consensus_delta(record: PredictionRecord, well: dict) -> float:
    """Compute only v2's selected typewell landscape, without unused candidates."""

    df = well["df"]
    gr = df["GR"].to_numpy()
    known = well["known"]
    deltas = []
    for width in BIN_WIDTHS:
        query = _make_atlas(record.prediction, gr[record.idx], width)
        prefix = _make_atlas(
            well["tvt_prefix"][known],
            gr[known],
            width,
        )
        typewell, _ = _calibrated_typewell_atlas(well, width)
        scale = _robust_scale(
            prefix.quartiles[:, 1] if not prefix.empty else query.quartiles[:, 1]
        )
        costs, supports = _landscape(query, typewell, width, scale)
        tag = str(width).replace(".", "p")
        features = _landscape_features(costs, supports, f"tw_{tag}")
        delta = features[f"tw_{tag}_delta_soft"]
        if np.isfinite(delta):
            deltas.append(float(delta))
    return float(np.median(deltas)) if deltas else float("nan")


def _evidence_for_records(
    records: Sequence[PredictionRecord],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, dict]]:
    """Load each well once while producing both frozen evidence components."""

    typewell_delta = np.empty(len(records), dtype=float)
    ordered_corrections: list[np.ndarray] = []
    diagnostics: dict[str, dict] = {}
    for index, record in enumerate(records):
        well = _load_inference_well(record.path)
        if len(record.idx) != int(well["tail"].sum()):
            raise ProtocolError(
                "strict outer gate requires full stride-1 suffix scoring"
            )
        typewell_delta[index] = _typewell_consensus_delta(record, well)
        df, typewell = well["df"], well["tw"]
        proposal = well["tvt_prefix"].copy()
        proposal[record.idx] = record.prediction
        result = ordered_reversible_interval_transport(
            horizontal_md=df["MD"].to_numpy(),
            horizontal_gr=df["GR"].to_numpy(),
            typewell_tvt=typewell["TVT"].to_numpy(),
            typewell_gr=typewell["GR"].to_numpy(),
            known_prefix_tvt=well["tvt_prefix"],
            proposed_tvt_path=proposal,
        )
        ordered_corrections.append(result.corrected_tvt[record.idx] - record.prediction)
        diag = result.diagnostics
        diagnostics[record.well] = {
            "status": diag.status,
            "observed_gr_fraction": float(diag.observed_gr_fraction),
            "viterbi_nodes": int(diag.viterbi_nodes),
            "states": int(diag.states),
            "forward_orientation_nodes": int(diag.forward_orientation_nodes),
            "reverse_orientation_nodes": int(diag.reverse_orientation_nodes),
            "reversal_count": int(diag.reversal_count),
            "boundary_state_nodes": int(diag.boundary_state_nodes),
        }
    return typewell_delta, ordered_corrections, diagnostics


def _logical_array_hash(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, dtypes, shapes, and C-order bytes independent of NPZ bytes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _expected_fold_wells(manifest: pd.DataFrame, repeat: int, fold: int) -> set[str]:
    selection = (manifest["repeat"] == repeat) & (manifest["outer_fold"] == fold)
    return set(manifest.loc[selection, "well"].astype(str))


def _validate_shard(
    shard: dict,
    audit: AuditBundle,
    repeat: int,
    fold: int,
    shard_path: Path,
) -> dict[str, np.ndarray]:
    if _read_sidecar(shard_path) != sha256_file(shard_path):
        raise ProtocolError("fold shard metadata hash drift")
    if shard.get("protocol_sha256") != audit.protocol_sha256:
        raise ProtocolError("fold shard protocol hash mismatch")
    if shard.get("manifest_sha256") != audit.protocol["manifest"]["sha256"]:
        raise ProtocolError("fold shard manifest hash mismatch")
    if (
        int(shard.get("repeat", -1)) != repeat
        or int(shard.get("outer_fold", -1)) != fold
    ):
        raise ProtocolError("fold shard identity mismatch")
    rows = shard.get("test_wells")
    if not isinstance(rows, list):
        raise ProtocolError("fold shard has no test-well rows")
    wells = [str(row.get("well")) for row in rows]
    if len(wells) != len(set(wells)):
        raise ProtocolError("fold shard contains duplicate test wells")
    if set(wells) != _expected_fold_wells(audit.manifest, repeat, fold):
        raise ProtocolError("fold shard test membership differs from frozen manifest")
    expected_groups = {
        str(row.well): str(row.typewell_profile_hash)
        for row in audit.manifest.loc[
            (audit.manifest["repeat"] == repeat)
            & (audit.manifest["outer_fold"] == fold)
        ].itertuples(index=False)
    }
    for metadata_index, row in enumerate(rows):
        if int(row.get("well_index", -1)) != metadata_index:
            raise ProtocolError("fold shard well-index ordering drift")
        if str(row.get("typewell_profile_hash")) != expected_groups[str(row["well"])]:
            raise ProtocolError("fold shard typewell group mismatch")
        count = int(row.get("n_rows", 0))
        if count <= 0:
            raise ProtocolError("fold shard contains an invalid prediction-row count")

    prediction_name = str(shard.get("prediction_file", ""))
    if Path(prediction_name).name != prediction_name:
        raise ProtocolError("unsafe prediction filename in fold shard")
    prediction_path = shard_path.with_name(prediction_name)
    if not prediction_path.is_file():
        raise ProtocolError(f"fold prediction artifact is missing: {prediction_path}")
    if sha256_file(prediction_path) != shard.get("prediction_sha256"):
        raise ProtocolError("fold prediction NPZ hash drift")
    expected_arrays = {
        "well_index",
        "row_index",
        "base_prediction",
        "typewell_prediction",
        "ordered_prediction",
        "joint_prediction",
    }
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_arrays:
            raise ProtocolError("fold prediction NPZ schema drift")
        arrays = {name: archive[name].copy() for name in expected_arrays}
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1 or lengths.pop() != int(shard.get("prediction_rows", -1)):
        raise ProtocolError("fold prediction arrays have inconsistent lengths")
    if _logical_array_hash(arrays) != shard.get("prediction_logical_sha256"):
        raise ProtocolError("fold prediction logical-array hash drift")
    if (
        arrays["well_index"].dtype.kind not in "iu"
        or arrays["row_index"].dtype.kind not in "iu"
    ):
        raise ProtocolError("fold prediction identity arrays must be integer typed")
    if np.any(arrays["well_index"] < 0) or np.any(arrays["well_index"] >= len(rows)):
        raise ProtocolError("fold prediction well index is out of range")
    for name in expected_arrays - {"well_index", "row_index"}:
        if not np.isfinite(arrays[name]).all():
            raise ProtocolError(f"fold prediction array is non-finite: {name}")
    for well_index, row in enumerate(rows):
        selected = arrays["well_index"] == well_index
        if int(selected.sum()) != int(row["n_rows"]):
            raise ProtocolError("fold prediction row count does not match metadata")
        indices = arrays["row_index"][selected]
        if len(indices) != len(np.unique(indices)) or np.any(indices < 0):
            raise ProtocolError(
                "fold prediction row indices are duplicated or negative"
            )
    return arrays


def _fold_inventory_digest(items: Sequence[dict]) -> str:
    encoded = json.dumps(
        list(items), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_fold_artifact_inventory_payload(inventory: dict) -> None:
    """Fail closed unless an inventory covers every fold artifact exactly once."""

    items = inventory.get("items")
    if not isinstance(items, list):
        raise ProtocolError("fold artifact inventory has no item list")
    expected = {
        (repeat, fold)
        for repeat in range(OUTER_REPEATS)
        for fold in range(OUTER_SPLITS)
    }
    identities = []
    shard_paths = []
    prediction_paths = []
    for item in items:
        if not isinstance(item, dict):
            raise ProtocolError("fold artifact inventory item is not an object")
        identity = (int(item.get("repeat", -1)), int(item.get("outer_fold", -1)))
        identities.append(identity)
        shard = item.get("shard_json")
        prediction = item.get("prediction_npz")
        if not isinstance(shard, dict) or not isinstance(prediction, dict):
            raise ProtocolError("fold artifact inventory item is incomplete")
        shard_paths.append(str(shard.get("path", "")))
        prediction_paths.append(str(prediction.get("path", "")))
        for artifact, label in ((shard, "shard JSON"), (prediction, "prediction NPZ")):
            name = str(artifact.get("name", ""))
            path = str(artifact.get("path", ""))
            byte_hash = str(artifact.get("byte_sha256", ""))
            if Path(path).name != name or len(byte_hash) != 64:
                raise ProtocolError(f"invalid {label} identity in fold inventory")
            try:
                int(byte_hash, 16)
            except ValueError as exc:
                raise ProtocolError(
                    f"invalid {label} SHA-256 in fold inventory"
                ) from exc
        logical_hash = str(prediction.get("logical_array_sha256", ""))
        if len(logical_hash) != 64:
            raise ProtocolError("invalid prediction logical digest in fold inventory")
        try:
            int(logical_hash, 16)
        except ValueError as exc:
            raise ProtocolError(
                "invalid prediction logical digest in fold inventory"
            ) from exc

    if len(identities) != len(set(identities)):
        raise ProtocolError("fold artifact inventory contains duplicate identities")
    if len(items) != len(expected) or set(identities) != expected:
        missing = sorted(expected - set(identities))
        extra = sorted(set(identities) - expected)
        raise ProtocolError(
            "fold artifact inventory is incomplete or unexpected; "
            f"missing={missing}, extra={extra}"
        )
    if len(shard_paths) != len(set(shard_paths)):
        raise ProtocolError("fold artifact inventory contains duplicate shard paths")
    if len(prediction_paths) != len(set(prediction_paths)):
        raise ProtocolError(
            "fold artifact inventory contains duplicate prediction paths"
        )
    if set(shard_paths) & set(prediction_paths):
        raise ProtocolError("fold artifact inventory aliases JSON and NPZ paths")
    if int(inventory.get("fold_count", -1)) != len(expected):
        raise ProtocolError("fold artifact inventory count is incorrect")
    if inventory.get("inventory_sha256") != _fold_inventory_digest(items):
        raise ProtocolError("fold artifact inventory logical digest mismatch")


def _build_fold_artifact_inventory(
    validated: Sequence[tuple[int, int, Path, dict, dict[str, np.ndarray]]],
) -> dict:
    """Inventory byte/logical identities for all already-validated fold shards."""

    items = []
    for repeat, fold, shard_path, shard, arrays in validated:
        shard_path = shard_path.resolve()
        if (
            shard_path.name != _fold_shard_name(repeat, fold)
            or not shard_path.is_file()
        ):
            raise ProtocolError("validated fold shard path or identity drifted")
        on_disk_shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if on_disk_shard != shard:
            raise ProtocolError("validated fold shard JSON changed before inventory")
        shard_hash = sha256_file(shard_path)
        if _read_sidecar(shard_path) != shard_hash:
            raise ProtocolError(
                "validated fold shard JSON hash drifted before inventory"
            )

        prediction_name = str(shard.get("prediction_file", ""))
        prediction_path = shard_path.with_name(prediction_name).resolve()
        if (
            prediction_path != _prediction_path(shard_path).resolve()
            or not prediction_path.is_file()
        ):
            raise ProtocolError("validated fold prediction path or identity drifted")
        prediction_hash = sha256_file(prediction_path)
        logical_hash = _logical_array_hash(arrays)
        if prediction_hash != shard.get("prediction_sha256"):
            raise ProtocolError("validated fold prediction byte hash drifted")
        if logical_hash != shard.get("prediction_logical_sha256"):
            raise ProtocolError("validated fold prediction logical digest drifted")
        items.append(
            {
                "repeat": int(repeat),
                "outer_fold": int(fold),
                "shard_json": {
                    "name": shard_path.name,
                    "path": str(shard_path),
                    "size_bytes": int(shard_path.stat().st_size),
                    "byte_sha256": shard_hash,
                    "sha256_sidecar_name": _protocol_sidecar(shard_path).name,
                    "sha256_sidecar_path": str(_protocol_sidecar(shard_path).resolve()),
                },
                "prediction_npz": {
                    "name": prediction_path.name,
                    "path": str(prediction_path),
                    "size_bytes": int(prediction_path.stat().st_size),
                    "byte_sha256": prediction_hash,
                    "logical_array_sha256": logical_hash,
                },
            }
        )
    items.sort(key=lambda item: (item["repeat"], item["outer_fold"]))
    inventory = {
        "fold_count": len(items),
        "items": items,
        "inventory_sha256": _fold_inventory_digest(items),
    }
    _validate_fold_artifact_inventory_payload(inventory)
    return inventory


def _run_one_fold(
    audit: AuditBundle,
    repeat: int,
    fold: int,
    output_dir: Path,
    resume: bool,
) -> Path:
    shard_path = output_dir / _fold_shard_name(repeat, fold)
    prediction_path = _prediction_path(shard_path)
    shard_sidecar = _protocol_sidecar(shard_path)
    if shard_path.exists():
        if not resume:
            raise ProtocolError(f"fold shard already exists: {shard_path}")
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        _validate_shard(shard, audit, repeat, fold, shard_path)
        return shard_path
    if prediction_path.exists() or shard_sidecar.exists():
        raise ProtocolError(
            f"incomplete/stale fold artifacts exist without metadata: {shard_path}"
        )

    started = time.time()
    frame = audit.manifest.loc[audit.manifest["repeat"] == repeat]
    test_frame = frame.loc[frame["outer_fold"] == fold]
    train_frame = frame.loc[frame["outer_fold"] != fold]
    train_groups = set(train_frame["typewell_profile_hash"].astype(str))
    test_groups = set(test_frame["typewell_profile_hash"].astype(str))
    if train_groups & test_groups:
        raise ProtocolError("exact typewell group crossed the outer boundary")
    data_dir = Path(audit.protocol["data"]["data_dir"])
    train_files = [
        str(_safe_train_file(data_dir, name))
        for name in train_frame.sort_values("well")["horizontal_file"].astype(str)
    ]
    test_files = [
        str(_safe_train_file(data_dir, name))
        for name in test_frame.sort_values("well")["horizontal_file"].astype(str)
    ]
    group_by_well = {
        str(row.well): str(row.typewell_profile_hash)
        for row in frame.itertuples(index=False)
    }

    # This cache contains outer-training labels only and is reused by the four
    # inner fits plus the final fit. Outer-test truth never enters this matrix.
    static_matrix = _build_static_training_matrix(train_files)
    train_records, test_records, base_meta = _fit_base_fold(
        train_files,
        test_files,
        group_by_well,
        static_matrix,
    )
    train_tw, train_ordered_raw, _ = _evidence_for_records(train_records)
    test_tw, test_ordered_raw, test_diagnostics = _evidence_for_records(test_records)
    if not (
        len(train_records) == len(train_tw) == len(train_ordered_raw)
        and len(test_records) == len(test_tw) == len(test_ordered_raw)
    ):
        raise ProtocolError("evidence output length mismatch")
    train_target = np.array([record.oracle_shift for record in train_records])
    train_tw_scalar, test_tw_scalar, typewell_shrink = _scalar_correction(
        train_tw,
        train_target,
        test_tw,
    )
    ordered_shrink = _calibrate_vector_shrink(train_records, train_ordered_raw)
    train_ordered = [ordered_shrink * correction for correction in train_ordered_raw]
    test_ordered = [ordered_shrink * correction for correction in test_ordered_raw]
    train_tw_arrays = [
        np.full(len(record.idx), correction)
        for record, correction in zip(train_records, train_tw_scalar, strict=True)
    ]
    test_tw_arrays = [
        np.full(len(record.idx), correction)
        for record, correction in zip(test_records, test_tw_scalar, strict=True)
    ]
    joint = _fit_joint_correction(train_records, train_tw_arrays, train_ordered)
    joint_corrections = [
        np.clip(joint[0] * first + joint[1] * second, -25.0, 25.0)
        for first, second in zip(test_tw_arrays, test_ordered, strict=True)
    ]
    if not (
        len(test_records)
        == len(test_tw_arrays)
        == len(test_ordered)
        == len(joint_corrections)
    ):
        raise ProtocolError("prediction-channel count mismatch")

    well_indices = []
    row_indices = []
    base_predictions = []
    typewell_predictions = []
    ordered_predictions = []
    joint_predictions = []
    test_wells = []
    for well_index, (record, tw, ordered, joint_correction) in enumerate(
        zip(
            test_records,
            test_tw_arrays,
            test_ordered,
            joint_corrections,
            strict=True,
        )
    ):
        n_rows = len(record.idx)
        if not (
            len(record.prediction)
            == len(tw)
            == len(ordered)
            == len(joint_correction)
            == n_rows
        ):
            raise ProtocolError(f"prediction length mismatch for {record.well}")
        if len(np.unique(record.idx)) != n_rows or np.any(record.idx < 0):
            raise ProtocolError(f"invalid suffix row indices for {record.well}")
        well_indices.append(np.full(n_rows, well_index, dtype=np.int32))
        row_indices.append(np.asarray(record.idx, dtype=np.int32))
        base_predictions.append(np.asarray(record.prediction, dtype=np.float64))
        typewell_predictions.append(
            np.asarray(record.prediction + tw, dtype=np.float64)
        )
        ordered_predictions.append(
            np.asarray(record.prediction + ordered, dtype=np.float64)
        )
        joint_predictions.append(
            np.asarray(record.prediction + joint_correction, dtype=np.float64)
        )
        test_wells.append(
            {
                "well": record.well,
                "well_index": well_index,
                "typewell_profile_hash": group_by_well[record.well],
                "n_rows": n_rows,
            }
        )
    arrays = {
        "well_index": np.concatenate(well_indices),
        "row_index": np.concatenate(row_indices),
        "base_prediction": np.concatenate(base_predictions),
        "typewell_prediction": np.concatenate(typewell_predictions),
        "ordered_prediction": np.concatenate(ordered_predictions),
        "joint_prediction": np.concatenate(joint_predictions),
    }
    if any(
        not np.isfinite(array).all()
        for name, array in arrays.items()
        if "index" not in name
    ):
        raise ProtocolError("non-finite prediction entered the sealed fold artifact")
    _atomic_write_npz(prediction_path, arrays)
    prediction_hash = sha256_file(prediction_path)
    logical_hash = _logical_array_hash(arrays)
    shard = {
        "status": "MEASURE_ONLY_PREDICTIONS_SEALED_TRUTH_UNREAD",
        "protocol_sha256": audit.protocol_sha256,
        "manifest_sha256": audit.protocol["manifest"]["sha256"],
        "method": METHOD,
        "repeat": repeat,
        "outer_fold": fold,
        "seed": OUTER_SEEDS[repeat],
        "train_wells": int(len(train_records)),
        "test_well_count": int(len(test_records)),
        "train_typewell_groups": int(len(train_groups)),
        "test_typewell_groups": int(len(test_groups)),
        "learned_from_outer_train_only": {
            **base_meta,
            "typewell_shrink": float(typewell_shrink),
            "ordered_shrink": float(ordered_shrink),
            "joint_coefficients": [float(joint[0]), float(joint[1])],
        },
        "test_diagnostics": test_diagnostics,
        "prediction_file": prediction_path.name,
        "prediction_sha256": prediction_hash,
        "prediction_logical_sha256": logical_hash,
        "prediction_rows": int(len(arrays["row_index"])),
        "prediction_channels": [
            "base_prediction",
            "typewell_prediction",
            "ordered_prediction",
            "joint_prediction",
        ],
        "test_wells": test_wells,
        "runtime_seconds": float(time.time() - started),
    }
    _atomic_write_json(shard_path, shard)
    _write_hash_sidecar(shard_path)
    _validate_shard(shard, audit, repeat, fold, shard_path)
    return shard_path


def run_folds(
    protocol_path: Path,
    output_dir: Path | None,
    folds: Sequence[tuple[int, int]],
    resume: bool,
) -> list[Path]:
    """Audit once, cache the static stride-8 matrix, and write metric-silent shards."""

    audit = audit_protocol(protocol_path, verify_data=True)
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    normalized = []
    for repeat, fold in folds:
        if repeat not in range(OUTER_REPEATS) or fold not in range(OUTER_SPLITS):
            raise ProtocolError(f"invalid fold identity ({repeat}, {fold})")
        if (repeat, fold) not in normalized:
            normalized.append((repeat, fold))
    pending = []
    completed = []
    for repeat, fold in normalized:
        shard_path = output_dir / _fold_shard_name(repeat, fold)
        if shard_path.exists() and resume:
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            _validate_shard(shard, audit, repeat, fold, shard_path)
            completed.append(shard_path)
        else:
            pending.append((repeat, fold))
    if not pending:
        print(
            "all requested fold shards already pass audit; metrics remain withheld",
            flush=True,
        )
        return completed

    for repeat, fold in pending:
        print(
            f"running frozen shard repeat {repeat + 1}/{OUTER_REPEATS}, "
            f"fold {fold + 1}/{OUTER_SPLITS}; interim metrics withheld",
            flush=True,
        )
        completed.append(
            _run_one_fold(
                audit,
                repeat,
                fold,
                output_dir,
                resume,
            )
        )
        # Recheck executable and data bytes after every long fold before the
        # prediction shard is accepted as part of the frozen run.
        audit_protocol(protocol_path, verify_data=True)
    print(
        f"completed {len(pending)} fold shard(s); metrics remain withheld", flush=True
    )
    return completed


def _summary(rows: pd.DataFrame, candidate_sse_column: str = "joint_sse") -> dict:
    count = rows["n_rows"].to_numpy(dtype=float)
    base_sse = rows["base_sse"].to_numpy(dtype=float)
    candidate_sse = rows[candidate_sse_column].to_numpy(dtype=float)
    # A well occurs once in each repeat. Aggregate its paired repeat errors
    # before well-level summaries so 770 wells do not masquerade as 1,540
    # independent wells.
    well_rows = (
        rows.assign(_candidate_sse=candidate_sse)
        .groupby("well", sort=True)
        .agg(
            n_rows=("n_rows", "sum"),
            base_sse=("base_sse", "sum"),
            candidate_sse=("_candidate_sse", "sum"),
        )
    )
    base_well = np.sqrt(
        well_rows["base_sse"].to_numpy(dtype=float)
        / well_rows["n_rows"].to_numpy(dtype=float)
    )
    candidate_well = np.sqrt(
        well_rows["candidate_sse"].to_numpy(dtype=float)
        / well_rows["n_rows"].to_numpy(dtype=float)
    )
    base_pooled = float(np.sqrt(base_sse.sum() / count.sum()))
    candidate_pooled = float(np.sqrt(candidate_sse.sum() / count.sum()))
    return {
        "base_pooled_row_rmse": base_pooled,
        "candidate_pooled_row_rmse": candidate_pooled,
        "pooled_row_rmse_gain_ft": base_pooled - candidate_pooled,
        "base_median_well_rmse": float(np.median(base_well)),
        "candidate_median_well_rmse": float(np.median(candidate_well)),
        "median_well_rmse_gain_ft": float(np.median(base_well - candidate_well)),
        "mean_well_rmse_gain_ft": float(np.mean(base_well - candidate_well)),
        "p90_base_well_rmse": float(np.quantile(base_well, 0.9)),
        "p90_candidate_well_rmse": float(np.quantile(candidate_well, 0.9)),
        "win_rate": float(np.mean(candidate_well < base_well)),
        "n_unique_wells": int(len(well_rows)),
        "n_well_repeat_predictions": int(len(rows)),
        "n_rows": int(count.sum()),
    }


def _repeat_pooled_gains(
    rows: pd.DataFrame,
    reference_sse_column: str = "base_sse",
    candidate_sse_column: str = "joint_sse",
) -> dict[str, float]:
    """Compute pooled gain independently within each frozen repeat."""

    required = {"repeat", "n_rows", reference_sse_column, candidate_sse_column}
    missing = required - set(rows.columns)
    if missing:
        raise ProtocolError(
            f"repeat summary rows are missing columns: {sorted(missing)}"
        )
    repeats = sorted(rows["repeat"].astype(int).unique().tolist())
    if repeats != list(range(OUTER_REPEATS)):
        raise ProtocolError(
            f"repeat summary expected repeats 0..{OUTER_REPEATS - 1}, got {repeats}"
        )
    gains = {}
    for repeat in repeats:
        frame = rows.loc[rows["repeat"].astype(int) == repeat]
        count = float(frame["n_rows"].sum())
        if count <= 0.0:
            raise ProtocolError(f"repeat {repeat} has no scoring rows")
        reference = float(frame[reference_sse_column].sum())
        candidate = float(frame[candidate_sse_column].sum())
        if reference < 0.0 or candidate < 0.0:
            raise ProtocolError("negative SSE entered repeat aggregation")
        gains[str(repeat)] = float(
            np.sqrt(reference / count) - np.sqrt(candidate / count)
        )
    return gains


def _paired_group_bootstrap_statistics(
    rows: pd.DataFrame,
    reference_sse_column: str,
    candidate_sse_column: str,
    draws: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    """Return repeat-coupled exact-profile bootstrap metadata and draws."""

    required = {
        "repeat",
        "typewell_profile_hash",
        "n_rows",
        reference_sse_column,
        candidate_sse_column,
    }
    missing = required - set(rows.columns)
    if missing:
        raise ProtocolError(f"bootstrap rows are missing columns: {sorted(missing)}")
    if draws <= 0:
        raise ProtocolError("bootstrap draws must be positive")
    repeats = sorted(rows["repeat"].astype(int).unique().tolist())
    if repeats != list(range(OUTER_REPEATS)):
        raise ProtocolError(
            f"bootstrap expected repeats 0..{OUTER_REPEATS - 1}, got {repeats}"
        )

    grouped = (
        rows.assign(_repeat=rows["repeat"].astype(int))
        .groupby(["typewell_profile_hash", "_repeat"], sort=True)
        .agg(
            n_rows=("n_rows", "sum"),
            reference_sse=(reference_sse_column, "sum"),
            candidate_sse=(candidate_sse_column, "sum"),
        )
    )
    groups = sorted(rows["typewell_profile_hash"].astype(str).unique().tolist())
    if len(groups) < 2:
        raise ProtocolError("group bootstrap needs at least two exact-typewell groups")
    # Every repeat contains the same frozen well population, so every exact
    # profile must occur in every repeat. Fail rather than silently imputing.
    expected_index = pd.MultiIndex.from_product(
        [groups, repeats], names=["typewell_profile_hash", "_repeat"]
    )
    grouped.index = pd.MultiIndex.from_tuples(
        [(str(group), int(repeat)) for group, repeat in grouped.index],
        names=grouped.index.names,
    )
    if set(grouped.index) != set(expected_index):
        raise ProtocolError("exact-profile groups are not represented in every repeat")
    grouped = grouped.reindex(expected_index)
    count = grouped["n_rows"].to_numpy(dtype=float).reshape(len(groups), len(repeats))
    reference = (
        grouped["reference_sse"]
        .to_numpy(dtype=float)
        .reshape(len(groups), len(repeats))
    )
    candidate = (
        grouped["candidate_sse"]
        .to_numpy(dtype=float)
        .reshape(len(groups), len(repeats))
    )
    if (
        not np.isfinite(count).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or np.any(count <= 0.0)
        or np.any(reference < 0.0)
        or np.any(candidate < 0.0)
    ):
        raise ProtocolError("invalid contribution entered group bootstrap")

    def statistic(sample: np.ndarray) -> tuple[float, list[float]]:
        repeat_gains = []
        for repeat_index in range(len(repeats)):
            denominator = float(count[sample, repeat_index].sum())
            repeat_gains.append(
                float(
                    np.sqrt(reference[sample, repeat_index].sum() / denominator)
                    - np.sqrt(candidate[sample, repeat_index].sum() / denominator)
                )
            )
        return float(np.mean(repeat_gains)), repeat_gains

    all_groups = np.arange(len(groups), dtype=np.int64)
    observed, observed_repeats = statistic(all_groups)
    rng = np.random.default_rng(seed)
    gain_draws = np.empty(draws, dtype=float)
    for draw in range(draws):
        # This one sample is reused for both repeats, so a duplicated profile's
        # wells and both repeat occurrences move as one cluster.
        sample = rng.integers(0, len(groups), len(groups))
        gain_draws[draw], _ = statistic(sample)
    return (
        {
            "unit": "exact typewell profile group",
            "repeat_coupling": "same sampled group multiplicities in both repeats",
            "statistic": "mean of repeat-wise pooled RMSE gains",
            "n_groups": int(len(groups)),
            "draws": int(draws),
            "seed": int(seed),
            "observed_mean_repeat_pooled_rmse_gain_ft": float(observed),
            "observed_repeat_pooled_rmse_gains_ft": {
                str(repeat): float(gain)
                for repeat, gain in zip(repeats, observed_repeats, strict=True)
            },
            "ci95_low_ft": float(np.quantile(gain_draws, 0.025)),
            "ci95_high_ft": float(np.quantile(gain_draws, 0.975)),
        },
        gain_draws,
    )


def _group_cluster_bootstrap(
    rows: pd.DataFrame,
    candidate_sse_column: str = "joint_sse",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired exact-profile bootstrap with the two repeats kept coupled."""

    result, _ = _paired_group_bootstrap_statistics(
        rows,
        "base_sse",
        candidate_sse_column,
        draws,
        seed,
    )
    return result


def _joint_vs_best_component_bootstrap(
    rows: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Conservatively compare joint with the better component in every draw."""

    typewell, typewell_draws = _paired_group_bootstrap_statistics(
        rows,
        "typewell_sse",
        "joint_sse",
        draws,
        seed,
    )
    ordered, ordered_draws = _paired_group_bootstrap_statistics(
        rows,
        "ordered_sse",
        "joint_sse",
        draws,
        seed,
    )
    if typewell["n_groups"] != ordered["n_groups"]:
        raise ProtocolError("component bootstraps used different exact-profile groups")
    conservative_draws = np.minimum(typewell_draws, ordered_draws)
    observed_typewell = typewell["observed_mean_repeat_pooled_rmse_gain_ft"]
    observed_ordered = ordered["observed_mean_repeat_pooled_rmse_gain_ft"]
    return {
        "unit": "exact typewell profile group",
        "repeat_coupling": "same sampled group multiplicities in both repeats",
        "definition": (
            "minimum of typewell-to-joint and ordered-to-joint gain in each "
            "resample; positive values mean joint beats the resample-best component"
        ),
        "n_groups": int(typewell["n_groups"]),
        "draws": int(draws),
        "seed": int(seed),
        "observed_joint_vs_typewell_gain_ft": float(observed_typewell),
        "observed_joint_vs_ordered_gain_ft": float(observed_ordered),
        "observed_joint_vs_best_component_gain_ft": float(
            min(observed_typewell, observed_ordered)
        ),
        "ci95_low_ft": float(np.quantile(conservative_draws, 0.025)),
        "ci95_high_ft": float(np.quantile(conservative_draws, 0.975)),
    }


def _top_positive_sse_removal(
    rows: pd.DataFrame,
    candidate_sse_column: str = "joint_sse",
    remove: int = TOP_POSITIVE_SSE_REMOVAL_WELLS,
) -> dict:
    """Remove the most beneficial wells across both repeats and rescore."""

    required = {"well", "repeat", "n_rows", "base_sse", candidate_sse_column}
    missing = required - set(rows.columns)
    if missing:
        raise ProtocolError(f"removal rows are missing columns: {sorted(missing)}")
    if remove <= 0:
        raise ProtocolError("positive-SSE removal count must be positive")
    ranked = (
        rows.groupby("well", sort=True)
        .agg(base_sse=("base_sse", "sum"), candidate_sse=(candidate_sse_column, "sum"))
        .reset_index()
    )
    ranked["positive_sse_benefit"] = ranked["base_sse"] - ranked["candidate_sse"]
    ranked = ranked.loc[ranked["positive_sse_benefit"] > 0.0].sort_values(
        ["positive_sse_benefit", "well"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = ranked.head(remove)
    removed_wells = selected["well"].astype(str).tolist()
    remaining = rows.loc[~rows["well"].astype(str).isin(removed_wells)].copy()
    if remaining.empty:
        raise ProtocolError("positive-SSE removal left no scoring wells")
    repeat_gains = _repeat_pooled_gains(
        remaining,
        "base_sse",
        candidate_sse_column,
    )
    mean_gain = float(np.mean(list(repeat_gains.values())))
    return {
        "definition": (
            "remove the ten wells with largest positive base-minus-joint SSE "
            "summed across both repeats"
        ),
        "requested_removal_count": int(remove),
        "removed_count": int(len(removed_wells)),
        "removed_wells": removed_wells,
        "removed_positive_sse_benefit": {
            str(row.well): float(row.positive_sse_benefit)
            for row in selected.itertuples(index=False)
        },
        "remaining_repeat_pooled_rmse_gains_ft": repeat_gains,
        "remaining_mean_repeat_pooled_rmse_gain_ft": mean_gain,
        "passed": bool(len(removed_wells) == remove and mean_gain > 0.0),
    }


def _coefficient_stability(learned: Sequence[dict]) -> dict:
    """Audit interval coefficients for sign flips and repeated bound hits."""

    expected_folds = OUTER_REPEATS * OUTER_SPLITS
    if len(learned) != expected_folds:
        raise ProtocolError(
            f"coefficient audit expected {expected_folds} folds, got {len(learned)}"
        )
    definitions = {
        "typewell_shrink": (
            np.array([row["typewell_shrink"] for row in learned], dtype=float),
            (0.0, 1.5),
        ),
        "ordered_shrink": (
            np.array([row["ordered_shrink"] for row in learned], dtype=float),
            (0.0, 1.5),
        ),
        "joint_typewell_coefficient": (
            np.array([row["joint_coefficients"][0] for row in learned], dtype=float),
            (-1.0, 2.0),
        ),
        "joint_ordered_coefficient": (
            np.array([row["joint_coefficients"][1] for row in learned], dtype=float),
            (-1.0, 2.0),
        ),
    }
    coefficients = {}
    for name, (values, bounds) in definitions.items():
        lower, upper = bounds
        if (
            not np.isfinite(values).all()
            or np.any(values < lower - COEFFICIENT_BOUND_TOLERANCE)
            or np.any(values > upper + COEFFICIENT_BOUND_TOLERANCE)
        ):
            raise ProtocolError(f"{name} is non-finite or outside frozen bounds")
        positive = bool(np.any(values > COEFFICIENT_BOUND_TOLERANCE))
        negative = bool(np.any(values < -COEFFICIENT_BOUND_TOLERANCE))
        sign_flip = positive and negative
        lower_hits = int(
            np.count_nonzero(
                np.isclose(values, lower, rtol=0.0, atol=COEFFICIENT_BOUND_TOLERANCE)
            )
        )
        upper_hits = int(
            np.count_nonzero(
                np.isclose(values, upper, rtol=0.0, atol=COEFFICIENT_BOUND_TOLERANCE)
            )
        )
        repeated_bound_hit = lower_hits >= 2 or upper_hits >= 2
        coefficients[name] = {
            "bounds": [lower, upper],
            "values_by_fold": values.tolist(),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "positive_fold_count": int(
                np.count_nonzero(values > COEFFICIENT_BOUND_TOLERANCE)
            ),
            "negative_fold_count": int(
                np.count_nonzero(values < -COEFFICIENT_BOUND_TOLERANCE)
            ),
            "zero_fold_count": int(
                np.count_nonzero(np.abs(values) <= COEFFICIENT_BOUND_TOLERANCE)
            ),
            "sign_flip": bool(sign_flip),
            "lower_bound_hits": lower_hits,
            "upper_bound_hits": upper_hits,
            "repeated_bound_hit": bool(repeated_bound_hit),
            "passed": bool(not sign_flip and not repeated_bound_hit),
        }
    return {
        "fold_order": [
            {"repeat": int(row["repeat"]), "outer_fold": int(row["outer_fold"])}
            for row in learned
        ],
        "absolute_tolerance": COEFFICIENT_BOUND_TOLERANCE,
        "repeated_bound_hit_definition": "two or more folds at one bound",
        "coefficients": coefficients,
        "passed": bool(all(row["passed"] for row in coefficients.values())),
    }


def _score_sealed_predictions(
    audit: AuditBundle,
    shard: dict,
    arrays: dict[str, np.ndarray],
    repeat: int,
    fold: int,
) -> list[dict]:
    """First and only phase that opens outer-validation suffix truth."""

    fold_manifest = audit.manifest.loc[
        (audit.manifest["repeat"] == repeat) & (audit.manifest["outer_fold"] == fold)
    ]
    manifest_by_well = {
        str(row.well): row for row in fold_manifest.itertuples(index=False)
    }
    data_dir = Path(audit.protocol["data"]["data_dir"])
    rows = []
    for well_index, metadata in enumerate(shard["test_wells"]):
        wid = str(metadata["well"])
        if int(metadata["well_index"]) != well_index:
            raise ProtocolError("fold metadata well-index ordering drift")
        manifest_row = manifest_by_well[wid]
        horizontal = str(_safe_train_file(data_dir, str(manifest_row.horizontal_file)))
        well = load_well(horizontal)
        if well is None or well["truth"] is None:
            raise ProtocolError(f"aggregate could not load validation truth for {wid}")
        selected = arrays["well_index"] == well_index
        indices = arrays["row_index"][selected].astype(np.int64, copy=False)
        expected_indices = np.flatnonzero(well["tail"])
        if not np.array_equal(indices, expected_indices):
            raise ProtocolError(
                f"sealed prediction rows do not equal full suffix for {wid}"
            )
        truth = np.asarray(well["truth"][indices], dtype=float)
        if not np.isfinite(truth).all():
            raise ProtocolError(f"validation truth is non-finite for {wid}")
        scoring = {
            channel: np.asarray(arrays[channel][selected], dtype=float)
            for channel in (
                "base_prediction",
                "typewell_prediction",
                "ordered_prediction",
                "joint_prediction",
            )
        }
        if any(len(prediction) != len(truth) for prediction in scoring.values()):
            raise ProtocolError(f"prediction/truth length mismatch for {wid}")
        rows.append(
            {
                "repeat": repeat,
                "outer_fold": fold,
                "well": wid,
                "typewell_profile_hash": str(metadata["typewell_profile_hash"]),
                "n_rows": int(len(truth)),
                "base_sse": float(np.sum((scoring["base_prediction"] - truth) ** 2)),
                "typewell_sse": float(
                    np.sum((scoring["typewell_prediction"] - truth) ** 2)
                ),
                "ordered_sse": float(
                    np.sum((scoring["ordered_prediction"] - truth) ** 2)
                ),
                "joint_sse": float(np.sum((scoring["joint_prediction"] - truth) ** 2)),
            }
        )
    return rows


def _persist_aggregate_artifacts(
    output_path: Path,
    scored: pd.DataFrame,
    result: dict,
) -> dict:
    """Atomically write scored rows and result, each with a SHA-256 sidecar."""

    output_path = output_path.resolve()
    scored_path = _scored_sse_path(output_path)
    output_sidecar = _protocol_sidecar(output_path)
    scored_sidecar = _protocol_sidecar(scored_path)
    for path in (scored_path, scored_sidecar, output_path, output_sidecar):
        if path.exists():
            raise ProtocolError(f"refusing to overwrite aggregate artifact: {path}")
    inventory = result.get("fold_artifact_inventory")
    if not isinstance(inventory, dict):
        raise ProtocolError("final result is missing the fold artifact inventory")
    _validate_fold_artifact_inventory_payload(inventory)
    for label in ("protocol_sha256", "manifest_sha256"):
        value = str(result.get(label, ""))
        if len(value) != 64:
            raise ProtocolError(f"final result has no valid {label}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ProtocolError(f"final result has no valid {label}") from exc
    missing = set(SCORED_SSE_COLUMNS) - set(scored.columns)
    if missing:
        raise ProtocolError(f"scored rows are missing columns: {sorted(missing)}")
    ordered = scored.loc[:, SCORED_SSE_COLUMNS].sort_values(
        ["repeat", "well"], kind="mergesort"
    )
    if ordered.duplicated(["repeat", "well"]).any():
        raise ProtocolError("scored SSE artifact contains duplicate well/repeat rows")
    scored_bytes = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    scored_hash = hashlib.sha256(scored_bytes).hexdigest()
    sealed_result = dict(result)
    sealed_result["scored_sse_artifact"] = {
        "file": scored_path.name,
        "sha256": scored_hash,
        "sha256_sidecar": scored_sidecar.name,
        "rows": int(len(ordered)),
        "columns": list(SCORED_SSE_COLUMNS),
    }
    sealed_result["result_sha256_sidecar"] = output_sidecar.name
    sealed_result["artifact_lineage"] = {
        "protocol_sha256": sealed_result.get("protocol_sha256"),
        "manifest_sha256": sealed_result.get("manifest_sha256"),
        "fold_artifact_inventory_sha256": inventory["inventory_sha256"],
        "scored_sse_sha256": scored_hash,
        "result_sealed_by": output_sidecar.name,
    }
    result_bytes = (json.dumps(sealed_result, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    # Both payloads are fully serialized before the first write. The CSV is
    # committed first because its digest is embedded in the final result.
    _atomic_write_bytes(scored_path, scored_bytes)
    _write_hash_sidecar(scored_path)
    _atomic_write_bytes(output_path, result_bytes)
    _write_hash_sidecar(output_path)
    return sealed_result


def aggregate_folds(
    protocol_path: Path,
    shard_dir: Path | None,
    output_path: Path,
) -> dict:
    """Require all frozen shards before revealing any repeated-CV metrics."""

    output_path = output_path.resolve()
    scored_output_path = _scored_sse_path(output_path)
    for path in (
        scored_output_path,
        _protocol_sidecar(scored_output_path),
        output_path,
        _protocol_sidecar(output_path),
    ):
        if path.exists():
            raise ProtocolError(f"refusing to overwrite aggregate artifact: {path}")
    audit = audit_protocol(protocol_path, verify_data=True)
    shard_dir = (
        shard_dir.resolve()
        if shard_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    expected_shards = {
        _fold_shard_name(repeat, fold)
        for repeat in range(OUTER_REPEATS)
        for fold in range(OUTER_SPLITS)
    }
    present_shards = {path.name for path in shard_dir.glob("repeat_*_fold_*.json")}
    if present_shards != expected_shards:
        missing = sorted(expected_shards - present_shards)
        extra = sorted(present_shards - expected_shards)
        raise ProtocolError(
            f"fold shard set is incomplete or unexpected; missing={missing}, extra={extra}"
        )
    shard_runtime = 0.0
    learned = []
    validated = []
    for repeat in range(OUTER_REPEATS):
        for fold in range(OUTER_SPLITS):
            path = shard_dir / _fold_shard_name(repeat, fold)
            if not path.is_file():
                raise ProtocolError(f"cannot aggregate incomplete run; missing {path}")
            shard = json.loads(path.read_text(encoding="utf-8"))
            arrays = _validate_shard(shard, audit, repeat, fold, path)
            shard_runtime += float(shard["runtime_seconds"])
            learned.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    **shard["learned_from_outer_train_only"],
                }
            )
            validated.append((repeat, fold, path, shard, arrays))

    fold_artifact_inventory = _build_fold_artifact_inventory(validated)

    # The validation suffix is still unopened here. Only after all ten shard
    # metadata/NPZ artifacts pass the sealed protocol do we load any truth.
    rows = []
    for repeat, fold, _, shard, arrays in validated:
        rows.extend(
            _score_sealed_predictions(
                audit,
                shard,
                arrays,
                repeat,
                fold,
            )
        )
    scored = pd.DataFrame(rows)
    expected = (
        audit.manifest[["repeat", "outer_fold", "well"]]
        .sort_values(["repeat", "well"])
        .reset_index(drop=True)
    )
    actual = (
        scored[["repeat", "outer_fold", "well"]]
        .sort_values(["repeat", "well"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    repeat_summaries = {
        str(repeat): _summary(scored.loc[scored["repeat"] == repeat], "joint_sse")
        for repeat in range(OUTER_REPEATS)
    }
    primary_repeat_gains = _repeat_pooled_gains(scored, "base_sse", "joint_sse")
    primary_mean_repeat_gain = float(np.mean(list(primary_repeat_gains.values())))
    aggregate = _summary(scored, "joint_sse")
    aggregate["primary_repeat_pooled_rmse_gains_ft"] = primary_repeat_gains
    aggregate["primary_mean_repeat_pooled_rmse_gain_ft"] = primary_mean_repeat_gain
    component_diagnostics = {}
    for component, column in (
        ("typewell", "typewell_sse"),
        ("ordered", "ordered_sse"),
    ):
        component_summary = _summary(scored, column)
        component_repeat_gains = _repeat_pooled_gains(scored, "base_sse", column)
        component_summary["repeat_pooled_rmse_gains_ft"] = component_repeat_gains
        component_summary["mean_repeat_pooled_rmse_gain_ft"] = float(
            np.mean(list(component_repeat_gains.values()))
        )
        component_diagnostics[component] = component_summary
    bootstrap = _group_cluster_bootstrap(scored, "joint_sse")
    joint_vs_best = _joint_vs_best_component_bootstrap(scored)
    top_positive_removal = _top_positive_sse_removal(scored, "joint_sse")
    coefficient_stability = _coefficient_stability(learned)
    support = (
        all(gain > 0.0 for gain in primary_repeat_gains.values())
        and primary_mean_repeat_gain >= UNRESOLVED_EFFECT_FT
        and bootstrap["ci95_low_ft"] > 0.0
        and aggregate["median_well_rmse_gain_ft"] >= UNRESOLVED_EFFECT_FT
        and joint_vs_best["ci95_low_ft"] > 0.0
        and top_positive_removal["passed"]
        and coefficient_stability["passed"]
    )
    result = {
        "status": "MEASURE_ONLY_SUPPORTS_FROZEN_CANDIDATE"
        if support
        else "MEASURE_ONLY_NOT_CONFIRMED",
        "protocol_sha256": audit.protocol_sha256,
        "manifest_sha256": audit.protocol["manifest"]["sha256"],
        "method": METHOD,
        "fold_artifact_inventory": fold_artifact_inventory,
        "repeat_summaries": repeat_summaries,
        "aggregate": aggregate,
        "predeclared_component_diagnostics_no_selection": component_diagnostics,
        "exact_typewell_group_bootstrap": bootstrap,
        "joint_vs_resample_best_component_group_bootstrap": joint_vs_best,
        "top_10_positive_sse_removal": top_positive_removal,
        "coefficient_sign_and_bound_stability": coefficient_stability,
        "predeclared_support_gate": {
            "primary_metric": "mean of repeat-wise pooled suffix-row RMSE gains versus base",
            "primary_observed_ft": primary_mean_repeat_gain,
            "primary_materiality": {
                "threshold_ft": UNRESOLVED_EFFECT_FT,
                "observed_ft": primary_mean_repeat_gain,
                "passed": bool(primary_mean_repeat_gain >= UNRESOLVED_EFFECT_FT),
            },
            "all_repeat_pooled_gains_positive": bool(
                all(gain > 0.0 for gain in primary_repeat_gains.values())
            ),
            "group_bootstrap_ci95_low_positive": bool(bootstrap["ci95_low_ft"] > 0.0),
            "paired_median_well_gain_at_least_ft": {
                "threshold_ft": UNRESOLVED_EFFECT_FT,
                "observed_ft": aggregate["median_well_rmse_gain_ft"],
                "passed": bool(
                    aggregate["median_well_rmse_gain_ft"] >= UNRESOLVED_EFFECT_FT
                ),
            },
            "joint_vs_resample_best_component_ci95_low_positive": bool(
                joint_vs_best["ci95_low_ft"] > 0.0
            ),
            "top_10_positive_sse_removal_gain_positive": bool(
                top_positive_removal["passed"]
            ),
            "coefficient_sign_and_bound_stability": bool(
                coefficient_stability["passed"]
            ),
            "passed": bool(support),
        },
        "learned_parameters_by_fold": learned,
        "sum_fold_runtime_seconds": shard_runtime,
        "interpretation": (
            "Repeated CV on a previously studied corpus measures stability only; "
            "it is not pristine external confirmation and cannot promote production OPEN."
        ),
    }
    commit_audit = audit_protocol(protocol_path, verify_data=True)
    if (
        commit_audit.protocol_sha256 != audit.protocol_sha256
        or commit_audit.protocol["manifest"]["sha256"]
        != audit.protocol["manifest"]["sha256"]
    ):
        raise ProtocolError("protocol or manifest drifted before aggregate commit")
    commit_validated = []
    for repeat, fold, path, _, _ in validated:
        current_shard = json.loads(path.read_text(encoding="utf-8"))
        current_arrays = _validate_shard(
            current_shard, commit_audit, repeat, fold, path
        )
        commit_validated.append((repeat, fold, path, current_shard, current_arrays))
    if _build_fold_artifact_inventory(commit_validated) != fold_artifact_inventory:
        raise ProtocolError("fold artifact inventory drifted before aggregate commit")
    result = _persist_aggregate_artifacts(output_path, scored, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _all_folds() -> list[tuple[int, int]]:
    return [
        (repeat, fold)
        for repeat in range(OUTER_REPEATS)
        for fold in range(OUTER_SPLITS)
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="write the fold manifest and protocol"
    )
    freeze.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing train/ and test/",
    )
    freeze.add_argument("--protocol", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="fail closed on any protocol drift")
    audit.add_argument("--protocol", type=Path, required=True)

    run = subparsers.add_parser("run", help="write resumable metric-silent fold shards")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-folds", action="store_true")
    selection.add_argument("--repeat", type=int)
    run.add_argument("--fold", type=int)
    run.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate", help="require all shards, then reveal metrics once"
    )
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--shard-dir", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.all_folds and args.fold is not None:
            parser.error("--fold cannot be used with --all-folds")
        if args.repeat is not None and args.fold is None:
            parser.error("--fold is required when --repeat is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        protocol, manifest, sidecar = freeze_protocol(args.data_dir, args.protocol)
        print(f"froze protocol {protocol}")
        print(f"froze manifest {manifest}")
        print(f"wrote protocol hash {sidecar}")
    elif args.command == "audit":
        bundle = audit_protocol(args.protocol, verify_data=True)
        print(
            f"audit passed: {len(bundle.manifest)} manifest rows, "
            f"protocol {bundle.protocol_sha256}"
        )
    elif args.command == "run":
        if args.all_folds:
            folds = _all_folds()
        else:
            if args.fold is None:
                raise ProtocolError("--fold is required when --repeat is used")
            folds = [(args.repeat, args.fold)]
        run_folds(args.protocol, args.output_dir, folds, args.resume)
    elif args.command == "aggregate":
        aggregate_folds(args.protocol, args.shard_dir, args.output)
    else:  # pragma: no cover - argparse enforces a known subcommand
        raise ProtocolError(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "METHOD",
    "OUTER_REPEATS",
    "OUTER_SEEDS",
    "OUTER_SPLITS",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "aggregate_folds",
    "audit_protocol",
    "freeze_protocol",
    "frozen_research_params",
    "run_folds",
]
