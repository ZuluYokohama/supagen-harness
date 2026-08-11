"""Sealed prediction-before-truth gate for the anchored structural field.

This module is deliberately separate from :mod:`research.structural_field`.
It binds the already-sealed interval incumbent, cross-fits every newly learned
quantity inside an outer training role, writes metric-silent prediction
artifacts, and permits suffix truth to be opened only after the complete shard
set has passed a byte and logical-hash audit.

The research status ceiling is ``MEASURE_ONLY``.  No result emitted here can
promote a production ``OPEN`` without a genuinely external confirmation set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from research import repeated_group_gate as incumbent_exact
from research import spatial_score_gate as incumbent_spatial
from research import structural_field as field_core


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "geosteern-anchored-structural-field-gate/1"
METHOD = "anchored_differential_field_over_sealed_joint_v1"
STATUS_CEILING = "MEASURE_ONLY"

# Re-pinned after the 2026-08-11 review pass edited both files (NumPy inverse
# shape normalisation, cut-edge bounding-box prefilter, wall-clock assertion
# removal). The pre-review digests are recorded in docs/SEAL_STATE.md.
CORE_SHA256 = "2c3203ba336bf30c501f1c5fdfb242412b3d4625a010eec3c729773c3dcc736e"
CORE_TEST_SHA256 = "ef8b081279705f5c7a2f1625d5a27e9769c0d6ad3f484684092bd5fc40f385e1"
SCORE_SHA256 = "6f5ea31f13181c63306e818764a7281aa04ae194a4a71b052ae5d59fcc8ed640"
SCORE_TEST_SHA256 = "f68a39645b03789dfd9b5e36f916e7128f586e72f8f1c5b58f570a79ee6202e6"
EXACT_PROTOCOL_NAME = "equal_ordered_joint_outer_protocol.json"
EXACT_MANIFEST_NAME = "equal_ordered_joint_outer_protocol_fold_manifest.csv"
EXACT_SHARD_DIR_NAME = "equal_ordered_joint_outer_protocol_folds"
SPATIAL_PROTOCOL_NAME = "equal_ordered_joint_spatial_protocol.json"
SPATIAL_INVENTORY_NAME = "equal_ordered_joint_spatial_protocol_data_inventory.csv"
SPATIAL_SHARD_DIR_NAME = "equal_ordered_joint_spatial_protocol_folds"
REGION_MANIFEST_NAME = "region_out.json"
PAD_MANIFEST_NAME = "pad_out.json"

OUTER_EXACT_REPEATS = 2
OUTER_FOLDS = 5
INNER_FOLDS = 4
MODES = ("exact", "region")
H_VALUES_FT = (5_000.0, 15_000.0, 30_000.0)
LAPLACIAN_VALUES = (0.3, 3.0)
# Iteration order is also the frozen tie break: shorter h, then stronger lambda.
GRID = tuple((h, lam) for h in H_VALUES_FT for lam in (3.0, 0.3))
PREDICTION_ARRAYS = (
    "well_index",
    "row_index",
    "base_prediction",
    "joint_prediction",
    "candidate_prediction",
    "field_confidence",
    "field_delta_without_prefix_bias",
    "prefix_bias_delta",
)
INFERENCE_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TRAINING_COLUMNS = (*INFERENCE_COLUMNS, "TVT")
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "psutil",
)
RUNTIME_SOURCE_FILES = (
    "geosteern/data.py",
    "geosteern/features.py",
    "geosteern/model.py",
    "research/interval_gate.py",
    "research/ordered_transport.py",
    "research/repeated_group_gate.py",
    "research/spatial_split.py",
    "research/spatial_score_gate.py",
    "research/structural_field.py",
    "research/test_structural_field.py",
    "research/structural_field_gate.py",
    "research/test_structural_field_gate.py",
    "research/structural_field_score.py",
    "research/test_structural_field_score.py",
)
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_DRAWS = 4_000

JACKKNIFE_DEFINITION = (
    "At each target raw suffix row, each of the four winning-cell inner models "
    "contributes the central numpy.gradient(delta0, MD, edge_order=1) derivative "
    "of field_delta_without_prefix_bias_tvt only where its persistent support_mask "
    "and both finite-difference neighbors are supported (one-sided endpoints use "
    "their two points). With at least three values, tau=1.4826*median(abs(v-"
    "median(v))); otherwise cJ=0. Sigma is the final all-outer-train selected "
    "model derivative_residual_scale floored at 1e-8. cJ=1/(1+(tau/sigma)^2), "
    "and final c=final core confidence*cJ. Prefix-bias derivative is excluded."
)
INNER_JACKKNIFE_DEFINITION = (
    "For heldout inner fold j, the primary proposal is the leave-j field model and "
    "the other three derivative-dispersion proposals are leave-{j,k} models for "
    "each k!=j. All four exclude j labels. The unordered leave-two model is shared "
    "only for its identical exclusion pair. Sigma is the leave-j model derivative "
    "residual scale. Thus theta/grid selection uses core*cJ under the same formula."
)


class GateError(RuntimeError):
    """Fail-closed protocol, lineage, role, or artifact error."""


class GateHold(GateError):
    """A measured condition requires HOLD rather than silent approximation."""


@dataclass(frozen=True)
class ArtifactDescriptor:
    scope: str
    name: str
    size_bytes: int
    byte_sha256: str
    logical_sha256: str | None = None


@dataclass(frozen=True)
class GateAudit:
    protocol: dict[str, Any]
    protocol_path: Path
    protocol_sha256: str
    data_dir: Path
    results_dir: Path
    manifest_dir: Path
    exact_manifest: pd.DataFrame
    spatial_inventory: pd.DataFrame
    region_manifest: dict[str, Any]


@dataclass(frozen=True)
class IncumbentSuffix:
    row_index: NDArray[np.int64]
    base: NDArray[np.float64]
    joint: NDArray[np.float64]


@dataclass
class WellPath:
    well_id: str
    md: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    tvt_input: NDArray[np.float64]
    base_full: NDArray[np.float64]
    joint_full: NDArray[np.float64]
    suffix_index: NDArray[np.int64]
    truth: NDArray[np.float64] | None


@dataclass(frozen=True)
class GridFit:
    h_ft: float
    laplacian: float
    theta_field: float
    theta_bias: float
    objective: float
    inner_models: tuple[field_core.StructuralFieldModel, ...]


@dataclass(frozen=True)
class _RawBaseRecord:
    well_id: str
    path: str
    row_index: NDArray[np.int64]
    raw_delta: NDArray[np.float64]
    target_delta: NDArray[np.float64]
    anchor_tvt: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _logical_array_hash(arrays: Mapping[str, NDArray[Any]]) -> str:
    return incumbent_exact._logical_array_hash(dict(arrays))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    incumbent_exact._atomic_write_json(path, dict(payload))


def _atomic_write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    incumbent_exact._atomic_write_npz(path, dict(arrays))


def _write_sidecar(path: Path) -> Path:
    sidecar = _sha_sidecar(path)
    if sidecar.exists():
        raise GateError(f"refusing to overwrite hash sidecar: {sidecar}")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    return sidecar


def _read_sidecar(path: Path) -> str:
    sidecar = _sha_sidecar(path)
    if not sidecar.is_file():
        raise GateError(f"missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name or not _is_sha256(fields[0]):
        raise GateError(f"malformed SHA-256 sidecar: {sidecar}")
    return fields[0]


def _safe_basename(root: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise GateError(f"unsafe artifact basename: {name!r}")
    resolved = (root / name).resolve()
    if resolved.parent != root.resolve():
        raise GateError(f"artifact escaped its declared scope: {name!r}")
    return resolved


def _scope_root(scope: str, results_dir: Path, manifest_dir: Path) -> Path:
    roots = {
        "results": results_dir,
        "exact_shards": results_dir / EXACT_SHARD_DIR_NAME,
        "spatial_shards": results_dir / SPATIAL_SHARD_DIR_NAME,
        "spatial_manifests": manifest_dir,
    }
    try:
        return roots[scope].resolve()
    except KeyError as exc:
        raise GateError(f"unknown artifact scope: {scope}") from exc


def _descriptor(path: Path, scope: str, logical: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"required artifact is missing: {path}")
    if logical is not None and not _is_sha256(logical):
        raise GateError(f"invalid logical digest for {path.name}")
    return asdict(
        ArtifactDescriptor(
            scope=scope,
            name=path.name,
            size_bytes=int(path.stat().st_size),
            byte_sha256=sha256_file(path),
            logical_sha256=logical,
        )
    )


def _validate_descriptor(
    raw: Mapping[str, Any], results_dir: Path, manifest_dir: Path
) -> Path:
    required = {"scope", "name", "size_bytes", "byte_sha256", "logical_sha256"}
    if set(raw) != required:
        raise GateError("artifact descriptor schema drift")
    root = _scope_root(str(raw["scope"]), results_dir, manifest_dir)
    path = _safe_basename(root, str(raw["name"]))
    if not path.is_file():
        raise GateError(f"inventoried artifact is missing: {path}")
    if int(raw["size_bytes"]) != path.stat().st_size:
        raise GateError(f"artifact size drift: {path.name}")
    if not _is_sha256(raw["byte_sha256"]) or sha256_file(path) != raw["byte_sha256"]:
        raise GateError(f"artifact byte hash drift: {path.name}")
    logical = raw["logical_sha256"]
    if logical is not None and not _is_sha256(logical):
        raise GateError(f"artifact logical hash is invalid: {path.name}")
    return path


def _contains_scoring_or_truth(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if "truth" in normalized or normalized.endswith(("_sse", "_rmse")):
                return True
            if _contains_scoring_or_truth(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_scoring_or_truth(item) for item in value)
    return False


def frozen_grid() -> tuple[dict[str, float], ...]:
    return tuple({"h_ft": h, "laplacian": lam} for h, lam in GRID)


def field_config(h_ft: float, laplacian: float) -> field_core.FieldConfig:
    if (float(h_ft), float(laplacian)) not in GRID:
        raise GateError("configuration is outside the frozen six-cell grid")
    return field_core.FieldConfig(
        resample_step_md=100.0,
        inducing_cell_ft=float(h_ft) / 2.0,
        support_length_ft=float(h_ft),
        graph_neighbors=6,
        graph_max_edge_ft=1.5 * float(h_ft),
        interpolation_neighbors=6,
        laplacian_strength=float(laplacian),
        circulation_strength=0.1,
        ridge_strength=1.0e-6,
        huber_delta=1.5,
        discontinuity_mad_threshold=4.0,
        min_effective_wells=1.5,
        min_directional_observability=0.05,
        max_distinct_support_wells=16,
        max_support_neighbors=4_096,
        blend_alpha=1.0,
    )


def _assert_no_coarsening(model: field_core.StructuralFieldModel) -> None:
    requested = float(model.config.inducing_cell_ft)
    actual = float(model.diagnostics.actual_inducing_cell_ft)
    if not math.isclose(requested, actual, rel_tol=0.0, abs_tol=1.0e-9):
        raise GateHold(
            "inducing-cell coarsening occurred; the frozen gate requires STOP/HOLD"
        )


def solve_theta(
    target_minus_joint: NDArray[np.float64],
    field_feature: NDArray[np.float64],
    bias_feature: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Exact two-parameter least squares on 0 <= bias <= field <= 1."""

    residual = np.asarray(target_minus_joint, dtype=float)
    first = np.asarray(field_feature, dtype=float)
    second = np.asarray(bias_feature, dtype=float)
    if not (residual.ndim == first.ndim == second.ndim == 1):
        raise GateError("theta inputs must be one-dimensional")
    if not (len(residual) == len(first) == len(second)) or len(residual) == 0:
        raise GateError("theta inputs have invalid lengths")
    finite = np.isfinite(residual) & np.isfinite(first) & np.isfinite(second)
    if not finite.all():
        raise GateError("theta inputs must be finite")

    design = np.column_stack((first, second))

    def objective(theta_field: float, theta_bias: float) -> float:
        error = residual - design @ np.array([theta_field, theta_bias])
        return float(error @ error)

    candidates: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    gram = design.T @ design
    rhs = design.T @ residual
    try:
        unconstrained = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        unconstrained = np.linalg.lstsq(design, residual, rcond=None)[0]
    a, b = map(float, unconstrained)
    if 0.0 <= b <= a <= 1.0:
        candidates.append((a, b))
    denom = float(first @ first)
    if denom > 0.0:
        candidates.append((float(np.clip(first @ residual / denom, 0.0, 1.0)), 0.0))
    denom = float(second @ second)
    if denom > 0.0:
        candidates.append(
            (1.0, float(np.clip(second @ (residual - first) / denom, 0.0, 1.0)))
        )
    diagonal = first + second
    denom = float(diagonal @ diagonal)
    if denom > 0.0:
        value = float(np.clip(diagonal @ residual / denom, 0.0, 1.0))
        candidates.append((value, value))
    best = min(
        candidates,
        key=lambda theta: (objective(*theta), theta[0], theta[1]),
    )
    return float(best[0]), float(best[1]), objective(*best)


def _choose_grid(candidates: Sequence[GridFit]) -> GridFit:
    if {(item.h_ft, item.laplacian) for item in candidates} != set(GRID):
        raise GateError("grid selection did not evaluate the frozen six cells")
    return min(
        candidates, key=lambda item: (item.objective, item.h_ft, -item.laplacian)
    )


def _pooled_gain(
    joint_sse: NDArray[np.float64],
    candidate_sse: NDArray[np.float64],
    rows: NDArray[np.float64],
    multiplicity: NDArray[np.float64],
) -> float:
    count = float(rows @ multiplicity)
    if count <= 0.0:
        raise GateError("bootstrap draw has no rows")
    joint = math.sqrt(float(joint_sse @ multiplicity) / count)
    candidate = math.sqrt(float(candidate_sse @ multiplicity) / count)
    return joint - candidate


def exact_group_bootstrap(rows: pd.DataFrame) -> dict[str, float | int]:
    """Frozen repeat-coupled exact-profile bootstrap from per-well SSE rows."""

    required = {
        "repeat",
        "typewell_profile_hash",
        "n_rows",
        "joint_sse",
        "candidate_sse",
    }
    if not required.issubset(rows.columns):
        raise GateError("exact bootstrap input schema is incomplete")
    repeats = sorted(rows["repeat"].astype(int).unique().tolist())
    if repeats != [0, 1]:
        raise GateError("exact bootstrap requires both repeats")
    groups = sorted(rows["typewell_profile_hash"].astype(str).unique())
    contributions = {}
    for repeat in repeats:
        frame = rows.loc[rows["repeat"] == repeat]
        grouped = frame.groupby("typewell_profile_hash", sort=True)[
            ["n_rows", "joint_sse", "candidate_sse"]
        ].sum()
        grouped = grouped.reindex(groups, fill_value=0.0)
        contributions[repeat] = grouped
    ones = np.ones(len(groups), dtype=float)
    point = float(
        np.mean(
            [
                _pooled_gain(
                    contributions[repeat]["joint_sse"].to_numpy(dtype=float),
                    contributions[repeat]["candidate_sse"].to_numpy(dtype=float),
                    contributions[repeat]["n_rows"].to_numpy(dtype=float),
                    ones,
                )
                for repeat in repeats
            ]
        )
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        multiplicity = np.bincount(
            rng.integers(0, len(groups), size=len(groups)), minlength=len(groups)
        ).astype(float)
        draws[draw] = np.mean(
            [
                _pooled_gain(
                    contributions[repeat]["joint_sse"].to_numpy(dtype=float),
                    contributions[repeat]["candidate_sse"].to_numpy(dtype=float),
                    contributions[repeat]["n_rows"].to_numpy(dtype=float),
                    multiplicity,
                )
                for repeat in repeats
            ]
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "point_gain_ft": point,
        "ci95_low_ft": float(np.quantile(draws, 0.025)),
        "ci95_high_ft": float(np.quantile(draws, 0.975)),
    }


def exhaustive_region_bootstrap(rows: pd.DataFrame) -> dict[str, float | int]:
    """Exactly enumerate all 5^5 ordered region-fold resamples."""

    required = {"fold", "n_rows", "joint_sse", "candidate_sse"}
    if not required.issubset(rows.columns):
        raise GateError("region bootstrap input schema is incomplete")
    grouped = rows.groupby("fold", sort=True)[
        ["n_rows", "joint_sse", "candidate_sse"]
    ].sum()
    if grouped.index.astype(int).tolist() != list(range(OUTER_FOLDS)):
        raise GateError("region bootstrap requires exactly folds 0..4")
    row_count = grouped["n_rows"].to_numpy(dtype=float)
    joint = grouped["joint_sse"].to_numpy(dtype=float)
    candidate = grouped["candidate_sse"].to_numpy(dtype=float)
    ones = np.ones(OUTER_FOLDS, dtype=float)
    point = _pooled_gain(joint, candidate, row_count, ones)
    values = np.empty(OUTER_FOLDS**OUTER_FOLDS, dtype=float)
    for index, sample in enumerate(
        itertools.product(range(OUTER_FOLDS), repeat=OUTER_FOLDS)
    ):
        multiplicity = np.bincount(sample, minlength=OUTER_FOLDS).astype(float)
        values[index] = _pooled_gain(joint, candidate, row_count, multiplicity)
    return {
        "draws": OUTER_FOLDS**OUTER_FOLDS,
        "point_gain_ft": float(point),
        "ci95_low_ft": float(np.quantile(values, 0.025)),
        "ci95_high_ft": float(np.quantile(values, 0.975)),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def _verify_incumbent_prediction(
    shard_path: Path,
    expected_status: str,
    expected_identity: Mapping[str, int | str],
) -> tuple[dict[str, Any], Path, str]:
    shard = _read_json(shard_path, "incumbent shard")
    if shard.get("status") != expected_status:
        raise GateError(f"incumbent shard status drift: {shard_path.name}")
    if _contains_scoring_or_truth(shard):
        raise GateError(f"incumbent shard is not metric-silent: {shard_path.name}")
    for key, value in expected_identity.items():
        if shard.get(key) != value:
            raise GateError(f"incumbent shard identity drift: {shard_path.name}:{key}")
    if _read_sidecar(shard_path) != sha256_file(shard_path):
        raise GateError(f"incumbent shard sidecar drift: {shard_path.name}")
    prediction_name = str(shard.get("prediction_file", ""))
    prediction_path = _safe_basename(shard_path.parent, prediction_name)
    if prediction_path.with_suffix(".json").name != shard_path.name:
        raise GateError("incumbent prediction basename does not match its shard")
    if sha256_file(prediction_path) != shard.get("prediction_sha256"):
        raise GateError(f"incumbent prediction byte drift: {prediction_path.name}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "well_index",
            "row_index",
            "base_prediction",
            "typewell_prediction",
            "ordered_prediction",
            "joint_prediction",
        }:
            raise GateError(f"incumbent NPZ schema drift: {prediction_path.name}")
        arrays = {name: archive[name].copy() for name in archive.files}
    if len({len(value) for value in arrays.values()}) != 1:
        raise GateError(f"incumbent NPZ lengths drift: {prediction_path.name}")
    if len(arrays["row_index"]) != int(shard.get("prediction_rows", -1)):
        raise GateError(f"incumbent NPZ row count drift: {prediction_path.name}")
    logical = _logical_array_hash(arrays)
    if logical != shard.get("prediction_logical_sha256"):
        raise GateError(f"incumbent NPZ logical drift: {prediction_path.name}")
    return shard, prediction_path, logical


def _incumbent_inventory(results_dir: Path, manifest_dir: Path) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    manifest_dir = manifest_dir.resolve()
    exact_protocol_path = _safe_basename(results_dir, EXACT_PROTOCOL_NAME)
    spatial_protocol_path = _safe_basename(results_dir, SPATIAL_PROTOCOL_NAME)
    exact_protocol = _read_json(exact_protocol_path, "exact incumbent protocol")
    spatial_protocol = _read_json(spatial_protocol_path, "spatial incumbent protocol")
    if _read_sidecar(exact_protocol_path) != sha256_file(exact_protocol_path):
        raise GateError("exact incumbent protocol sidecar drift")
    if _read_sidecar(spatial_protocol_path) != sha256_file(spatial_protocol_path):
        raise GateError("spatial incumbent protocol sidecar drift")

    # Re-run each incumbent's own frozen audit before trusting any prediction
    # shard.  These validators bind semantic fold membership, not just bytes.
    try:
        exact_audit = incumbent_exact.audit_protocol(
            exact_protocol_path, verify_data=False
        )
        spatial_audit = incumbent_spatial.audit_protocol(
            spatial_protocol_path, verify_data=False
        )
    except incumbent_exact.ProtocolError as exc:
        raise GateError("frozen incumbent protocol audit failed") from exc

    exact_manifest_path = _safe_basename(results_dir, EXACT_MANIFEST_NAME)
    if exact_audit.manifest_path.resolve() != exact_manifest_path:
        raise GateError("exact incumbent audit resolved a different manifest")
    exact_manifest = pd.read_csv(exact_manifest_path)
    incumbent_exact._validate_manifest(exact_manifest, exact_protocol)
    # The parent exact protocol froze the literal CSV bytes as its manifest
    # identity; retain that exact interpretation rather than reserializing.
    exact_manifest_logical = sha256_file(exact_manifest_path)
    if exact_manifest_logical != exact_protocol.get("manifest", {}).get("sha256"):
        raise GateError("exact incumbent manifest byte drift")

    spatial_inventory_path = _safe_basename(results_dir, SPATIAL_INVENTORY_NAME)
    if spatial_audit.inventory_path.resolve() != spatial_inventory_path:
        raise GateError("spatial incumbent audit resolved a different inventory")
    spatial_inventory = incumbent_spatial._validate_inventory(
        pd.read_csv(spatial_inventory_path)
    )
    if sha256_file(spatial_inventory_path) != spatial_protocol.get("data", {}).get(
        "inventory_sha256"
    ):
        raise GateError("spatial incumbent data inventory drift")
    region_manifest_path = _safe_basename(manifest_dir, REGION_MANIFEST_NAME)
    pad_manifest_path = _safe_basename(manifest_dir, PAD_MANIFEST_NAME)
    expected_manifest_paths = {
        "region_out": region_manifest_path,
        "pad_out": pad_manifest_path,
    }
    if {
        mode: path.resolve() for mode, path in spatial_audit.manifest_paths.items()
    } != expected_manifest_paths:
        raise GateError("spatial incumbent audit resolved stale manifest paths")
    region_manifest = _read_json(region_manifest_path, "region manifest")
    pad_manifest = _read_json(pad_manifest_path, "pad manifest")
    incumbent_spatial._validate_spatial_manifest(
        region_manifest, "region_out", spatial_inventory
    )
    incumbent_spatial._validate_spatial_manifest(
        pad_manifest, "pad_out", spatial_inventory
    )

    exact_shard_root = _scope_root("exact_shards", results_dir, manifest_dir)
    exact_shards = []
    for repeat in range(OUTER_EXACT_REPEATS):
        for fold in range(OUTER_FOLDS):
            shard_path = _safe_basename(
                exact_shard_root, f"repeat_{repeat}_fold_{fold}.json"
            )
            shard, prediction_path, logical = _verify_incumbent_prediction(
                shard_path,
                "MEASURE_ONLY_PREDICTIONS_SEALED_TRUTH_UNREAD",
                {"repeat": repeat, "outer_fold": fold},
            )
            try:
                incumbent_exact._validate_shard(
                    shard, exact_audit, repeat, fold, shard_path
                )
            except incumbent_exact.ProtocolError as exc:
                raise GateError(
                    "exact incumbent shard failed frozen membership audit: "
                    f"repeat={repeat}, fold={fold}"
                ) from exc
            exact_shards.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "metadata": _descriptor(shard_path, "exact_shards"),
                    "metadata_sidecar": _descriptor(
                        _sha_sidecar(shard_path), "exact_shards"
                    ),
                    "prediction": _descriptor(prediction_path, "exact_shards", logical),
                }
            )

    spatial_shard_root = _scope_root("spatial_shards", results_dir, manifest_dir)
    region_shards = []
    for fold in range(OUTER_FOLDS):
        shard_path = _safe_basename(spatial_shard_root, f"region_out_fold_{fold}.json")
        shard, prediction_path, logical = _verify_incumbent_prediction(
            shard_path,
            "MEASURE_ONLY_SPATIAL_PREDICTIONS_SEALED_TRUTH_UNREAD",
            {"mode": "region_out", "fold": fold},
        )
        try:
            incumbent_spatial._validate_shard(
                shard, spatial_audit, "region_out", fold, shard_path
            )
        except incumbent_exact.ProtocolError as exc:
            raise GateError(
                f"region incumbent shard failed frozen membership audit: fold={fold}"
            ) from exc
        region_shards.append(
            {
                "fold": fold,
                "metadata": _descriptor(shard_path, "spatial_shards"),
                "metadata_sidecar": _descriptor(
                    _sha_sidecar(shard_path), "spatial_shards"
                ),
                "prediction": _descriptor(prediction_path, "spatial_shards", logical),
            }
        )

    payload = {
        "exact": {
            "protocol": _descriptor(exact_protocol_path, "results"),
            "protocol_sidecar": _descriptor(
                _sha_sidecar(exact_protocol_path), "results"
            ),
            "manifest": _descriptor(
                exact_manifest_path, "results", exact_manifest_logical
            ),
            "shards": exact_shards,
        },
        "region": {
            "protocol": _descriptor(spatial_protocol_path, "results"),
            "protocol_sidecar": _descriptor(
                _sha_sidecar(spatial_protocol_path), "results"
            ),
            "data_inventory": _descriptor(spatial_inventory_path, "results"),
            "manifest": _descriptor(
                region_manifest_path,
                "spatial_manifests",
                str(region_manifest["manifest_sha256"]),
            ),
            "pad_manifest": _descriptor(
                pad_manifest_path,
                "spatial_manifests",
                str(pad_manifest["manifest_sha256"]),
            ),
            "shards": region_shards,
        },
    }
    return {**payload, "inventory_sha256": _canonical_digest(payload)}


def _evaluation_contract() -> dict[str, Any]:
    return {
        "status_ceiling": STATUS_CEILING,
        "interval_coefficient": 1.0,
        "candidate_formula": (
            "sealed_joint + final_core_confidence*jackknife_confidence*"
            "(theta_field*field_delta_without_prefix_bias + "
            "theta_bias*prefix_bias_delta)"
        ),
        "theta_constraints": "0 <= theta_bias <= theta_field <= 1",
        "inner_group": "exact typewell profile equality",
        "inner_folds": INNER_FOLDS,
        "inner_incumbent": (
            "strict ten-model base cross-fit: four leave-one-fold-out models generate "
            "each heldout path, while six leave-two-fold-out models generate calibration-"
            "fold paths excluding both heldout j and calibration fold k; for each j, "
            "base shrink and all path-dependent typewell/ordered/joint calibrators use "
            "only those j-excluding calibration predictions before application to j"
        ),
        "field_fit_target": (
            "fitting-role labeled TRUE TVT+Z derivatives; cross-fitted base is used "
            "only as the heldout/target prediction policy"
        ),
        "outer_incumbent": "already sealed fold-local incumbent joint artifact",
        "jackknife_confidence": JACKKNIFE_DEFINITION,
        "inner_jackknife_confidence": INNER_JACKKNIFE_DEFINITION,
        "grid": list(frozen_grid()),
        "tie_break": "shorter h, then stronger Laplacian",
        "exact_primary_gates": {
            "mean_repeat_pooled_gain_ft_at_least": 1.0,
            "both_repeat_gains_positive": True,
            "exact_group_bootstrap_ci95_low_positive": True,
            "paired_median_well_gain_ft_at_least": 0.5,
            "top10_positive_sse_removal_gain_positive": True,
            "p90_point_worsening_ft_at_most": 0.2,
            "p90_one_sided_ci_worsening_ft_below": 0.5,
        },
        "region_primary_gates": {
            "pooled_gain_ft_at_least": 0.75,
            "minimum_fold_gain_ft_at_least": 0.2,
            "exhaustive_cluster_bootstrap_ci95_low_positive": True,
            "top10_positive_sse_removal_gain_positive": True,
            "p90_point_worsening_ft_at_most": 0.2,
            "p90_one_sided_ci_worsening_ft_below": 0.5,
            "each_region_p90_worsening_ft_at_most": 1.0,
        },
        "descriptive_non_gate_diagnostics": {
            "supported_fraction_overall": True,
            "supported_fraction_by_region": True,
        },
        "prediction_barrier": (
            "every required JSON/NPZ shard must pass byte, logical, membership, and "
            "schema audit before aggregate opens any validation suffix TVT"
        ),
        "bootstrap": {
            "exact": {
                "unit": "exact typewell profile group",
                "repeat_coupling": (
                    "one sampled group multiplicity is applied to all wells in both "
                    "repeats; each draw averages repeat-wise pooled RMSE gains"
                ),
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "interval": "two-sided percentile 95%",
            },
            "region": {
                "unit": "sealed region fold",
                "method": "exhaustive ordered resampling of five folds with replacement",
                "draws": 5**5,
                "interval": "two-sided percentile 95%",
            },
        },
        "p90_definition": {
            "unit": "well",
            "well_value": "full-suffix row RMSE",
            "quantile": "numpy.quantile(q=0.9, method='linear')",
            "point_worsening": "candidate well-RMSE p90 minus sealed-joint well-RMSE p90",
            "one_sided_bound": (
                "95th percentile of bootstrap point-worsening values using the same "
                "mode-specific resampling multiplicities"
            ),
        },
        "runtime_acceptance": {
            "field_wall_seconds_at_most": 1_800.0,
            "field_peak_rss_gib_at_most": 8.0,
            "total_wall_seconds_at_most": 3_600.0,
            "total_peak_rss_gib_at_most": 12.0,
            "extrapolated_two_worker_fifteen_fold_seconds_at_most": 28_800.0,
            "extrapolation": "largest measured fold wall time multiplied by ceil(15/2)",
            "caps_solver_or_coarsening": "forbidden; inducing coarsening is STOP",
        },
    }


def _packages() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def _source_hashes() -> dict[str, str]:
    paths = {name: ROOT / name for name in RUNTIME_SOURCE_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise GateError(f"runtime source is missing: {missing}")
    result = {name: sha256_file(path) for name, path in paths.items()}
    if (
        result["research/structural_field.py"] != CORE_SHA256
        or result["research/test_structural_field.py"] != CORE_TEST_SHA256
    ):
        raise GateError("audited structural-field core identity drift")
    if (
        result["research/structural_field_score.py"] != SCORE_SHA256
        or result["research/test_structural_field_score.py"] != SCORE_TEST_SHA256
    ):
        raise GateError("audited structural-field scorer identity drift")
    return result


def freeze_protocol(
    data_dir: Path,
    results_dir: Path,
    manifest_dir: Path,
    protocol_path: Path,
) -> tuple[Path, Path]:
    """Freeze all choices and fresh incumbent identities without opening truth."""

    data_dir = data_dir.resolve()
    results_dir = results_dir.resolve()
    manifest_dir = manifest_dir.resolve()
    protocol_path = protocol_path.resolve()
    sidecar = _sha_sidecar(protocol_path)
    if protocol_path.exists() or sidecar.exists():
        raise GateError(f"refusing to overwrite frozen protocol: {protocol_path}")
    gate_test_path = ROOT / "research/test_structural_field_gate.py"
    if not gate_test_path.is_file():
        raise GateError("gate tests must exist and be bound before protocol freeze")
    inventory = _incumbent_inventory(results_dir, manifest_dir)
    spatial_inventory = incumbent_spatial._validate_inventory(
        pd.read_csv(_safe_basename(results_dir, SPATIAL_INVENTORY_NAME))
    )
    _audit_data_files(spatial_inventory, data_dir)
    benchmark_work_proxy = _compute_work_proxy(data_dir, results_dir, manifest_dir)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_FIELD_SCORING_MEASURE_ONLY",
        "status_ceiling": STATUS_CEILING,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
            "manifest_dir": str(manifest_dir),
        },
        "source_sha256": _source_hashes(),
        "packages": _packages(),
        "field_config_contract": {
            "resample_step_md": 100.0,
            "inducing_cell": "h/2",
            "support_length": "h",
            "graph_max_edge": "1.5*h",
            "graph_neighbors": 6,
            "interpolation_neighbors": 6,
            "circulation_strength": 0.1,
            "ridge_strength": 1.0e-6,
            "huber_delta": 1.5,
            "discontinuity_mad_threshold": 4.0,
            "min_effective_wells": 1.5,
            "min_directional_observability": 0.05,
            "max_distinct_support_wells": 16,
            "max_support_neighbors": 4_096,
            "coarsening": "STOP",
        },
        "evaluation": _evaluation_contract(),
        "benchmark_work_proxy": benchmark_work_proxy,
        "incumbent_pretruth_inventory": inventory,
        "notes": [
            "Freeze and run validation reads use MD/X/Y/Z/TVT_input only.",
            "Training TVT may be opened only for an active outer-training role.",
            "Formation surfaces, geology labels, and images are never inputs.",
            "Artifact entries bind verified basenames under declared roots, not stale paths.",
            "No external confirmation is present; production OPEN is impossible here.",
        ],
    }
    _atomic_write_json(protocol_path, protocol)
    _write_sidecar(protocol_path)
    return protocol_path, sidecar


def _validate_inventory_payload(
    inventory: Mapping[str, Any], results_dir: Path, manifest_dir: Path
) -> None:
    payload = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if inventory.get("inventory_sha256") != _canonical_digest(payload):
        raise GateError("incumbent pretruth inventory digest drift")
    if set(payload) != {"exact", "region"}:
        raise GateError("incumbent inventory mode schema drift")
    exact = payload["exact"]
    region = payload["region"]
    if len(exact.get("shards", [])) != 10 or len(region.get("shards", [])) != 5:
        raise GateError("incumbent inventory shard count drift")
    descriptors: list[Mapping[str, Any]] = []
    for item in (exact["protocol"], exact["protocol_sidecar"], exact["manifest"]):
        descriptors.append(item)
    for item in (
        region["protocol"],
        region["protocol_sidecar"],
        region["data_inventory"],
        region["manifest"],
        region["pad_manifest"],
    ):
        descriptors.append(item)
    identities = []
    for expected, item in zip(
        [(r, f) for r in range(2) for f in range(5)],
        exact["shards"],
        strict=True,
    ):
        if (int(item.get("repeat", -1)), int(item.get("fold", -1))) != expected:
            raise GateError("exact incumbent shard inventory identity drift")
        descriptors.extend(
            (item["metadata"], item["metadata_sidecar"], item["prediction"])
        )
    for fold, item in enumerate(region["shards"]):
        if int(item.get("fold", -1)) != fold:
            raise GateError("region incumbent shard inventory identity drift")
        descriptors.extend(
            (item["metadata"], item["metadata_sidecar"], item["prediction"])
        )
    for descriptor in descriptors:
        path = _validate_descriptor(descriptor, results_dir, manifest_dir)
        identities.append(str(path))
    if len(identities) != len(set(identities)):
        raise GateError("incumbent inventory aliases artifact paths")


def _audit_data_files(inventory: pd.DataFrame, data_dir: Path) -> None:
    train_root = (data_dir / "train").resolve()
    for row in inventory.itertuples(index=False):
        horizontal = _safe_basename(train_root, str(row.horizontal_file))
        typewell = _safe_basename(train_root, str(row.typewell_file))
        if sha256_file(horizontal) != str(row.horizontal_sha256):
            raise GateError(f"horizontal CSV byte drift: {horizontal.name}")
        if sha256_file(typewell) != str(row.typewell_sha256):
            raise GateError(f"typewell CSV byte drift: {typewell.name}")


def _work_proxy_well_stats(
    data_dir: Path, inventory: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    train_root = (data_dir / "train").resolve()
    stats: dict[str, dict[str, Any]] = {}
    base_config = field_config(5_000.0, 3.0)
    for row in inventory.itertuples(index=False):
        well_id = str(row.well)
        path = _safe_basename(train_root, str(row.horizontal_file))
        frame = _read_well_csv(path, "benchmark")
        md = frame["MD"].to_numpy(dtype=float)
        x = frame["X"].to_numpy(dtype=float)
        y = frame["Y"].to_numpy(dtype=float)
        z = frame["Z"].to_numpy(dtype=float)
        tvt_input = frame["TVT_input"].to_numpy(dtype=float)
        field_core._validate_md(md)
        knots = field_core._resampling_knots(
            md,
            base_config.resample_step_md,
            base_config.max_resampled_intervals_per_well,
        )
        xk = np.interp(knots, md, x)
        yk = np.interp(knots, md, y)
        zk = np.interp(knots, md, z)
        delta_md = np.diff(knots)
        u = np.column_stack((np.diff(xk) / delta_md, np.diff(yk) / delta_md))
        lateral = (
            np.abs(np.diff(zk) / delta_md) <= base_config.lateral_max_abs_dz_dmd
        ) & (np.linalg.norm(u, axis=1) >= base_config.min_horizontal_speed)
        midpoint = np.column_stack(
            ((xk[:-1] + xk[1:]) / 2.0, (yk[:-1] + yk[1:]) / 2.0)
        )[lateral]
        cells = {}
        for h_ft in H_VALUES_FT:
            cell = h_ft / 2.0
            keys = np.floor(midpoint / cell).astype(np.int64)
            cells[str(int(h_ft))] = {
                (int(first), int(second)) for first, second in keys
            }
        prefix = _known_prefix(tvt_input)
        anchor_md = float(md[prefix - 1])
        prediction_knots = field_core._prediction_knots(md, anchor_md, base_config)
        prediction_prefix = int(
            np.searchsorted(prediction_knots, anchor_md, side="right")
        )
        bias_start = int(
            np.searchsorted(
                prediction_knots,
                anchor_md - base_config.prefix_bias_window_md,
                side="left",
            )
        )
        bias_midpoints = max(0, prediction_prefix - 1 - bias_start)
        stats[well_id] = {
            "derivative_observations": int(lateral.sum()),
            "support_queries": int(len(prediction_knots) + bias_midpoints),
            "raw_rows": int(len(frame)),
            "static_rows_upper": int(math.ceil(len(frame) / 8.0)),
            "cells": cells,
        }
    return stats


def _compute_work_proxy(
    data_dir: Path, results_dir: Path, manifest_dir: Path
) -> dict[str, Any]:
    exact_manifest = pd.read_csv(_safe_basename(results_dir, EXACT_MANIFEST_NAME))
    inventory = incumbent_spatial._validate_inventory(
        pd.read_csv(_safe_basename(results_dir, SPATIAL_INVENTORY_NAME))
    )
    region_manifest = _read_json(
        _safe_basename(manifest_dir, REGION_MANIFEST_NAME), "region manifest"
    )
    minimal = GateAudit(
        protocol={},
        protocol_path=Path(),
        protocol_sha256="",
        data_dir=data_dir,
        results_dir=results_dir,
        manifest_dir=manifest_dir,
        exact_manifest=exact_manifest,
        spatial_inventory=inventory,
        region_manifest=region_manifest,
    )
    stats = _work_proxy_well_stats(data_dir, inventory)
    rows = []
    for mode, repeat, fold in _all_fold_identities():
        training, validation, embargo, _ = _outer_roles(minimal, mode, repeat, fold)
        derivative_observations = sum(
            stats[well_id]["derivative_observations"] for well_id in training
        )
        node_upper_by_h: dict[str, int] = {}
        for h_ft in H_VALUES_FT:
            key = str(int(h_ft))
            union: set[tuple[int, int]] = set()
            for well_id in training:
                union.update(stats[well_id]["cells"][key])
            node_upper_by_h[key] = len(union)
        observation_work = 37 * derivative_observations
        node_work = 20 * sum(node_upper_by_h.values()) + max(node_upper_by_h.values())
        training_queries = sum(
            stats[well_id]["support_queries"] for well_id in training
        )
        validation_queries = sum(
            stats[well_id]["support_queries"] for well_id in validation
        )
        support_query_work = 24 * training_queries + 5 * validation_queries
        base_static_work = 6 * sum(
            stats[well_id]["static_rows_upper"] for well_id in training
        )
        base_path_work = 4 * sum(stats[well_id]["raw_rows"] for well_id in training)
        proxy_units = (
            observation_work
            + 100 * node_work
            + 4_096 * support_query_work
            + base_static_work
            + base_path_work
        )
        rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "training_wells": len(training),
                "validation_wells": len(validation),
                "embargo_wells": len(embargo),
                "derivative_observations": derivative_observations,
                "inducing_node_upper_by_h": node_upper_by_h,
                "support_query_count": support_query_work,
                "base_static_row_work": base_static_work,
                "base_full_path_row_work": base_path_work,
                "proxy_units": int(proxy_units),
            }
        )
    rows.sort(key=lambda item: (item["mode"], item["repeat"], item["fold"]))
    maximizing = max(
        rows,
        key=lambda item: (
            item["proxy_units"],
            item["derivative_observations"],
            sum(item["inducing_node_upper_by_h"].values()),
            item["support_query_count"],
            item["mode"] == "region",
            -item["repeat"],
            -item["fold"],
        ),
    )
    payload = {
        "definition": {
            "derivative_observation_multiplier": 37,
            "inducing_node_multiplier": (
                "20 times sum of full-training node upper bounds over h, plus max(h) "
                "for the selected final refit upper bound"
            ),
            "inducing_node_proxy_weight": 100,
            "support_query_multiplier": (
                "24 strict inner proposals per training target plus 5 final/jackknife "
                "proposals per validation target"
            ),
            "support_query_proxy_weight": 4_096,
            "strict_base_static_multiplier": 6,
            "strict_base_full_path_multiplier": 4,
            "selection": "maximum proxy_units with frozen deterministic tie break",
            "columns": list(INFERENCE_COLUMNS),
        },
        "folds": rows,
        "maximizing_identity": {
            "mode": maximizing["mode"],
            "repeat": maximizing["repeat"],
            "fold": maximizing["fold"],
        },
    }
    return {**payload, "proxy_sha256": _canonical_digest(payload)}


def _validate_work_proxy_payload(proxy: Mapping[str, Any]) -> None:
    if set(proxy) != {"definition", "folds", "maximizing_identity", "proxy_sha256"}:
        raise GateError("benchmark work-proxy top-level schema drift")
    payload = {key: value for key, value in proxy.items() if key != "proxy_sha256"}
    if proxy.get("proxy_sha256") != _canonical_digest(payload):
        raise GateError("benchmark work-proxy logical digest drift")
    identities = []
    rows = proxy.get("folds")
    if not isinstance(rows, list) or len(rows) != len(_all_fold_identities()):
        raise GateError("benchmark work-proxy fold inventory drift")
    expected_row_keys = {
        "mode",
        "repeat",
        "fold",
        "training_wells",
        "validation_wells",
        "embargo_wells",
        "derivative_observations",
        "inducing_node_upper_by_h",
        "support_query_count",
        "base_static_row_work",
        "base_full_path_row_work",
        "proxy_units",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_row_keys:
            raise GateError("benchmark work-proxy fold schema drift")
        identity = (str(row["mode"]), int(row["repeat"]), int(row["fold"]))
        identities.append(identity)
        if identity not in set(_all_fold_identities()):
            raise GateError("benchmark work-proxy contains an invalid fold")
        if set(row["inducing_node_upper_by_h"]) != {
            str(int(value)) for value in H_VALUES_FT
        }:
            raise GateError("benchmark work-proxy h-node schema drift")
        numeric = [
            row["training_wells"],
            row["validation_wells"],
            row["embargo_wells"],
            row["derivative_observations"],
            *row["inducing_node_upper_by_h"].values(),
            row["support_query_count"],
            row["base_static_row_work"],
            row["base_full_path_row_work"],
            row["proxy_units"],
        ]
        if any(int(value) < 0 or float(value) != int(value) for value in numeric):
            raise GateError("benchmark work-proxy contains an invalid count")
    if len(set(identities)) != len(identities) or set(identities) != set(
        _all_fold_identities()
    ):
        raise GateError("benchmark work-proxy identities are incomplete or aliased")
    maximum = max(
        rows,
        key=lambda item: (
            int(item["proxy_units"]),
            int(item["derivative_observations"]),
            sum(int(value) for value in item["inducing_node_upper_by_h"].values()),
            int(item["support_query_count"]),
            item["mode"] == "region",
            -int(item["repeat"]),
            -int(item["fold"]),
        ),
    )
    expected_maximum = {
        "mode": maximum["mode"],
        "repeat": maximum["repeat"],
        "fold": maximum["fold"],
    }
    if proxy.get("maximizing_identity") != expected_maximum:
        raise GateError("benchmark work-proxy maximizing identity drift")


def audit_protocol(
    protocol_path: Path,
    *,
    verify_data: bool = True,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    manifest_dir: Path | None = None,
) -> GateAudit:
    """Audit protocol, source, incumbents, manifests, and inference inventory."""

    protocol_path = protocol_path.resolve()
    expected = _read_sidecar(protocol_path)
    actual = sha256_file(protocol_path)
    if expected != actual:
        raise GateError("structural-field protocol hash drift")
    protocol = _read_json(protocol_path, "structural-field protocol")
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise GateError("unsupported structural-field protocol version")
    if (
        protocol.get("method") != METHOD
        or protocol.get("status_ceiling") != STATUS_CEILING
    ):
        raise GateError("structural-field method/status drift")
    if protocol.get("evaluation") != _evaluation_contract():
        raise GateError("frozen evaluation contract drift")
    if (
        tuple(
            (float(item["h_ft"]), float(item["laplacian"]))
            for item in protocol["evaluation"]["grid"]
        )
        != GRID
    ):
        raise GateError("frozen grid order drift")
    roots = protocol.get("roots", {})
    live_data = (data_dir or Path(str(roots.get("data_dir", "")))).resolve()
    live_results = (results_dir or Path(str(roots.get("results_dir", "")))).resolve()
    live_manifests = (
        manifest_dir or Path(str(roots.get("manifest_dir", "")))
    ).resolve()
    frozen_proxy = protocol.get("benchmark_work_proxy")
    if not isinstance(frozen_proxy, Mapping):
        raise GateError("protocol lacks a frozen benchmark work proxy")
    _validate_work_proxy_payload(frozen_proxy)
    expected_sources = _source_hashes()
    if protocol.get("source_sha256") != expected_sources:
        raise GateError("bound source identity drift")
    if protocol.get("packages") != _packages():
        raise GateError("package environment drift")
    inventory = protocol.get("incumbent_pretruth_inventory")
    if not isinstance(inventory, Mapping):
        raise GateError("protocol lacks incumbent pretruth inventory")
    _validate_inventory_payload(inventory, live_results, live_manifests)
    # Deep-validate schemas and logical hashes, not just their descriptor bytes.
    fresh = _incumbent_inventory(live_results, live_manifests)
    if fresh != inventory:
        raise GateError("fresh incumbent inventory differs from frozen inventory")
    exact_manifest = pd.read_csv(_safe_basename(live_results, EXACT_MANIFEST_NAME))
    spatial_inventory = incumbent_spatial._validate_inventory(
        pd.read_csv(_safe_basename(live_results, SPATIAL_INVENTORY_NAME))
    )
    region_manifest = _read_json(
        _safe_basename(live_manifests, REGION_MANIFEST_NAME), "region manifest"
    )
    if verify_data:
        _audit_data_files(spatial_inventory, live_data)
        live_proxy = _compute_work_proxy(live_data, live_results, live_manifests)
        if live_proxy != frozen_proxy:
            raise GateError("inference-visible benchmark work proxy drift")
    return GateAudit(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=actual,
        data_dir=live_data,
        results_dir=live_results,
        manifest_dir=live_manifests,
        exact_manifest=exact_manifest,
        spatial_inventory=spatial_inventory,
        region_manifest=region_manifest,
    )


def _outer_roles(
    audit: GateAudit, mode: str, repeat: int, fold: int
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    if mode == "exact":
        if repeat not in range(OUTER_EXACT_REPEATS) or fold not in range(OUTER_FOLDS):
            raise GateError("invalid exact outer identity")
        frame = audit.exact_manifest.loc[audit.exact_manifest["repeat"] == repeat]
        validation = sorted(
            frame.loc[frame["outer_fold"] == fold, "well"].astype(str).tolist()
        )
        training = sorted(
            frame.loc[frame["outer_fold"] != fold, "well"].astype(str).tolist()
        )
        group_by_well = dict(
            zip(
                frame["well"].astype(str),
                frame["typewell_profile_hash"].astype(str),
                strict=True,
            )
        )
        embargo_ids: list[str] = []
    elif mode == "region":
        if repeat != 0 or fold not in range(OUTER_FOLDS):
            raise GateError("region mode has repeat=0 and folds 0..4 only")
        matches = [
            row
            for row in audit.region_manifest.get("folds", [])
            if int(row.get("fold", -1)) == fold
        ]
        if len(matches) != 1:
            raise GateError("region manifest fold identity drift")
        frozen = matches[0]
        # Exact order is retained and asserted; embargo is not repurposed.
        training = [str(value) for value in frozen["training_ids"]]
        validation = [str(value) for value in frozen["validation_ids"]]
        embargo_ids = [str(value) for value in frozen["embargo_ids"]]
        embargo = set(embargo_ids)
        if set(training) & embargo or set(validation) & embargo:
            raise GateError("region embargo entered the training role")
        group_by_well = {
            str(row["well_id"]): str(row["equality_group"])
            for row in audit.region_manifest["wells"]
        }
    else:
        raise GateError(f"unsupported gate mode: {mode}")
    if set(training) & set(validation):
        raise GateError("outer training and validation roles overlap")
    train_groups = {group_by_well[wid] for wid in training}
    validation_groups = {group_by_well[wid] for wid in validation}
    if train_groups & validation_groups:
        raise GateError("exact-profile equality group crossed the outer boundary")
    return training, validation, embargo_ids, group_by_well


def _inventory_by_well(audit: GateAudit) -> pd.DataFrame:
    frame = audit.spatial_inventory.copy()
    frame["well"] = frame["well"].astype(str)
    if frame["well"].duplicated().any():
        raise GateError("data inventory contains duplicate well IDs")
    return frame.set_index("well", drop=False)


def _well_file(audit: GateAudit, well_id: str) -> Path:
    inventory = _inventory_by_well(audit)
    if well_id not in inventory.index:
        raise GateError(f"well is absent from data inventory: {well_id}")
    return _safe_basename(
        (audit.data_dir / "train").resolve(),
        str(inventory.loc[well_id, "horizontal_file"]),
    )


def _load_incumbent_suffixes(
    audit: GateAudit, mode: str, repeat: int
) -> dict[str, IncumbentSuffix]:
    inventory = audit.protocol["incumbent_pretruth_inventory"]
    if mode == "exact":
        items = [
            item
            for item in inventory["exact"]["shards"]
            if int(item["repeat"]) == repeat
        ]
    elif mode == "region":
        if repeat != 0:
            raise GateError("region incumbent has no repeated realization")
        items = list(inventory["region"]["shards"])
    else:
        raise GateError(f"unsupported incumbent mode: {mode}")
    if len(items) != OUTER_FOLDS:
        raise GateError("incumbent shard selection is incomplete")

    result: dict[str, IncumbentSuffix] = {}
    for item in items:
        metadata_path = _validate_descriptor(
            item["metadata"], audit.results_dir, audit.manifest_dir
        )
        prediction_path = _validate_descriptor(
            item["prediction"], audit.results_dir, audit.manifest_dir
        )
        shard = _read_json(metadata_path, "incumbent shard")
        with np.load(prediction_path, allow_pickle=False) as archive:
            arrays = {
                name: archive[name].copy()
                for name in (
                    "well_index",
                    "row_index",
                    "base_prediction",
                    "joint_prediction",
                )
            }
        rows = shard.get("test_wells")
        if not isinstance(rows, list):
            raise GateError("incumbent shard lacks well metadata")
        for expected_index, row in enumerate(rows):
            if int(row.get("well_index", -1)) != expected_index:
                raise GateError("incumbent well-index metadata drift")
            well_id = str(row.get("well", ""))
            selected = arrays["well_index"] == expected_index
            if int(selected.sum()) != int(row.get("n_rows", -1)):
                raise GateError("incumbent per-well row count drift")
            if well_id in result:
                raise GateError("incumbent well appears in multiple folds")
            result[well_id] = IncumbentSuffix(
                row_index=np.asarray(arrays["row_index"][selected], dtype=np.int64),
                base=np.asarray(arrays["base_prediction"][selected], dtype=np.float64),
                joint=np.asarray(
                    arrays["joint_prediction"][selected], dtype=np.float64
                ),
            )
    expected = set(_inventory_by_well(audit).index.astype(str))
    if set(result) != expected:
        raise GateError(
            "incumbent realization does not cover the frozen population once"
        )
    return result


def _read_well_csv(path: Path, role: str) -> pd.DataFrame:
    if role == "training":
        columns = TRAINING_COLUMNS
    elif role in {"validation", "benchmark"}:
        columns = INFERENCE_COLUMNS
    else:
        raise GateError(f"invalid CSV role: {role}")
    # The explicit usecols call is the on-disk truth quarantine.  Validation
    # suffix TVT and every surface/geology column are never parsed into memory.
    frame = pd.read_csv(path, usecols=list(columns))
    if tuple(frame.columns) != columns:
        frame = frame.loc[:, list(columns)]
    if frame.empty or frame[list(INFERENCE_COLUMNS[:-1])].isna().any().any():
        raise GateError(f"invalid inference coordinates: {path.name}")
    return frame


def _known_prefix(tvt_input: NDArray[np.float64]) -> int:
    missing = np.flatnonzero(np.isnan(tvt_input))
    if len(missing) == 0:
        raise GateError("TVT_input must expose a sealed suffix")
    prefix = int(missing[0])
    if prefix < 2 or not np.isfinite(tvt_input[:prefix]).all():
        raise GateError("TVT_input prefix is invalid")
    if not np.isnan(tvt_input[prefix:]).all():
        raise GateError("TVT_input missingness is not contiguous")
    return prefix


def _compose_well(
    well_id: str,
    frame: pd.DataFrame,
    suffix: IncumbentSuffix,
    role: str,
) -> WellPath:
    tvt_input = frame["TVT_input"].to_numpy(dtype=float)
    prefix = _known_prefix(tvt_input)
    expected_index = np.arange(prefix, len(frame), dtype=np.int64)
    if not np.array_equal(suffix.row_index, expected_index):
        raise GateError(f"sealed incumbent suffix is not contiguous for {well_id}")
    if not (
        len(suffix.base) == len(suffix.joint) == len(expected_index)
        and np.isfinite(suffix.base).all()
        and np.isfinite(suffix.joint).all()
    ):
        raise GateError(f"sealed incumbent values are invalid for {well_id}")
    base = tvt_input.copy()
    joint = tvt_input.copy()
    base[expected_index] = suffix.base
    joint[expected_index] = suffix.joint
    # Both policies use the exact available prefix, never an incumbent estimate.
    if not np.array_equal(base[:prefix], tvt_input[:prefix]) or not np.array_equal(
        joint[:prefix], tvt_input[:prefix]
    ):
        raise GateError("prefix-policy convention failed")
    truth: NDArray[np.float64] | None
    if role == "training":
        if "TVT" not in frame:
            raise GateError("training role lacks TVT")
        truth = frame["TVT"].to_numpy(dtype=float)
        if not np.isfinite(truth).all():
            raise GateError(f"training TVT is non-finite for {well_id}")
    else:
        if "TVT" in frame:
            raise GateError("validation role unexpectedly parsed TVT")
        truth = None
    return WellPath(
        well_id=well_id,
        md=frame["MD"].to_numpy(dtype=float),
        x=frame["X"].to_numpy(dtype=float),
        y=frame["Y"].to_numpy(dtype=float),
        z=frame["Z"].to_numpy(dtype=float),
        tvt_input=tvt_input,
        base_full=base,
        joint_full=joint,
        suffix_index=expected_index,
        truth=truth,
    )


def _load_role_wells(
    audit: GateAudit,
    well_ids: Sequence[str],
    suffixes: Mapping[str, IncumbentSuffix],
    role: str,
) -> dict[str, WellPath]:
    result = {}
    for well_id in well_ids:
        if well_id not in suffixes:
            raise GateError(f"incumbent path is missing for {well_id}")
        frame = _read_well_csv(_well_file(audit, well_id), role)
        result[well_id] = _compose_well(well_id, frame, suffixes[well_id], role)
    return result


def _inner_fold_ids(
    static_matrix: tuple[pd.DataFrame, NDArray[np.float64], NDArray[Any]],
    group_by_well: Mapping[str, str],
) -> dict[str, int]:
    x_frame, y, row_wells = static_matrix
    groups = np.asarray([group_by_well[str(wid)] for wid in row_wells])
    splitter = GroupKFold(INNER_FOLDS)
    result: dict[str, int] = {}
    for fold, (_, validation_index) in enumerate(splitter.split(x_frame, y, groups)):
        for well_id in set(map(str, row_wells[validation_index])):
            previous = result.setdefault(well_id, fold)
            if previous != fold:
                raise GateError("an exact-profile well crossed inner folds")
    if set(result) != set(map(str, row_wells)):
        raise GateError("inner fold assignment did not cover every training well")
    for group in set(group_by_well[wid] for wid in result):
        folds = {result[wid] for wid in result if group_by_well[wid] == group}
        if len(folds) != 1:
            raise GateError("an exact-profile equality group crossed inner folds")
    return result


def _raw_base_oof(
    audit: GateAudit,
    training_ids: Sequence[str],
    group_by_well: Mapping[str, str],
) -> tuple[
    dict[int, dict[str, _RawBaseRecord]],
    dict[str, int],
    dict[int, NDArray[np.float64]],
    NDArray[np.float64],
    NDArray[Any],
    dict[str, Any],
]:
    """Fit strict leave-one and leave-two base paths with no heldout alias."""

    train_files = [str(_well_file(audit, wid)) for wid in training_ids]
    training_groups = {wid: group_by_well[wid] for wid in training_ids}
    static_matrix = incumbent_exact._build_static_training_matrix(train_files)
    fold_by_well = _inner_fold_ids(static_matrix, training_groups)
    x_frame, y, row_wells = static_matrix
    row_folds = np.asarray([fold_by_well[str(wid)] for wid in row_wells], dtype=int)
    files_by_id = {incumbent_exact.well_id(path): path for path in train_files}

    def predict_records(
        model: Any, requested_ids: Sequence[str]
    ) -> dict[str, _RawBaseRecord]:
        predicted: dict[str, _RawBaseRecord] = {}
        for well_id in requested_ids:
            well = incumbent_exact.load_well(files_by_id[well_id])
            if well is None:
                raise GateError(f"could not load outer-training well {well_id}")
            feature_frame, indices, target_delta = incumbent_exact.point_frame(
                well, stride=incumbent_exact.EVALUATION_STRIDE
            )
            predicted[well_id] = _RawBaseRecord(
                well_id=well_id,
                path=files_by_id[well_id],
                row_index=np.asarray(indices, dtype=np.int64),
                raw_delta=np.asarray(
                    model.predict(feature_frame[x_frame.columns]), dtype=np.float64
                ),
                target_delta=np.asarray(target_delta, dtype=np.float64),
                anchor_tvt=float(well["tvt_prefix"][well["known"]][-1]),
            )
        return predicted

    raw_records = {fold: {} for fold in range(INNER_FOLDS)}
    raw_rows = {
        fold: np.full(len(y), np.nan, dtype=float) for fold in range(INNER_FOLDS)
    }
    leave_one_roles = []
    for heldout_fold in range(INNER_FOLDS):
        fit_mask = row_folds != heldout_fold
        heldout_mask = ~fit_mask
        fit_ids = sorted(
            well_id for well_id in training_ids if fold_by_well[well_id] != heldout_fold
        )
        predicted_ids = sorted(
            well_id for well_id in training_ids if fold_by_well[well_id] == heldout_fold
        )
        if not fit_ids or not predicted_ids or set(fit_ids) & set(predicted_ids):
            raise GateError("leave-one base role is empty or aliased")
        model = incumbent_exact.lgb.LGBMRegressor(
            **incumbent_exact.frozen_research_params()
        ).fit(x_frame.loc[fit_mask], y[fit_mask])
        raw_rows[heldout_fold][heldout_mask] = model.predict(x_frame.loc[heldout_mask])
        raw_records[heldout_fold].update(predict_records(model, predicted_ids))
        leave_one_roles.append(
            {
                "heldout_fold": heldout_fold,
                "fitting_ids_sha256": _id_digest(fit_ids),
                "predicted_ids_sha256": _id_digest(predicted_ids),
            }
        )

    pair_roles = []
    seen_pairs = set()
    for first, second in itertools.combinations(range(INNER_FOLDS), 2):
        pair = (first, second)
        if pair in seen_pairs:
            raise GateError("leave-two base role was aliased")
        seen_pairs.add(pair)
        fit_mask = (row_folds != first) & (row_folds != second)
        fit_ids = sorted(
            well_id for well_id in training_ids if fold_by_well[well_id] not in pair
        )
        first_ids = sorted(
            well_id for well_id in training_ids if fold_by_well[well_id] == first
        )
        second_ids = sorted(
            well_id for well_id in training_ids if fold_by_well[well_id] == second
        )
        if (
            not fit_ids
            or not first_ids
            or not second_ids
            or set(fit_ids) & (set(first_ids) | set(second_ids))
            or set(first_ids) & set(second_ids)
        ):
            raise GateError("leave-two base role is empty or aliased")
        model = incumbent_exact.lgb.LGBMRegressor(
            **incumbent_exact.frozen_research_params()
        ).fit(x_frame.loc[fit_mask], y[fit_mask])
        # Heldout=first calibrates on second, and heldout=second calibrates on
        # first. The same pair model excludes both roles in either direction.
        second_mask = row_folds == second
        raw_rows[first][second_mask] = model.predict(x_frame.loc[second_mask])
        second_predictions = predict_records(model, second_ids)
        if set(raw_records[first]) & set(second_predictions):
            raise GateError("leave-two prediction aliased a leave-one path")
        raw_records[first].update(second_predictions)

        first_mask = row_folds == first
        raw_rows[second][first_mask] = model.predict(x_frame.loc[first_mask])
        first_predictions = predict_records(model, first_ids)
        if set(raw_records[second]) & set(first_predictions):
            raise GateError("leave-two prediction aliased a leave-one path")
        raw_records[second].update(first_predictions)
        pair_roles.append(
            {
                "excluded_folds": [first, second],
                "fitting_ids_sha256": _id_digest(fit_ids),
                "first_predicted_ids_sha256": _id_digest(first_ids),
                "second_predicted_ids_sha256": _id_digest(second_ids),
            }
        )
    expected_pairs = set(itertools.combinations(range(INNER_FOLDS), 2))
    if seen_pairs != expected_pairs or len(pair_roles) != 6:
        raise GateError("leave-two base role inventory is incomplete")
    if any(set(records) != set(training_ids) for records in raw_records.values()):
        raise GateError("strict base paths do not cover every heldout/calibration role")
    for heldout_fold, values in raw_rows.items():
        calibration = row_folds != heldout_fold
        if not np.isfinite(values[calibration]).all():
            raise GateError("strict base row predictions are incomplete")
    return (
        raw_records,
        fold_by_well,
        raw_rows,
        np.asarray(y),
        np.asarray(row_wells),
        {
            "base_model_count": 10,
            "leave_one_roles": leave_one_roles,
            "leave_two_roles": pair_roles,
        },
    )


def _crossfit_incumbent_training(
    audit: GateAudit,
    training_ids: Sequence[str],
    group_by_well: Mapping[str, str],
) -> tuple[
    dict[int, dict[str, IncumbentSuffix]],
    dict[str, int],
    dict[str, Any],
]:
    """Cross-fit base shrink and path-dependent interval calibration."""

    raw_by_fold, fold_by_well, raw_rows, y, row_wells, base_roles = _raw_base_oof(
        audit, training_ids, group_by_well
    )
    by_fold: dict[int, dict[str, IncumbentSuffix]] = {}
    fold_parameters = []
    for heldout_fold in range(INNER_FOLDS):
        calibration_row_mask = np.asarray(
            [fold_by_well[str(wid)] != heldout_fold for wid in row_wells],
            dtype=bool,
        )
        if not calibration_row_mask.any():
            raise GateError("base-shrink calibration row role is empty")
        base_shrink = incumbent_exact.calibrate_shrink(
            raw_rows[heldout_fold][calibration_row_mask], y[calibration_row_mask]
        )
        records = [
            incumbent_exact.PredictionRecord(
                well=well_id,
                path=record.path,
                idx=record.row_index,
                prediction=record.anchor_tvt + float(base_shrink) * record.raw_delta,
                truth=record.anchor_tvt + record.target_delta,
            )
            for well_id, record in sorted(raw_by_fold[heldout_fold].items())
        ]
        calibration = [
            record for record in records if fold_by_well[record.well] != heldout_fold
        ]
        heldout = [
            record for record in records if fold_by_well[record.well] == heldout_fold
        ]
        if not calibration or not heldout:
            raise GateError("inner incumbent calibration fold is empty")
        calibration_groups = {group_by_well[record.well] for record in calibration}
        heldout_groups = {group_by_well[record.well] for record in heldout}
        if calibration_groups & heldout_groups:
            raise GateError("incumbent calibration shares a heldout equality group")
        # Evidence depends on the base path, so it is recomputed under each
        # fold-specific shrink. It never reads suffix TVT through its inference view.
        typewell_raw, ordered_raw, _ = incumbent_exact._evidence_for_records(records)
        tw_by_well = dict(
            zip((record.well for record in records), typewell_raw, strict=True)
        )
        ordered_by_well = dict(
            zip((record.well for record in records), ordered_raw, strict=True)
        )
        calibration_tw = np.asarray([tw_by_well[record.well] for record in calibration])
        heldout_tw = np.asarray([tw_by_well[record.well] for record in heldout])
        target_shift = np.asarray([record.oracle_shift for record in calibration])
        calibration_tw_scaled, heldout_tw_scaled, tw_shrink = (
            incumbent_exact._scalar_correction(calibration_tw, target_shift, heldout_tw)
        )
        calibration_ordered_raw = [
            ordered_by_well[record.well] for record in calibration
        ]
        heldout_ordered_raw = [ordered_by_well[record.well] for record in heldout]
        ordered_shrink = incumbent_exact._calibrate_vector_shrink(
            calibration, calibration_ordered_raw
        )
        calibration_ordered = [
            ordered_shrink * value for value in calibration_ordered_raw
        ]
        heldout_ordered = [ordered_shrink * value for value in heldout_ordered_raw]
        calibration_tw_arrays = [
            np.full(len(record.idx), correction, dtype=float)
            for record, correction in zip(
                calibration, calibration_tw_scaled, strict=True
            )
        ]
        joint_coefficients = incumbent_exact._fit_joint_correction(
            calibration, calibration_tw_arrays, calibration_ordered
        )
        fold_paths = {
            record.well: IncumbentSuffix(
                row_index=np.asarray(record.idx, dtype=np.int64),
                base=np.asarray(record.prediction, dtype=np.float64),
                # Calibration-well joint is deliberately not used by the field fit.
                joint=np.asarray(record.prediction, dtype=np.float64),
            )
            for record in records
        }
        for record, tw_value, ordered_value in zip(
            heldout, heldout_tw_scaled, heldout_ordered, strict=True
        ):
            correction = np.clip(
                joint_coefficients[0] * float(tw_value)
                + joint_coefficients[1] * ordered_value,
                -25.0,
                25.0,
            )
            fold_paths[record.well] = IncumbentSuffix(
                row_index=np.asarray(record.idx, dtype=np.int64),
                base=np.asarray(record.prediction, dtype=np.float64),
                joint=np.asarray(record.prediction + correction, dtype=np.float64),
            )
        by_fold[heldout_fold] = fold_paths
        fold_parameters.append(
            {
                "inner_fold": heldout_fold,
                "calibration_wells": len(calibration),
                "heldout_wells": len(heldout),
                "base_shrink": float(base_shrink),
                "typewell_shrink": float(tw_shrink),
                "ordered_shrink": float(ordered_shrink),
                "joint_coefficients": [
                    float(joint_coefficients[0]),
                    float(joint_coefficients[1]),
                ],
            }
        )
    if set(by_fold) != set(range(INNER_FOLDS)) or any(
        set(paths) != set(training_ids) for paths in by_fold.values()
    ):
        raise GateError("cross-fitted incumbent fold paths are incomplete")
    return (
        by_fold,
        fold_by_well,
        {
            "strict_base_crossfit": base_roles,
            "crossfit_fold_parameters": fold_parameters,
            "path_dependent_evidence_passes": INNER_FOLDS,
        },
    )


def _as_training_wells(
    wells: Iterable[WellPath],
) -> list[field_core.TrainingWell]:
    result = []
    for well in wells:
        if well.truth is None:
            raise GateError("a field fitting-role well lacks labeled TVT")
        result.append(
            field_core.TrainingWell(
                well_id=well.well_id,
                md=well.md,
                x=well.x,
                y=well.y,
                z=well.z,
                tvt=well.truth,
            )
        )
    return result


def _predict_core(
    model: field_core.StructuralFieldModel, well: WellPath
) -> field_core.StructuralPrediction:
    truncations = _support_query_truncation_count(model, well)
    if truncations:
        raise GateHold(
            f"support-query truncation detected at {truncations} target knot(s)"
        )
    return field_core.predict_structural_field(
        model,
        md=well.md,
        x=well.x,
        y=well.y,
        z=well.z,
        tvt_input=well.tvt_input,
        policy_tvt=well.base_full,
    )


def _support_query_truncation_count(
    model: field_core.StructuralFieldModel, well: WellPath
) -> int:
    """Count bounded support queries that cannot see the declared support set."""

    prefix = _known_prefix(well.tvt_input)
    anchor_md = float(well.md[prefix - 1])
    knots = field_core._prediction_knots(well.md, anchor_md, model.config)
    points = np.column_stack(
        (
            np.interp(knots, well.md, well.x),
            np.interp(knots, well.md, well.y),
        )
    )
    knot_prefix_rows = int(np.searchsorted(knots, anchor_md, side="right"))
    bias_start = int(
        np.searchsorted(
            knots,
            anchor_md - model.config.prefix_bias_window_md,
            side="left",
        )
    )
    interval_start = np.arange(bias_start, knot_prefix_rows - 1)
    if len(interval_start):
        prefix_midpoints = (points[interval_start] + points[interval_start + 1]) / 2.0
        points = np.vstack((points, prefix_midpoints))
    support_count = len(model.support_xy)
    k = min(model.config.max_support_neighbors, support_count)
    if support_count <= k:
        return 0
    declared_wells = min(
        model.config.max_distinct_support_wells,
        int(np.unique(model.support_well_index).size),
    )
    truncations = 0
    for point in points:
        inside_count = int(
            model.support_tree.query_ball_point(
                point, model.config.support_length_ft, return_length=True
            )
        )
        if inside_count <= k:
            continue
        distances, indices = model.support_tree.query(point, k=k)
        distances = np.atleast_1d(np.asarray(distances, dtype=float))
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        inside = distances < model.config.support_length_ft
        distinct = np.unique(model.support_well_index[indices[inside]]).size
        if (
            float(distances[-1]) < model.config.support_length_ft
            and distinct < declared_wells
        ):
            truncations += 1
    return truncations


def _fit_field_grid(
    training_wells_by_fold: Mapping[int, Mapping[str, WellPath]],
    fold_by_well: Mapping[str, int],
) -> tuple[GridFit, dict[str, Any]]:
    if set(training_wells_by_fold) != set(range(INNER_FOLDS)) or any(
        set(paths) != set(fold_by_well) for paths in training_wells_by_fold.values()
    ):
        raise GateError("field training paths and inner fold roles differ")
    if {int(value) for value in fold_by_well.values()} != set(range(INNER_FOLDS)):
        raise GateError("field inner split does not contain four folds")
    score_cells: list[GridFit] = []
    retained_models: tuple[field_core.StructuralFieldModel, ...] = ()
    retained_key: tuple[float, float, float] | None = None
    retained_theta = (0.0, 0.0)
    fit_count = 0
    role_contract = {
        "leave_one_excluded_folds": [[fold] for fold in range(INNER_FOLDS)],
        "leave_two_excluded_folds": [
            list(pair) for pair in itertools.combinations(range(INNER_FOLDS), 2)
        ],
        "pair_models_shared_only_by_identical_unordered_exclusion_role": True,
    }
    for h_ft, laplacian in GRID:
        config = field_config(h_ft, laplacian)
        leave_one_models: dict[int, field_core.StructuralFieldModel] = {}
        leave_two_models: dict[tuple[int, int], field_core.StructuralFieldModel] = {}
        residual_blocks = []
        field_blocks = []
        bias_blocks = []
        for heldout_fold in range(INNER_FOLDS):
            training_wells = training_wells_by_fold[heldout_fold]
            fitting = [
                well
                for well_id, well in training_wells.items()
                if int(fold_by_well[well_id]) != heldout_fold
            ]
            heldout = [
                well
                for well_id, well in training_wells.items()
                if int(fold_by_well[well_id]) == heldout_fold
            ]
            if not fitting or not heldout:
                raise GateError("field inner fold has an empty role")
            model = field_core.fit_structural_field(_as_training_wells(fitting), config)
            _assert_no_coarsening(model)
            fit_count += 1
            leave_one_models[heldout_fold] = model
        for pair in itertools.combinations(range(INNER_FOLDS), 2):
            # Field fitting consumes TRUE TVT only, so the unordered pair role
            # is exactly symmetric and can be safely shared in both directions.
            training_wells = training_wells_by_fold[pair[0]]
            fitting = [
                well
                for well_id, well in training_wells.items()
                if int(fold_by_well[well_id]) not in pair
            ]
            if not fitting:
                raise GateError("leave-two field fitting role is empty")
            model = field_core.fit_structural_field(_as_training_wells(fitting), config)
            _assert_no_coarsening(model)
            fit_count += 1
            if pair in leave_two_models:
                raise GateError("leave-two field model cache role aliased")
            leave_two_models[pair] = model
        if set(leave_one_models) != set(range(INNER_FOLDS)) or set(
            leave_two_models
        ) != set(itertools.combinations(range(INNER_FOLDS), 2)):
            raise GateError("strict field jackknife role inventory is incomplete")
        for heldout_fold in range(INNER_FOLDS):
            training_wells = training_wells_by_fold[heldout_fold]
            heldout = [
                well
                for well_id, well in training_wells.items()
                if int(fold_by_well[well_id]) == heldout_fold
            ]
            primary_model = leave_one_models[heldout_fold]
            for well in heldout:
                if well.truth is None:
                    raise GateError("inner calibration path lacks training TVT")
                prediction = _predict_core(primary_model, well)
                jackknife_models = [primary_model] + [
                    leave_two_models[tuple(sorted((heldout_fold, other_fold)))]
                    for other_fold in range(INNER_FOLDS)
                    if other_fold != heldout_fold
                ]
                if len({id(model) for model in jackknife_models}) != INNER_FOLDS:
                    raise GateError("strict inner jackknife model roles aliased")
                jackknife_proposals = [prediction] + [
                    _predict_core(model, well) for model in jackknife_models[1:]
                ]
                idx = well.suffix_index
                c_j = jackknife_confidence(
                    well.md,
                    jackknife_proposals,
                    primary_model.diagnostics.derivative_residual_scale,
                    int(idx[0]),
                )
                confidence = prediction.confidence[idx] * c_j[idx]
                residual_blocks.append(well.truth[idx] - well.joint_full[idx])
                field_blocks.append(
                    confidence * prediction.field_delta_without_prefix_bias_tvt[idx]
                )
                bias_blocks.append(confidence * prediction.prefix_bias_delta_tvt[idx])
        theta_field, theta_bias, objective = solve_theta(
            np.concatenate(residual_blocks),
            np.concatenate(field_blocks),
            np.concatenate(bias_blocks),
        )
        cell = GridFit(
            h_ft=h_ft,
            laplacian=laplacian,
            theta_field=theta_field,
            theta_bias=theta_bias,
            objective=objective,
            inner_models=(),
        )
        score_cells.append(cell)
        key = (objective, h_ft, -laplacian)
        if retained_key is None or key < retained_key:
            retained_key = key
            retained_models = tuple(
                leave_one_models[fold] for fold in range(INNER_FOLDS)
            )
            retained_theta = (theta_field, theta_bias)
    selected_cell = _choose_grid(score_cells)
    if (
        retained_key is None
        or (
            selected_cell.theta_field,
            selected_cell.theta_bias,
        )
        != retained_theta
    ):
        raise GateError("streaming grid retention differs from frozen tie break")
    selected = GridFit(
        h_ft=selected_cell.h_ft,
        laplacian=selected_cell.laplacian,
        theta_field=selected_cell.theta_field,
        theta_bias=selected_cell.theta_bias,
        objective=selected_cell.objective,
        inner_models=retained_models,
    )
    if len(selected.inner_models) != INNER_FOLDS or fit_count != len(GRID) * 10:
        raise GateError("field grid did not execute exactly six-by-ten strict fits")
    return selected, {
        "evaluated_grid": list(frozen_grid()),
        "inner_field_fits": fit_count,
        "strict_models_per_grid_cell": 10,
        "role_contract": role_contract,
        "tie_break": "shorter h, then stronger Laplacian",
    }


def _persistent_derivative_support(
    mask: NDArray[np.bool_], suffix_start: int
) -> NDArray[np.bool_]:
    mask = np.asarray(mask, dtype=bool)
    valid = np.zeros(len(mask), dtype=bool)
    if len(mask) < 2 or suffix_start < 0 or suffix_start >= len(mask):
        return valid
    if suffix_start + 1 < len(mask):
        valid[suffix_start] = mask[suffix_start] and mask[suffix_start + 1]
    if suffix_start + 1 < len(mask) - 1:
        valid[suffix_start + 1 : -1] = (
            mask[suffix_start:-2]
            & mask[suffix_start + 1 : -1]
            & mask[suffix_start + 2 :]
        )
    if len(mask) - 1 > suffix_start:
        valid[-1] = mask[-2] and mask[-1]
    return valid


def jackknife_confidence(
    md: NDArray[np.float64],
    proposals: Sequence[field_core.StructuralPrediction],
    sigma: float,
    suffix_start: int,
) -> NDArray[np.float64]:
    """Frozen four-model derivative-MAD confidence, independent of truth."""

    md = np.asarray(md, dtype=float)
    if len(proposals) != INNER_FOLDS:
        raise GateError("jackknife confidence requires exactly four inner models")
    if len(md) < 2 or not np.all(np.diff(md) > 0.0):
        raise GateError("jackknife MD must be strictly increasing")
    derivatives = []
    valid_rows = []
    for proposal in proposals:
        delta = np.asarray(proposal.field_delta_without_prefix_bias_tvt, dtype=float)
        support = np.asarray(proposal.support_mask, dtype=bool)
        if len(delta) != len(md) or len(support) != len(md):
            raise GateError("jackknife proposal length mismatch")
        suffix_support = support[suffix_start:]
        if len(suffix_support):
            lost = np.flatnonzero(~suffix_support)
            if len(lost) and suffix_support[lost[0] :].any():
                raise GateError("persistent support reactivated after loss")
        derivative = np.gradient(delta, md, edge_order=1)
        if suffix_start + 1 < len(md):
            derivative[suffix_start] = (
                delta[suffix_start + 1] - delta[suffix_start]
            ) / (md[suffix_start + 1] - md[suffix_start])
        derivatives.append(derivative)
        valid_rows.append(_persistent_derivative_support(support, suffix_start))
    derivative_array = np.vstack(derivatives)
    valid_array = np.vstack(valid_rows)
    count = valid_array.sum(axis=0)
    confidence = np.zeros(len(md), dtype=float)
    scale = max(float(sigma), 1.0e-8)
    for row in np.flatnonzero(count >= 3):
        values = derivative_array[valid_array[:, row], row]
        median = float(np.median(values))
        tau = 1.4826 * float(np.median(np.abs(values - median)))
        confidence[row] = 1.0 / (1.0 + (tau / scale) ** 2)
    confidence[:suffix_start] = 0.0
    # Four persistent masks make the >=3 condition monotone after the suffix
    # begins; a nonzero cJ therefore cannot reactivate downstream.
    active = confidence[suffix_start:] > 0.0
    lost = np.flatnonzero(~active)
    if len(lost) and active[lost[0] :].any():
        raise GateError("jackknife confidence reactivated after support loss")
    return confidence


def _prediction_diagnostics(
    prediction: field_core.StructuralPrediction,
) -> dict[str, Any]:
    diag = prediction.diagnostics
    return {
        "status": diag.status,
        "evaluation_rows": int(diag.evaluation_rows),
        "prefix_rows": int(diag.prefix_rows),
        "suffix_rows": int(diag.suffix_rows),
        "prefix_bias": float(diag.prefix_bias),
        "prefix_bias_intervals": int(diag.prefix_bias_intervals),
        "mean_training_midpoint_distance_ft": float(
            diag.nearest_resampled_training_midpoint_distance_mean_ft
        ),
        "max_training_midpoint_distance_ft": float(
            diag.nearest_resampled_training_midpoint_distance_max_ft
        ),
        "effective_well_support_mean": float(diag.effective_well_support_mean),
        "query_direction_observability_mean": float(
            diag.query_direction_observability_mean
        ),
        "cut_edge_crossings": int(diag.cut_edge_crossings),
        "fallback_fraction": float(diag.fallback_fraction),
        "mean_core_confidence": float(diag.mean_confidence),
    }


def _predict_candidate(
    final_model: field_core.StructuralFieldModel,
    inner_models: Sequence[field_core.StructuralFieldModel],
    selected: GridFit,
    well: WellPath,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    final = _predict_core(final_model, well)
    inner = [_predict_core(model, well) for model in inner_models]
    c_j = jackknife_confidence(
        well.md,
        inner,
        final_model.diagnostics.derivative_residual_scale,
        int(well.suffix_index[0]),
    )
    confidence = final.confidence * c_j
    candidate = well.joint_full + confidence * (
        selected.theta_field * final.field_delta_without_prefix_bias_tvt
        + selected.theta_bias * final.prefix_bias_delta_tvt
    )
    candidate[: int(well.suffix_index[0])] = well.tvt_input[: int(well.suffix_index[0])]
    idx = well.suffix_index
    arrays = {
        "row_index": idx.astype(np.int32),
        "base_prediction": well.base_full[idx].astype(np.float64),
        "joint_prediction": well.joint_full[idx].astype(np.float64),
        "candidate_prediction": candidate[idx].astype(np.float64),
        "field_confidence": confidence[idx].astype(np.float64),
        "field_delta_without_prefix_bias": final.field_delta_without_prefix_bias_tvt[
            idx
        ].astype(np.float64),
        "prefix_bias_delta": final.prefix_bias_delta_tvt[idx].astype(np.float64),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise GateError(f"non-finite candidate artifact for {well.well_id}")
    if np.any(arrays["field_confidence"] < 0.0) or np.any(
        arrays["field_confidence"] > 1.0 + 1.0e-12
    ):
        raise GateError("field confidence escaped [0, 1]")
    expected = arrays["joint_prediction"] + arrays["field_confidence"] * (
        selected.theta_field * arrays["field_delta_without_prefix_bias"]
        + selected.theta_bias * arrays["prefix_bias_delta"]
    )
    if not np.allclose(
        arrays["candidate_prediction"], expected, rtol=0.0, atol=1.0e-10
    ):
        raise GateError("candidate formula drift")
    diagnostics = {
        **_prediction_diagnostics(final),
        "mean_jackknife_confidence": float(np.mean(c_j[idx])),
        "mean_final_confidence": float(np.mean(confidence[idx])),
        "supported_fraction": float(np.mean(confidence[idx] > 0.0)),
    }
    return arrays, diagnostics


def _field_shard_name(mode: str, repeat: int, fold: int) -> str:
    return (
        f"exact_repeat_{repeat}_fold_{fold}.json"
        if mode == "exact"
        else f"region_fold_{fold}.json"
    )


def _field_prediction_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".npz")


def _with_suffix(template: WellPath, suffix: IncumbentSuffix) -> WellPath:
    prefix = int(template.suffix_index[0])
    if not np.array_equal(suffix.row_index, template.suffix_index):
        raise GateError(f"cross-fitted incumbent index drift for {template.well_id}")
    base = template.tvt_input.copy()
    joint = template.tvt_input.copy()
    base[prefix:] = suffix.base
    joint[prefix:] = suffix.joint
    return WellPath(
        well_id=template.well_id,
        md=template.md,
        x=template.x,
        y=template.y,
        z=template.z,
        tvt_input=template.tvt_input,
        base_full=base,
        joint_full=joint,
        suffix_index=template.suffix_index,
        truth=template.truth,
    )


def _id_digest(values: Sequence[str]) -> str:
    return _canonical_digest(list(values))


def _model_metadata(model: field_core.StructuralFieldModel) -> dict[str, Any]:
    diag = model.diagnostics
    return {
        "training_wells": int(diag.wells),
        "resampled_intervals": int(diag.resampled_intervals),
        "inducing_nodes": int(diag.inducing_nodes),
        "graph_edges": int(diag.graph_edges),
        "graph_faces": int(diag.graph_faces),
        "discontinuity_candidates": int(diag.discontinuity_candidates),
        "graph_components_after_cuts": int(diag.graph_components_after_cuts),
        "requested_inducing_cell_ft": float(model.config.inducing_cell_ft),
        "actual_inducing_cell_ft": float(diag.actual_inducing_cell_ft),
        "derivative_residual_scale": float(diag.derivative_residual_scale),
        "solver_stop_codes": [int(value) for value in diag.solver_stop_codes],
    }


def _require_exact_keys(
    value: object, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GateError(f"{label} schema drift")
    return value


def _reject_sensitive_field_names(value: object, path: str = "shard") -> None:
    """Reject label/metric vocabulary outside the one audited model diagnostic."""

    allowed = {"derivative_residual_scale"}
    sensitive = ("truth", "target", "label", "oracle", "residual", "error")
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            if key not in allowed and (
                any(token in key for token in sensitive)
                or key in {"sse", "rmse"}
                or key.endswith(("_sse", "_rmse"))
            ):
                raise GateError(f"metric-silent schema rejected field {path}.{raw_key}")
            _reject_sensitive_field_names(nested, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive_field_names(nested, f"{path}[{index}]")


def _validate_model_metadata_schema(value: object, label: str) -> None:
    model = _require_exact_keys(
        value,
        {
            "training_wells",
            "resampled_intervals",
            "inducing_nodes",
            "graph_edges",
            "graph_faces",
            "discontinuity_candidates",
            "graph_components_after_cuts",
            "requested_inducing_cell_ft",
            "actual_inducing_cell_ft",
            "derivative_residual_scale",
            "solver_stop_codes",
        },
        label,
    )
    for key in (
        "training_wells",
        "resampled_intervals",
        "inducing_nodes",
        "graph_edges",
        "graph_faces",
        "discontinuity_candidates",
        "graph_components_after_cuts",
    ):
        if int(model[key]) < 0 or float(model[key]) != int(model[key]):
            raise GateError(f"{label} has an invalid count: {key}")
    requested = _finite_nonnegative(
        model["requested_inducing_cell_ft"], f"{label} requested cell"
    )
    actual = _finite_nonnegative(
        model["actual_inducing_cell_ft"], f"{label} actual cell"
    )
    _finite_nonnegative(model["derivative_residual_scale"], f"{label} derivative scale")
    if not math.isclose(requested, actual, rel_tol=0.0, abs_tol=1.0e-9):
        raise GateHold(f"{label} records forbidden inducing-cell coarsening")
    stop_codes = model["solver_stop_codes"]
    if (
        not isinstance(stop_codes, list)
        or len(stop_codes) != 2
        or any(
            isinstance(value, bool)
            or int(value) != value
            or int(value) not in {0, 1, 2, 4, 5}
            for value in stop_codes
        )
    ):
        raise GateError(f"{label} solver-stop schema drift")


def _validate_field_shard_schema(shard: Mapping[str, Any]) -> None:
    """Strict recursive allowlist for a metric-silent field prediction shard."""

    _reject_sensitive_field_names(shard)
    _require_exact_keys(
        shard,
        {
            "status",
            "status_ceiling",
            "method",
            "protocol_sha256",
            "benchmark_file",
            "benchmark_sha256",
            "incumbent_inventory_sha256",
            "mode",
            "repeat",
            "fold",
            "outer_role_sha256",
            "training_well_count",
            "validation_well_count",
            "embargo_well_count",
            "learned_from_outer_training_only",
            "validation_diagnostics",
            "prediction_file",
            "prediction_sha256",
            "prediction_logical_sha256",
            "prediction_rows",
            "prediction_channels",
            "validation_wells",
            "runtime_seconds",
        },
        "field shard",
    )
    _require_exact_keys(
        shard["outer_role_sha256"],
        {"training_ids", "validation_ids", "embargo_ids"},
        "field shard outer roles",
    )
    learned = _require_exact_keys(
        shard["learned_from_outer_training_only"],
        {
            "incumbent_crossfit",
            "field_grid",
            "selected_field_cell",
            "final_field_model",
            "jackknife_inner_models",
            "support_query_truncation_count",
        },
        "field shard learned parameters",
    )
    incumbent = _require_exact_keys(
        learned["incumbent_crossfit"],
        {
            "strict_base_crossfit",
            "crossfit_fold_parameters",
            "path_dependent_evidence_passes",
        },
        "field shard incumbent crossfit",
    )
    base = _require_exact_keys(
        incumbent["strict_base_crossfit"],
        {"base_model_count", "leave_one_roles", "leave_two_roles"},
        "field shard strict base crossfit",
    )
    leave_one = base["leave_one_roles"]
    leave_two = base["leave_two_roles"]
    if not isinstance(leave_one, list) or len(leave_one) != INNER_FOLDS:
        raise GateError("field shard leave-one role inventory drift")
    if not isinstance(leave_two, list) or len(leave_two) != 6:
        raise GateError("field shard leave-two role inventory drift")
    for index, role in enumerate(leave_one):
        _require_exact_keys(
            role,
            {"heldout_fold", "fitting_ids_sha256", "predicted_ids_sha256"},
            f"field shard leave-one role {index}",
        )
    for index, role in enumerate(leave_two):
        _require_exact_keys(
            role,
            {
                "excluded_folds",
                "fitting_ids_sha256",
                "first_predicted_ids_sha256",
                "second_predicted_ids_sha256",
            },
            f"field shard leave-two role {index}",
        )
    fold_parameters = incumbent["crossfit_fold_parameters"]
    if not isinstance(fold_parameters, list) or len(fold_parameters) != INNER_FOLDS:
        raise GateError("field shard incumbent fold-parameter inventory drift")
    for index, parameters in enumerate(fold_parameters):
        _require_exact_keys(
            parameters,
            {
                "inner_fold",
                "calibration_wells",
                "heldout_wells",
                "base_shrink",
                "typewell_shrink",
                "ordered_shrink",
                "joint_coefficients",
            },
            f"field shard incumbent fold parameters {index}",
        )
    grid = _require_exact_keys(
        learned["field_grid"],
        {
            "evaluated_grid",
            "inner_field_fits",
            "strict_models_per_grid_cell",
            "role_contract",
            "tie_break",
        },
        "field shard grid",
    )
    _require_exact_keys(
        grid["role_contract"],
        {
            "leave_one_excluded_folds",
            "leave_two_excluded_folds",
            "pair_models_shared_only_by_identical_unordered_exclusion_role",
        },
        "field shard grid role contract",
    )
    _require_exact_keys(
        learned["selected_field_cell"],
        {"h_ft", "laplacian", "theta_field", "theta_bias"},
        "field shard selected cell",
    )
    _validate_model_metadata_schema(learned["final_field_model"], "final field model")
    inner_models = learned["jackknife_inner_models"]
    if not isinstance(inner_models, list) or len(inner_models) != INNER_FOLDS:
        raise GateError("field shard jackknife model inventory drift")
    for index, model in enumerate(inner_models):
        _validate_model_metadata_schema(model, f"jackknife field model {index}")

    diagnostics = shard["validation_diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise GateError("field shard validation diagnostics schema drift")
    diagnostic_keys = {
        "status",
        "evaluation_rows",
        "prefix_rows",
        "suffix_rows",
        "prefix_bias",
        "prefix_bias_intervals",
        "mean_training_midpoint_distance_ft",
        "max_training_midpoint_distance_ft",
        "effective_well_support_mean",
        "query_direction_observability_mean",
        "cut_edge_crossings",
        "fallback_fraction",
        "mean_core_confidence",
        "mean_jackknife_confidence",
        "mean_final_confidence",
        "supported_fraction",
    }
    for well_id, diagnostic in diagnostics.items():
        _require_exact_keys(
            diagnostic, diagnostic_keys, f"field shard diagnostic {well_id}"
        )
    rows = shard["validation_wells"]
    if not isinstance(rows, list):
        raise GateError("field shard validation-well schema drift")
    for index, row in enumerate(rows):
        _require_exact_keys(
            row,
            {"well", "well_index", "equality_group", "n_rows"},
            f"field shard validation well {index}",
        )


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise GateError(f"{label} is not finite and nonnegative")
    return number


def _frozen_worst_proxy_row(audit: GateAudit) -> Mapping[str, Any]:
    proxy = audit.protocol.get("benchmark_work_proxy")
    if not isinstance(proxy, Mapping):
        raise GateError("protocol lacks a benchmark work proxy")
    _validate_work_proxy_payload(proxy)
    identity = proxy["maximizing_identity"]
    matches = [
        row
        for row in proxy["folds"]
        if (
            str(row["mode"]),
            int(row["repeat"]),
            int(row["fold"]),
        )
        == (
            str(identity["mode"]),
            int(identity["repeat"]),
            int(identity["fold"]),
        )
    ]
    if len(matches) != 1:
        raise GateError("frozen worst-work proxy identity is ambiguous")
    return matches[0]


def _validate_benchmark(path: Path, audit: GateAudit) -> str:
    path = path.resolve()
    if _read_sidecar(path) != sha256_file(path):
        raise GateError("benchmark sidecar drift")
    benchmark = _read_json(path, "field benchmark")
    _require_exact_keys(
        benchmark,
        {
            "status",
            "status_ceiling",
            "method",
            "protocol_sha256",
            "benchmark_work_proxy_sha256",
            "worst_work_identity",
            "work_shape",
            "selected_cell",
            "timing_seconds",
            "memory_gib",
            "acceptance",
            "all_acceptance_pass",
            "validation_metrics",
        },
        "benchmark",
    )
    if benchmark.get("status") != "MEASURE_ONLY_TRUTH_QUARANTINED_BENCHMARK":
        raise GateError("benchmark status drift")
    if (
        benchmark.get("status_ceiling") != STATUS_CEILING
        or benchmark.get("method") != METHOD
        or benchmark.get("protocol_sha256") != audit.protocol_sha256
        or benchmark.get("validation_metrics")
        != "withheld; validation TVT was not parsed"
    ):
        raise GateError("benchmark protocol identity drift")
    proxy = audit.protocol["benchmark_work_proxy"]
    if benchmark.get("benchmark_work_proxy_sha256") != proxy["proxy_sha256"]:
        raise GateError("benchmark work-proxy digest drift")
    proxy_row = _frozen_worst_proxy_row(audit)
    expected_identity = {
        "mode": str(proxy_row["mode"]),
        "repeat": int(proxy_row["repeat"]),
        "fold": int(proxy_row["fold"]),
        "training_wells": int(proxy_row["training_wells"]),
        "validation_wells": int(proxy_row["validation_wells"]),
        "embargo_wells": int(proxy_row["embargo_wells"]),
        "proxy_units": int(proxy_row["proxy_units"]),
    }
    identity = _require_exact_keys(
        benchmark.get("worst_work_identity"),
        set(expected_identity),
        "benchmark identity",
    )
    if dict(identity) != expected_identity:
        raise GateError("benchmark did not measure the frozen worst-work identity")

    shape = _require_exact_keys(
        benchmark.get("work_shape"),
        {
            "strict_base_models",
            "leave_one_base_models",
            "leave_two_base_models",
            "grid_cells",
            "strict_inner_field_models_per_cell",
            "retained_outer_jackknife_models",
            "inner_field_fits",
            "selected_outer_refits",
            "prediction_rows",
            "solver_caps_changed",
            "coarsening_allowed",
            "support_query_truncation_count",
        },
        "benchmark work shape",
    )
    if (
        int(shape.get("grid_cells", -1)) != 6
        or int(shape.get("strict_base_models", -1)) != 10
        or int(shape.get("leave_one_base_models", -1)) != 4
        or int(shape.get("leave_two_base_models", -1)) != 6
        or int(shape.get("strict_inner_field_models_per_cell", -1)) != 10
        or int(shape.get("retained_outer_jackknife_models", -1)) != 4
        or int(shape.get("inner_field_fits", -1)) != 60
        or int(shape.get("selected_outer_refits", -1)) != 1
        or int(shape.get("support_query_truncation_count", -1)) != 0
        or bool(shape.get("solver_caps_changed", True))
        or bool(shape.get("coarsening_allowed", True))
    ):
        raise GateError("benchmark work shape or cap/coarsening contract drift")
    if int(shape.get("prediction_rows", 0)) <= 0:
        raise GateError("benchmark prediction-row work is empty")
    selected = _require_exact_keys(
        benchmark.get("selected_cell"),
        {"h_ft", "laplacian", "theta_field", "theta_bias"},
        "benchmark selected cell",
    )
    h_ft = _finite_nonnegative(selected["h_ft"], "benchmark selected h")
    laplacian = _finite_nonnegative(
        selected["laplacian"], "benchmark selected Laplacian"
    )
    theta_field = _finite_nonnegative(selected["theta_field"], "benchmark theta_field")
    theta_bias = _finite_nonnegative(selected["theta_bias"], "benchmark theta_bias")
    if (h_ft, laplacian) not in GRID or not (0.0 <= theta_bias <= theta_field <= 1.0):
        raise GateError("benchmark selected cell escaped the frozen grid/bounds")

    timing = _require_exact_keys(
        benchmark.get("timing_seconds"),
        {
            "incumbent_crossfit",
            "field",
            "total",
            "extrapolated_two_worker_fifteen_fold",
        },
        "benchmark timing",
    )
    incumbent_seconds = _finite_nonnegative(
        timing["incumbent_crossfit"], "benchmark incumbent time"
    )
    field_seconds = _finite_nonnegative(timing["field"], "benchmark field time")
    total_seconds = _finite_nonnegative(timing["total"], "benchmark total time")
    extrapolated = _finite_nonnegative(
        timing["extrapolated_two_worker_fifteen_fold"],
        "benchmark extrapolated time",
    )
    if incumbent_seconds > total_seconds or field_seconds > total_seconds:
        raise GateError("benchmark component time exceeds total time")
    expected_extrapolated = total_seconds * math.ceil(15 / 2)
    if not math.isclose(
        extrapolated, expected_extrapolated, rel_tol=1.0e-12, abs_tol=1.0e-9
    ):
        raise GateError("benchmark extrapolation arithmetic drift")
    memory = _require_exact_keys(
        benchmark.get("memory_gib"),
        {"field_peak_rss", "total_peak_rss"},
        "benchmark memory",
    )
    field_rss = _finite_nonnegative(memory["field_peak_rss"], "benchmark field RSS")
    total_rss = _finite_nonnegative(memory["total_peak_rss"], "benchmark total RSS")

    acceptance = _require_exact_keys(
        benchmark.get("acceptance"),
        {
            "field_wall",
            "field_peak_rss",
            "total_wall",
            "total_peak_rss",
            "two_worker_fifteen_fold",
            "no_caps_solver_or_coarsening",
            "no_support_query_truncation",
        },
        "benchmark acceptance",
    )
    expected_keys = {
        "field_wall",
        "field_peak_rss",
        "total_wall",
        "total_peak_rss",
        "two_worker_fifteen_fold",
        "no_caps_solver_or_coarsening",
        "no_support_query_truncation",
    }
    if set(acceptance) != expected_keys:
        raise GateError("benchmark acceptance schema drift")
    thresholds = _evaluation_contract()["runtime_acceptance"]
    recomputed = {
        "field_wall": field_seconds <= thresholds["field_wall_seconds_at_most"],
        "field_peak_rss": field_rss <= thresholds["field_peak_rss_gib_at_most"],
        "total_wall": total_seconds <= thresholds["total_wall_seconds_at_most"],
        "total_peak_rss": total_rss <= thresholds["total_peak_rss_gib_at_most"],
        "two_worker_fifteen_fold": extrapolated
        <= thresholds["extrapolated_two_worker_fifteen_fold_seconds_at_most"],
        "no_caps_solver_or_coarsening": not bool(shape["solver_caps_changed"])
        and not bool(shape["coarsening_allowed"]),
        "no_support_query_truncation": int(shape["support_query_truncation_count"])
        == 0,
    }
    if any(not isinstance(value, bool) for value in acceptance.values()):
        raise GateError("benchmark acceptance values are not booleans")
    if dict(acceptance) != recomputed or benchmark.get(
        "all_acceptance_pass"
    ) is not all(recomputed.values()):
        raise GateError("benchmark acceptance was not recomputed from measurements")
    if not all(recomputed.values()):
        raise GateHold("benchmark acceptance failed; run is STOP")
    return sha256_file(path)


def _validate_field_shard(
    shard: Mapping[str, Any],
    audit: GateAudit,
    mode: str,
    repeat: int,
    fold: int,
    shard_path: Path,
    benchmark_path: Path,
) -> dict[str, NDArray[Any]]:
    _validate_field_shard_schema(shard)
    if shard.get("status") != "MEASURE_ONLY_FIELD_PREDICTIONS_SEALED_TRUTH_UNREAD":
        raise GateError("field shard status drift")
    if shard.get("status_ceiling") != STATUS_CEILING:
        raise GateError("field shard status ceiling drift")
    if _read_sidecar(shard_path) != sha256_file(shard_path):
        raise GateError("field shard metadata hash drift")
    if shard.get("protocol_sha256") != audit.protocol_sha256:
        raise GateError("field shard protocol identity drift")
    benchmark_sha256 = _validate_benchmark(benchmark_path, audit)
    if (
        shard.get("benchmark_file") != benchmark_path.name
        or shard.get("benchmark_sha256") != benchmark_sha256
    ):
        raise GateError("field shard benchmark lineage drift")
    if (
        shard.get("incumbent_inventory_sha256")
        != audit.protocol["incumbent_pretruth_inventory"]["inventory_sha256"]
    ):
        raise GateError("field shard incumbent lineage drift")
    if (
        shard.get("method") != METHOD
        or shard.get("mode") != mode
        or int(shard.get("repeat", -1)) != repeat
        or int(shard.get("fold", -1)) != fold
    ):
        raise GateError("field shard identity drift")
    training_ids, validation_ids, embargo_ids, group_by_well = _outer_roles(
        audit, mode, repeat, fold
    )
    if shard.get("outer_role_sha256") != {
        "training_ids": _id_digest(training_ids),
        "validation_ids": _id_digest(validation_ids),
        "embargo_ids": _id_digest(embargo_ids),
    }:
        raise GateError("field shard outer-role digest drift")
    if (
        int(shard.get("training_well_count", -1)) != len(training_ids)
        or int(shard.get("validation_well_count", -1)) != len(validation_ids)
        or int(shard.get("embargo_well_count", -1)) != len(embargo_ids)
    ):
        raise GateError("field shard outer-role count drift")
    runtime = _finite_nonnegative(shard.get("runtime_seconds"), "field shard runtime")
    if not math.isfinite(runtime):
        raise GateError("field shard runtime drift")
    learned = shard.get("learned_from_outer_training_only")
    if not isinstance(learned, Mapping):
        raise GateError("field shard lacks learned parameters")
    if int(learned.get("support_query_truncation_count", -1)) != 0:
        raise GateHold("field shard records support-query truncation")
    selected = learned.get("selected_field_cell")
    if not isinstance(selected, Mapping) or set(selected) != {
        "h_ft",
        "laplacian",
        "theta_field",
        "theta_bias",
    }:
        raise GateError("field shard selected-cell schema drift")
    h_ft = float(selected["h_ft"])
    laplacian = float(selected["laplacian"])
    theta_field = float(selected["theta_field"])
    theta_bias = float(selected["theta_bias"])
    if (h_ft, laplacian) not in GRID or not (0.0 <= theta_bias <= theta_field <= 1.0):
        raise GateError("field shard selected parameters escaped frozen bounds")
    incumbent = learned["incumbent_crossfit"]
    base_roles = incumbent["strict_base_crossfit"]
    if (
        int(base_roles["base_model_count"]) != 10
        or int(incumbent["path_dependent_evidence_passes"]) != INNER_FOLDS
    ):
        raise GateError("field shard strict incumbent work shape drift")
    leave_one = base_roles["leave_one_roles"]
    if [int(role["heldout_fold"]) for role in leave_one] != list(range(INNER_FOLDS)):
        raise GateError("field shard leave-one exclusion roles drift")
    leave_two_pairs = [
        tuple(map(int, role["excluded_folds"]))
        for role in base_roles["leave_two_roles"]
    ]
    if leave_two_pairs != list(itertools.combinations(range(INNER_FOLDS), 2)):
        raise GateError("field shard leave-two exclusion roles drift")
    for role in [*leave_one, *base_roles["leave_two_roles"]]:
        for key, value in role.items():
            if key.endswith("sha256") and not _is_sha256(value):
                raise GateError("field shard incumbent role digest drift")
    parameters = incumbent["crossfit_fold_parameters"]
    if [int(row["inner_fold"]) for row in parameters] != list(range(INNER_FOLDS)):
        raise GateError("field shard incumbent calibration fold roles drift")
    for row in parameters:
        if int(row["calibration_wells"]) + int(row["heldout_wells"]) != len(
            training_ids
        ):
            raise GateError("field shard incumbent calibration membership count drift")
        scalars = [
            float(row["base_shrink"]),
            float(row["typewell_shrink"]),
            float(row["ordered_shrink"]),
        ]
        joint = np.asarray(row["joint_coefficients"], dtype=float)
        if (
            not np.isfinite(scalars).all()
            or not (0.0 <= scalars[0] <= 1.2)
            or not (0.0 <= scalars[1] <= 1.5)
            or not (0.0 <= scalars[2] <= 1.5)
            or joint.shape != (2,)
            or not np.isfinite(joint).all()
            or np.any(joint < -1.0)
            or np.any(joint > 2.0)
        ):
            raise GateError("field shard incumbent calibration parameter drift")
    field_grid = learned["field_grid"]
    if (
        field_grid["evaluated_grid"] != list(frozen_grid())
        or int(field_grid["inner_field_fits"]) != 60
        or int(field_grid["strict_models_per_grid_cell"]) != 10
        or field_grid["tie_break"] != "shorter h, then stronger Laplacian"
        or field_grid["role_contract"]
        != {
            "leave_one_excluded_folds": [[value] for value in range(INNER_FOLDS)],
            "leave_two_excluded_folds": [
                list(pair) for pair in itertools.combinations(range(INNER_FOLDS), 2)
            ],
            "pair_models_shared_only_by_identical_unordered_exclusion_role": True,
        }
    ):
        raise GateError("field shard strict field-grid contract drift")
    expected_cell = h_ft / 2.0
    for metadata in [
        learned["final_field_model"],
        *learned["jackknife_inner_models"],
    ]:
        if not math.isclose(
            float(metadata["requested_inducing_cell_ft"]),
            expected_cell,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise GateError("field shard model metadata differs from selected cell")
    if int(learned["final_field_model"]["training_wells"]) != len(training_ids):
        raise GateError("final field model training-well count drift")
    if shard.get("prediction_channels") != list(PREDICTION_ARRAYS[2:]):
        raise GateError("field shard prediction-channel schema drift")
    diagnostics = shard["validation_diagnostics"]
    if set(map(str, diagnostics)) != set(validation_ids):
        raise GateError("field shard validation-diagnostic membership drift")
    for well_id, diagnostic in diagnostics.items():
        for key, value in diagnostic.items():
            if key != "status" and not math.isfinite(float(value)):
                raise GateError(
                    f"field shard diagnostic is non-finite: {well_id}:{key}"
                )
        if diagnostic["status"] != "anchored_field_100ft_knots_with_policy_fallback":
            raise GateError("field shard prediction diagnostic status drift")
        for key in (
            "evaluation_rows",
            "prefix_rows",
            "suffix_rows",
            "prefix_bias_intervals",
            "cut_edge_crossings",
        ):
            if int(diagnostic[key]) < 0 or float(diagnostic[key]) != int(
                diagnostic[key]
            ):
                raise GateError(f"field shard diagnostic count drift: {well_id}:{key}")
        for key in (
            "fallback_fraction",
            "mean_core_confidence",
            "mean_jackknife_confidence",
            "mean_final_confidence",
            "supported_fraction",
            "query_direction_observability_mean",
        ):
            if not 0.0 <= float(diagnostic[key]) <= 1.0 + 1.0e-12:
                raise GateError(f"field shard diagnostic range drift: {well_id}:{key}")
        for key in (
            "mean_training_midpoint_distance_ft",
            "max_training_midpoint_distance_ft",
            "effective_well_support_mean",
        ):
            if float(diagnostic[key]) < 0.0:
                raise GateError(f"field shard diagnostic sign drift: {well_id}:{key}")
    prediction_name = str(shard.get("prediction_file", ""))
    prediction_path = _safe_basename(shard_path.parent, prediction_name)
    if prediction_path != _field_prediction_path(shard_path).resolve():
        raise GateError("field prediction basename drift")
    if sha256_file(prediction_path) != shard.get("prediction_sha256"):
        raise GateError("field prediction byte hash drift")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_ARRAYS):
            raise GateError("field prediction NPZ schema drift")
        arrays = {name: archive[name].copy() for name in PREDICTION_ARRAYS}
    if len({len(value) for value in arrays.values()}) != 1 or len(
        arrays["row_index"]
    ) != int(shard.get("prediction_rows", -1)):
        raise GateError("field prediction array length drift")
    if _logical_array_hash(arrays) != shard.get("prediction_logical_sha256"):
        raise GateError("field prediction logical hash drift")
    if (
        arrays["well_index"].dtype.kind not in "iu"
        or arrays["row_index"].dtype.kind not in "iu"
    ):
        raise GateError("field prediction identity arrays are not integers")
    for name in PREDICTION_ARRAYS[2:]:
        if not np.isfinite(arrays[name]).all():
            raise GateError(f"field prediction contains non-finite {name}")
    if np.any(arrays["field_confidence"] < 0.0) or np.any(
        arrays["field_confidence"] > 1.0 + 1.0e-12
    ):
        raise GateError("field prediction confidence drift")
    formula = arrays["joint_prediction"] + arrays["field_confidence"] * (
        theta_field * arrays["field_delta_without_prefix_bias"]
        + theta_bias * arrays["prefix_bias_delta"]
    )
    if not np.allclose(arrays["candidate_prediction"], formula, rtol=0.0, atol=1.0e-10):
        raise GateError("sealed candidate formula drift")
    rows = shard.get("validation_wells")
    if not isinstance(rows, list):
        raise GateError("field shard lacks validation-well metadata")
    wells = [str(row.get("well", "")) for row in rows]
    if wells != validation_ids:
        raise GateError("field shard validation ordering drift")
    inventory = _inventory_by_well(audit)
    for well_index, row in enumerate(rows):
        well_id = wells[well_index]
        if int(row.get("well_index", -1)) != well_index:
            raise GateError("field shard well index drift")
        if str(row.get("equality_group", "")) != group_by_well[well_id]:
            raise GateError("field shard equality-group drift")
        selected_rows = arrays["well_index"] == well_index
        expected_count = int(inventory.loc[well_id, "suffix_rows"])
        if (
            int(selected_rows.sum()) != expected_count
            or int(row.get("n_rows", -1)) != expected_count
        ):
            raise GateError("field shard per-well row count drift")
        prefix = int(inventory.loc[well_id, "prefix_rows"])
        total = int(inventory.loc[well_id, "rows"])
        diagnostic = diagnostics[well_id]
        if (
            int(diagnostic["evaluation_rows"]) <= 0
            or int(diagnostic["prefix_rows"]) != prefix
            or int(diagnostic["suffix_rows"]) != expected_count
        ):
            raise GateError("field shard diagnostic row-count drift")
        if not np.array_equal(
            arrays["row_index"][selected_rows], np.arange(prefix, total)
        ):
            raise GateError("field shard row indices are not the contiguous suffix")
    if set(wells) != set(validation_ids):
        raise GateError("field shard validation coverage drift")
    sealed_incumbent = _load_incumbent_suffixes(audit, mode, repeat)
    for well_index, well_id in enumerate(wells):
        expected = sealed_incumbent.get(well_id)
        if expected is None:
            raise GateError("role-exact incumbent lacks a validation well")
        selected_rows = arrays["well_index"] == well_index
        if (
            not np.array_equal(arrays["row_index"][selected_rows], expected.row_index)
            or not np.array_equal(
                arrays["base_prediction"][selected_rows], expected.base
            )
            or not np.array_equal(
                arrays["joint_prediction"][selected_rows], expected.joint
            )
        ):
            raise GateError("field shard comparator differs from role-exact incumbent")
    return arrays


def _run_one_fold(
    audit: GateAudit,
    mode: str,
    repeat: int,
    fold: int,
    output_dir: Path,
    resume: bool,
    benchmark_path: Path,
    benchmark_sha256: str,
) -> Path:
    shard_path = (output_dir / _field_shard_name(mode, repeat, fold)).resolve()
    prediction_path = _field_prediction_path(shard_path)
    sidecar_path = _sha_sidecar(shard_path)
    if shard_path.exists():
        if not resume:
            raise GateError(f"field shard already exists: {shard_path}")
        shard = _read_json(shard_path, "field shard")
        _validate_field_shard(
            shard, audit, mode, repeat, fold, shard_path, benchmark_path
        )
        return shard_path
    if prediction_path.exists() or sidecar_path.exists():
        raise GateError(f"incomplete/stale field fold artifacts: {shard_path}")

    started = time.perf_counter()
    training_ids, validation_ids, embargo_ids, group_by_well = _outer_roles(
        audit, mode, repeat, fold
    )
    sealed_suffixes = _load_incumbent_suffixes(audit, mode, repeat)
    crossfit_by_fold, fold_by_well, incumbent_metadata = _crossfit_incumbent_training(
        audit, training_ids, group_by_well
    )
    # Load training truth exactly once after outer roles are fixed. Validation
    # is loaded through the inference-only usecols path below.
    templates = _load_role_wells(
        audit,
        training_ids,
        {wid: sealed_suffixes[wid] for wid in training_ids},
        "training",
    )
    training_wells_by_fold = {
        inner_fold: {
            wid: _with_suffix(templates[wid], suffixes[wid]) for wid in training_ids
        }
        for inner_fold, suffixes in crossfit_by_fold.items()
    }
    selected, grid_metadata = _fit_field_grid(training_wells_by_fold, fold_by_well)
    final_training_wells = {
        wid: _with_suffix(templates[wid], crossfit_by_fold[int(fold_by_well[wid])][wid])
        for wid in training_ids
    }
    final_model = field_core.fit_structural_field(
        _as_training_wells(final_training_wells.values()),
        field_config(selected.h_ft, selected.laplacian),
    )
    _assert_no_coarsening(final_model)
    validation_wells = _load_role_wells(
        audit,
        validation_ids,
        {wid: sealed_suffixes[wid] for wid in validation_ids},
        "validation",
    )

    array_blocks: dict[str, list[NDArray[Any]]] = {
        name: [] for name in PREDICTION_ARRAYS
    }
    rows = []
    diagnostics = {}
    for well_index, well_id in enumerate(validation_ids):
        arrays, diag = _predict_candidate(
            final_model, selected.inner_models, selected, validation_wells[well_id]
        )
        n_rows = len(arrays["row_index"])
        array_blocks["well_index"].append(np.full(n_rows, well_index, dtype=np.int32))
        for name in PREDICTION_ARRAYS[1:]:
            array_blocks[name].append(arrays[name])
        rows.append(
            {
                "well": well_id,
                "well_index": well_index,
                "equality_group": group_by_well[well_id],
                "n_rows": n_rows,
            }
        )
        diagnostics[well_id] = diag
    sealed_arrays = {
        name: np.concatenate(blocks) for name, blocks in array_blocks.items()
    }
    _atomic_write_npz(prediction_path, sealed_arrays)
    learned = {
        "incumbent_crossfit": incumbent_metadata,
        "field_grid": grid_metadata,
        "selected_field_cell": {
            "h_ft": float(selected.h_ft),
            "laplacian": float(selected.laplacian),
            "theta_field": float(selected.theta_field),
            "theta_bias": float(selected.theta_bias),
        },
        "final_field_model": _model_metadata(final_model),
        "jackknife_inner_models": [
            _model_metadata(model) for model in selected.inner_models
        ],
        "support_query_truncation_count": 0,
    }
    shard = {
        "status": "MEASURE_ONLY_FIELD_PREDICTIONS_SEALED_TRUTH_UNREAD",
        "status_ceiling": STATUS_CEILING,
        "method": METHOD,
        "protocol_sha256": audit.protocol_sha256,
        "benchmark_file": benchmark_path.name,
        "benchmark_sha256": benchmark_sha256,
        "incumbent_inventory_sha256": audit.protocol["incumbent_pretruth_inventory"][
            "inventory_sha256"
        ],
        "mode": mode,
        "repeat": repeat,
        "fold": fold,
        "outer_role_sha256": {
            "training_ids": _id_digest(training_ids),
            "validation_ids": _id_digest(validation_ids),
            "embargo_ids": _id_digest(embargo_ids),
        },
        "training_well_count": len(training_ids),
        "validation_well_count": len(validation_ids),
        "embargo_well_count": len(embargo_ids),
        "learned_from_outer_training_only": learned,
        "validation_diagnostics": diagnostics,
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_logical_sha256": _logical_array_hash(sealed_arrays),
        "prediction_rows": int(len(sealed_arrays["row_index"])),
        "prediction_channels": list(PREDICTION_ARRAYS[2:]),
        "validation_wells": rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _validate_field_shard_schema(shard)
    _atomic_write_json(shard_path, shard)
    _write_sidecar(shard_path)
    _validate_field_shard(shard, audit, mode, repeat, fold, shard_path, benchmark_path)
    return shard_path


def _all_fold_identities() -> list[tuple[str, int, int]]:
    return [
        *(("exact", repeat, fold) for repeat in range(2) for fold in range(5)),
        *(("region", 0, fold) for fold in range(5)),
    ]


def run_folds(
    protocol_path: Path,
    benchmark_path: Path,
    output_dir: Path | None,
    folds: Sequence[tuple[str, int, int]],
    resume: bool,
) -> list[Path]:
    """Write metric-silent candidate shards; never score validation suffixes."""

    audit = audit_protocol(protocol_path, verify_data=True)
    benchmark_path = benchmark_path.resolve()
    benchmark_sha256 = _validate_benchmark(benchmark_path, audit)
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    normalized = []
    valid = set(_all_fold_identities())
    for identity in folds:
        if identity not in valid:
            raise GateError(f"invalid field fold identity: {identity}")
        if identity not in normalized:
            normalized.append(identity)
    pending = []
    completed = []
    for mode, repeat, fold in normalized:
        path = output_dir / _field_shard_name(mode, repeat, fold)
        if path.exists() and resume:
            shard = _read_json(path, "field shard")
            _validate_field_shard(
                shard, audit, mode, repeat, fold, path, benchmark_path
            )
            completed.append(path)
        else:
            pending.append((mode, repeat, fold))
    if pending:
        output_dir.mkdir(parents=True, exist_ok=True)
    for mode, repeat, fold in pending:
        print(
            f"running {mode} repeat={repeat} fold={fold}; validation metrics withheld",
            flush=True,
        )
        completed.append(
            _run_one_fold(
                audit,
                mode,
                repeat,
                fold,
                output_dir,
                resume,
                benchmark_path,
                benchmark_sha256,
            )
        )
        audit_protocol(protocol_path, verify_data=True)
    print(
        f"completed {len(pending)} field shard(s); validation truth remains unread",
        flush=True,
    )
    return completed


class _RssSampler:
    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = int(self._process.memory_info().rss)

    def __enter__(self) -> _RssSampler:
        def sample() -> None:
            while not self._stop.wait(0.1):
                self.peak_bytes = max(
                    self.peak_bytes, int(self._process.memory_info().rss)
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_bytes = max(self.peak_bytes, int(self._process.memory_info().rss))


def _largest_outer_identity(audit: GateAudit) -> tuple[str, int, int]:
    """Return the frozen inference-visible worst-work proxy identity."""

    row = _frozen_worst_proxy_row(audit)
    return str(row["mode"]), int(row["repeat"]), int(row["fold"])


def benchmark_gate(
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the frozen worst-work fold compute shape without validation scoring."""

    output_path = output_path.resolve()
    if output_path.exists() or _sha_sidecar(output_path).exists():
        raise GateError(f"refusing to overwrite benchmark artifact: {output_path}")
    total_start = time.perf_counter()
    with _RssSampler() as total_memory:
        audit = audit_protocol(protocol_path, verify_data=True)
        mode, repeat, fold = _largest_outer_identity(audit)
        training_ids, validation_ids, embargo_ids, group_by_well = _outer_roles(
            audit, mode, repeat, fold
        )
        sealed_suffixes = _load_incumbent_suffixes(audit, mode, repeat)
        incumbent_start = time.perf_counter()
        crossfit_by_fold, fold_by_well, _ = _crossfit_incumbent_training(
            audit, training_ids, group_by_well
        )
        incumbent_seconds = time.perf_counter() - incumbent_start
        templates = _load_role_wells(
            audit,
            training_ids,
            {wid: sealed_suffixes[wid] for wid in training_ids},
            "training",
        )
        training_wells_by_fold = {
            inner_fold: {
                wid: _with_suffix(templates[wid], suffixes[wid]) for wid in training_ids
            }
            for inner_fold, suffixes in crossfit_by_fold.items()
        }
        field_start = time.perf_counter()
        with _RssSampler() as field_memory:
            selected, grid_metadata = _fit_field_grid(
                training_wells_by_fold, fold_by_well
            )
            final_training_wells = {
                wid: _with_suffix(
                    templates[wid], crossfit_by_fold[fold_by_well[wid]][wid]
                )
                for wid in training_ids
            }
            final_model = field_core.fit_structural_field(
                _as_training_wells(final_training_wells.values()),
                field_config(selected.h_ft, selected.laplacian),
            )
            _assert_no_coarsening(final_model)
            validation = _load_role_wells(
                audit,
                validation_ids,
                {wid: sealed_suffixes[wid] for wid in validation_ids},
                "benchmark",
            )
            prediction_rows = 0
            for well_id in validation_ids:
                arrays, _ = _predict_candidate(
                    final_model,
                    selected.inner_models,
                    selected,
                    validation[well_id],
                )
                prediction_rows += len(arrays["row_index"])
        field_seconds = time.perf_counter() - field_start
    total_seconds = time.perf_counter() - total_start
    gib = float(1024**3)
    extrapolated = total_seconds * math.ceil(15 / 2)
    thresholds = _evaluation_contract()["runtime_acceptance"]
    acceptance = {
        "field_wall": field_seconds <= thresholds["field_wall_seconds_at_most"],
        "field_peak_rss": field_memory.peak_bytes / gib
        <= thresholds["field_peak_rss_gib_at_most"],
        "total_wall": total_seconds <= thresholds["total_wall_seconds_at_most"],
        "total_peak_rss": total_memory.peak_bytes / gib
        <= thresholds["total_peak_rss_gib_at_most"],
        "two_worker_fifteen_fold": extrapolated
        <= thresholds["extrapolated_two_worker_fifteen_fold_seconds_at_most"],
        "no_caps_solver_or_coarsening": True,
        "no_support_query_truncation": True,
    }
    result = {
        "status": "MEASURE_ONLY_TRUTH_QUARANTINED_BENCHMARK",
        "status_ceiling": STATUS_CEILING,
        "method": METHOD,
        "protocol_sha256": audit.protocol_sha256,
        "benchmark_work_proxy_sha256": audit.protocol["benchmark_work_proxy"][
            "proxy_sha256"
        ],
        "worst_work_identity": {
            "mode": mode,
            "repeat": repeat,
            "fold": fold,
            "training_wells": len(training_ids),
            "validation_wells": len(validation_ids),
            "embargo_wells": len(embargo_ids),
            "proxy_units": int(_frozen_worst_proxy_row(audit)["proxy_units"]),
        },
        "work_shape": {
            "strict_base_models": 10,
            "leave_one_base_models": 4,
            "leave_two_base_models": 6,
            "grid_cells": len(GRID),
            "strict_inner_field_models_per_cell": 10,
            "retained_outer_jackknife_models": INNER_FOLDS,
            "inner_field_fits": grid_metadata["inner_field_fits"],
            "selected_outer_refits": 1,
            "prediction_rows": prediction_rows,
            "solver_caps_changed": False,
            "coarsening_allowed": False,
            "support_query_truncation_count": 0,
        },
        "selected_cell": {
            "h_ft": selected.h_ft,
            "laplacian": selected.laplacian,
            "theta_field": selected.theta_field,
            "theta_bias": selected.theta_bias,
        },
        "timing_seconds": {
            "incumbent_crossfit": incumbent_seconds,
            "field": field_seconds,
            "total": total_seconds,
            "extrapolated_two_worker_fifteen_fold": extrapolated,
        },
        "memory_gib": {
            "field_peak_rss": field_memory.peak_bytes / gib,
            "total_peak_rss": total_memory.peak_bytes / gib,
        },
        "acceptance": acceptance,
        "all_acceptance_pass": all(acceptance.values()),
        "validation_metrics": "withheld; validation TVT was not parsed",
    }
    commit_audit = audit_protocol(protocol_path, verify_data=True)
    if commit_audit.protocol_sha256 != audit.protocol_sha256:
        raise GateError("protocol/source/data drifted before benchmark commit")
    _atomic_write_json(output_path, result)
    _write_sidecar(output_path)
    return result


def build_pretruth_field_inventory(
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path | None,
) -> dict[str, Any]:
    """Validate and inventory all 15 candidate shards before any truth boundary."""

    audit = audit_protocol(protocol_path, verify_data=True)
    benchmark_path = benchmark_path.resolve()
    benchmark_sha256 = _validate_benchmark(benchmark_path, audit)
    shard_dir = (
        shard_dir.resolve()
        if shard_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    expected_names = {
        _field_shard_name(mode, repeat, fold)
        for mode, repeat, fold in _all_fold_identities()
    }
    present_json = {path.name for path in shard_dir.glob("*.json")}
    present_npz = {path.name for path in shard_dir.glob("*.npz")}
    if present_json != expected_names or present_npz != {
        Path(name).with_suffix(".npz").name for name in expected_names
    }:
        raise GateError("all-shards-before-truth barrier is incomplete or unexpected")
    items = []
    for mode, repeat, fold in _all_fold_identities():
        shard_path = shard_dir / _field_shard_name(mode, repeat, fold)
        shard = _read_json(shard_path, "field shard")
        arrays = _validate_field_shard(
            shard, audit, mode, repeat, fold, shard_path, benchmark_path
        )
        prediction_path = _field_prediction_path(shard_path)
        items.append(
            {
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "metadata_name": shard_path.name,
                "metadata_sha256": sha256_file(shard_path),
                "metadata_sidecar_sha256": sha256_file(_sha_sidecar(shard_path)),
                "prediction_name": prediction_path.name,
                "prediction_sha256": sha256_file(prediction_path),
                "prediction_logical_sha256": _logical_array_hash(arrays),
            }
        )
    payload = {
        "protocol_sha256": audit.protocol_sha256,
        "benchmark_file": benchmark_path.name,
        "benchmark_sha256": benchmark_sha256,
        "incumbent_inventory_sha256": audit.protocol["incumbent_pretruth_inventory"][
            "inventory_sha256"
        ],
        "shard_count": len(items),
        "items": items,
    }
    return {**payload, "inventory_sha256": _canonical_digest(payload)}


def aggregate_barrier(
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path | None,
    inventory_output: Path,
) -> dict[str, Any]:
    """Persist the all-shard pretruth barrier; scoring follows independent audit."""

    protocol_path = protocol_path.resolve()
    benchmark_path = benchmark_path.resolve()
    resolved_shard_dir = (
        shard_dir.resolve()
        if shard_dir is not None
        else protocol_path.with_name(protocol_path.stem + "_folds")
    )
    inventory_output = inventory_output.resolve()
    try:
        inventory_output.relative_to(resolved_shard_dir)
    except ValueError:
        pass
    else:
        raise GateError("pretruth inventory output must be outside the shard directory")
    reserved = {
        protocol_path,
        _sha_sidecar(protocol_path),
        benchmark_path,
        _sha_sidecar(benchmark_path),
    }
    if inventory_output in reserved or _sha_sidecar(inventory_output) in reserved:
        raise GateError("pretruth inventory output aliases a frozen gate artifact")
    if inventory_output.exists() or _sha_sidecar(inventory_output).exists():
        raise GateError(f"refusing to overwrite pretruth inventory: {inventory_output}")
    inventory = build_pretruth_field_inventory(
        protocol_path, benchmark_path, resolved_shard_dir
    )
    sealed = {
        "status": "MEASURE_ONLY_ALL_FIELD_SHARDS_AUDITED_TRUTH_STILL_UNREAD",
        "status_ceiling": STATUS_CEILING,
        "pretruth_field_inventory": inventory,
        "next_phase": (
            "HOLD: aggregate scoring is intentionally staged after independent "
            "review of this prediction barrier"
        ),
    }
    _atomic_write_json(inventory_output, sealed)
    _write_sidecar(inventory_output)
    return sealed


def _parse_fold(value: str) -> tuple[str, int, int]:
    fields = value.split(":")
    try:
        if len(fields) == 3 and fields[0] == "exact":
            identity = ("exact", int(fields[1]), int(fields[2]))
        elif len(fields) == 2 and fields[0] == "region":
            identity = ("region", 0, int(fields[1]))
        else:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "fold must be exact:REPEAT:FOLD or region:FOLD"
        ) from exc
    if identity not in set(_all_fold_identities()):
        raise argparse.ArgumentTypeError(f"fold is outside the frozen set: {value}")
    return identity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze choices before field scoring")
    freeze.add_argument("--data-dir", type=Path, required=True)
    freeze.add_argument("--results-dir", type=Path, required=True)
    freeze.add_argument("--manifest-dir", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="audit frozen lineage and data bytes")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path)
    audit.add_argument("--results-dir", type=Path)
    audit.add_argument("--manifest-dir", type=Path)
    audit.add_argument("--no-data", action="store_true")

    run = subparsers.add_parser("run", help="write metric-silent field shards")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--fold", type=_parse_fold, action="append")
    run.add_argument("--resume", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="truth-quarantined frozen worst-work resource benchmark"
    )
    benchmark.add_argument("--protocol", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="seal the all-shard pretruth inventory; scoring remains HOLD"
    )
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--benchmark", type=Path, required=True)
    aggregate.add_argument("--shard-dir", type=Path)
    aggregate.add_argument("--inventory-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        protocol, sidecar = freeze_protocol(
            args.data_dir, args.results_dir, args.manifest_dir, args.protocol
        )
        print(f"froze MEASURE_ONLY field protocol {protocol}")
        print(f"wrote protocol identity {sidecar}")
    elif args.command == "audit":
        bundle = audit_protocol(
            args.protocol,
            verify_data=not args.no_data,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            manifest_dir=args.manifest_dir,
        )
        print(f"field audit passed: {bundle.protocol_sha256}")
    elif args.command == "run":
        run_folds(
            args.protocol,
            args.benchmark,
            args.output_dir,
            args.fold or _all_fold_identities(),
            args.resume,
        )
    elif args.command == "benchmark":
        result = benchmark_gate(args.protocol, args.output)
        print(json.dumps(result, indent=2), flush=True)
    elif args.command == "aggregate":
        result = aggregate_barrier(
            args.protocol, args.benchmark, args.shard_dir, args.inventory_output
        )
        print(json.dumps(result, indent=2), flush=True)
    else:  # pragma: no cover - argparse enforces the finite command set.
        raise GateError(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
