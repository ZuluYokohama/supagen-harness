# flake8: noqa: E501
"""Deterministic spatial split manifests for the wellbore competition.

The split constructor is deliberately label-blind.  From horizontal-well CSVs it
reads only ``MD``, ``X``, ``Y`` and the missingness mask of ``TVT_input``.  Exact
typewell equality is detected by hashing raw file bytes; typewell values are
never parsed.  No GR, suffix TVT, geology, formation-surface, or image content is
used.

Two split modes are provided:

``pad_out``
    Connect well centroids within 1,500 ft, union exact trajectory/typewell
    equality groups, then balance whole components over five folds by suffix-row
    count.

``region_out``
    Fit deterministic five-region KMeans in raw XY feet, keep exact equality
    groups indivisible, and embargo candidate training wells within 5,000 ft by
    centroid or 1,500 ft by a 100-MD-ft resampled trajectory point cloud.

Every returned/written manifest is canonicalized and sealed with SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


ALLOWED_HORIZONTAL_COLUMNS = ("MD", "X", "Y", "TVT_input")
DEFAULT_EXCLUDED_IDS = ("000d7d20", "00bbac68", "00e12e8b")
SCHEMA_VERSION = 1


class SplitConstructionError(RuntimeError):
    """Raised when requested spatial/equality constraints cannot coexist."""


def _require_positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise SplitConstructionError(f"{name} must be finite and positive; got {value}")


@dataclass(frozen=True)
class SplitConstraints:
    """Minimum support required for every emitted fold."""

    min_validation_wells: int = 100
    min_validation_suffix_rows: int = 400_000
    min_training_wells: int = 500
    min_training_suffix_rows: int = 2_400_000


@dataclass(frozen=True)
class WellGeometry:
    """Inference-safe geometry and equality fingerprints for one well."""

    well_id: str
    source_file: str
    typewell_file: str
    centroid_x: float
    centroid_y: float
    n_rows: int
    n_suffix_rows: int
    trajectory_sha256: str
    typewell_sha256: str
    resampled_xy: np.ndarray = field(repr=False, compare=False)
    resample_spacing_ft: float = 100.0

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray([self.centroid_x, self.centroid_y], dtype=float)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying a SHA-256 over its canonical unsealed payload."""

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    payload["manifest_sha256"] = digest
    return payload


def verify_manifest_sha256(manifest: Mapping[str, Any]) -> bool:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        return False
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return actual == expected


def write_manifest(manifest: Mapping[str, Any], output_path: Path | str) -> Path:
    """Write a stable, human-readable, hash-sealed JSON manifest."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal_manifest(manifest)
    path.write_text(
        json.dumps(sealed, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_float_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype=np.float64).copy(order="C")
    array[array == 0.0] = 0.0  # collapse -0.0
    array[np.isnan(array)] = np.nan  # canonical NumPy quiet NaN
    return np.asarray(array, dtype="<f8", order="C").tobytes(order="C")


def _trajectory_sha256(md: np.ndarray, x: np.ndarray, y: np.ndarray) -> str:
    matrix = np.column_stack((md, x, y))
    digest = hashlib.sha256()
    digest.update(b"MD,X,Y\0")
    digest.update(int(matrix.shape[0]).to_bytes(8, "little", signed=False))
    digest.update(_canonical_float_bytes(matrix))
    return digest.hexdigest()


def _resample_polyline(
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spacing: float,
) -> np.ndarray:
    valid = np.isfinite(md) & np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        raise SplitConstructionError("trajectory has fewer than two finite MD/XY rows")
    data = np.column_stack((md[valid], x[valid], y[valid]))
    data = data[np.argsort(data[:, 0], kind="mergesort")]
    unique_md, first = np.unique(data[:, 0], return_index=True)
    data = data[first]
    if len(unique_md) < 2 or unique_md[-1] <= unique_md[0]:
        raise SplitConstructionError("trajectory has no positive MD span")
    grid = np.arange(unique_md[0], unique_md[-1], spacing, dtype=float)
    if len(grid) == 0 or grid[0] != unique_md[0]:
        grid = np.insert(grid, 0, unique_md[0])
    if grid[-1] != unique_md[-1]:
        grid = np.append(grid, unique_md[-1])
    return np.column_stack(
        (
            np.interp(grid, unique_md, data[:, 1]),
            np.interp(grid, unique_md, data[:, 2]),
        )
    )


def _discover_horizontal_files(train_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    patterns = ("*__horizontal_well.csv", "*__horizontal_well_TRAIN.csv")
    for pattern in patterns:
        for path in sorted(train_dir.glob(pattern)):
            well_id = path.name.split("__", 1)[0]
            previous = paths.get(well_id)
            if previous is not None and previous.resolve() != path.resolve():
                raise SplitConstructionError(
                    f"multiple horizontal files discovered for {well_id}: "
                    f"{previous.name}, {path.name}"
                )
            paths[well_id] = path
    if not paths:
        raise SplitConstructionError(f"no horizontal-well CSVs found in {train_dir}")
    return paths


def _find_typewell_file(train_dir: Path, well_id: str) -> Path:
    matches = sorted(
        path
        for path in train_dir.glob(f"{well_id}__typewell*.csv")
        if "horizontal" not in path.name.lower()
    )
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise SplitConstructionError(
            f"expected exactly one typewell file for {well_id}; found {names}"
        )
    return matches[0]


def load_well_geometries(
    train_dir: Path | str,
    *,
    excluded_ids: Iterable[str] = DEFAULT_EXCLUDED_IDS,
    resample_spacing: float = 100.0,
) -> list[WellGeometry]:
    """Load only inference-safe geometry/mask columns and exact raw hashes."""

    _require_positive_finite("resample_spacing", resample_spacing)
    root = Path(train_dir)
    excluded = set(excluded_ids)
    horizontal = _discover_horizontal_files(root)
    wells: list[WellGeometry] = []
    for well_id in sorted(horizontal):
        if well_id in excluded:
            continue
        path = horizontal[well_id]
        try:
            frame = pd.read_csv(path, usecols=list(ALLOWED_HORIZONTAL_COLUMNS))
        except ValueError as exc:
            raise SplitConstructionError(
                f"{path.name} does not provide exactly the required split columns"
            ) from exc
        missing = sorted(set(ALLOWED_HORIZONTAL_COLUMNS) - set(frame.columns))
        if missing:
            raise SplitConstructionError(f"{path.name} missing columns: {missing}")
        md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(dtype=float)
        xy_valid = np.isfinite(x) & np.isfinite(y)
        if not xy_valid.any():
            raise SplitConstructionError(f"{path.name} has no finite XY rows")
        typewell = _find_typewell_file(root, well_id)
        wells.append(
            WellGeometry(
                well_id=well_id,
                source_file=path.name,
                typewell_file=typewell.name,
                centroid_x=float(np.mean(x[xy_valid])),
                centroid_y=float(np.mean(y[xy_valid])),
                n_rows=int(len(frame)),
                n_suffix_rows=int(frame["TVT_input"].isna().sum()),
                trajectory_sha256=_trajectory_sha256(md, x, y),
                typewell_sha256=_sha256_file(typewell),
                resampled_xy=_resample_polyline(md, x, y, resample_spacing),
                resample_spacing_ft=float(resample_spacing),
            )
        )
    if not wells:
        raise SplitConstructionError("all discovered wells were excluded")
    return wells


def _union_equal_hashes(
    wells: Sequence[WellGeometry],
) -> tuple[_UnionFind, dict[str, Any]]:
    union = _UnionFind(len(wells))
    diagnostics: dict[str, Any] = {}
    for attribute, label in (
        ("trajectory_sha256", "trajectory"),
        ("typewell_sha256", "typewell"),
    ):
        groups: dict[str, list[int]] = {}
        for index, well in enumerate(wells):
            groups.setdefault(getattr(well, attribute), []).append(index)
        duplicate = [indices for indices in groups.values() if len(indices) > 1]
        for indices in duplicate:
            for other in indices[1:]:
                union.union(indices[0], other)
        diagnostics[f"exact_{label}_duplicate_groups"] = len(duplicate)
        diagnostics[f"wells_in_exact_{label}_duplicate_groups"] = int(
            sum(len(indices) for indices in duplicate)
        )
        diagnostics[f"largest_exact_{label}_group"] = int(
            max((len(indices) for indices in duplicate), default=1)
        )
    return union, diagnostics


def _groups_from_union(union: _UnionFind, size: int) -> list[list[int]]:
    by_root: dict[int, list[int]] = {}
    for index in range(size):
        by_root.setdefault(union.find(index), []).append(index)
    return list(by_root.values())


def _stable_group_ids(
    wells: Sequence[WellGeometry], groups: Sequence[Sequence[int]], prefix: str
) -> tuple[list[str], dict[int, str]]:
    ordered = sorted(groups, key=lambda group: min(wells[i].well_id for i in group))
    ids: list[str] = []
    index_to_id: dict[int, str] = {}
    for number, group in enumerate(ordered):
        group_id = f"{prefix}_{number:04d}"
        ids.append(group_id)
        for index in group:
            index_to_id[index] = group_id
    return ids, index_to_id


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {key: None for key in ("min", "p10", "p50", "p90", "max")}
    result = np.quantile(finite, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        key: float(value)
        for key, value in zip(("min", "p10", "p50", "p90", "max"), result)
    }


def _nearest_polyline_distances(
    wells: Sequence[WellGeometry], query: Sequence[int], reference: Sequence[int]
) -> np.ndarray:
    if not query or not reference:
        return np.full(len(query), np.inf, dtype=float)
    reference_points = np.concatenate([wells[i].resampled_xy for i in reference])
    tree = cKDTree(reference_points)
    distances = []
    for index in query:
        point_distances, _ = tree.query(wells[index].resampled_xy, k=1)
        distances.append(float(np.min(point_distances)))
    return np.asarray(distances, dtype=float)


def _fold_diagnostics(
    wells: Sequence[WellGeometry],
    validation: Sequence[int],
    training: Sequence[int],
    embargo: Sequence[int],
) -> dict[str, Any]:
    xy = np.asarray([well.centroid for well in wells])
    if training:
        centroid_distances, _ = cKDTree(xy[list(training)]).query(
            xy[list(validation)], k=1
        )
    else:
        centroid_distances = np.full(len(validation), np.inf)
    polyline_distances = _nearest_polyline_distances(wells, validation, training)
    return {
        "validation_wells": len(validation),
        "validation_suffix_rows": int(sum(wells[i].n_suffix_rows for i in validation)),
        "training_wells": len(training),
        "training_suffix_rows": int(sum(wells[i].n_suffix_rows for i in training)),
        "embargo_wells": len(embargo),
        "embargo_suffix_rows": int(sum(wells[i].n_suffix_rows for i in embargo)),
        "nearest_training_centroid_ft": _quantiles(centroid_distances),
        "nearest_training_resampled_polyline_ft": _quantiles(polyline_distances),
    }


def _check_fold_support(
    diagnostics: Mapping[str, Any], constraints: SplitConstraints, fold: int
) -> None:
    checks = (
        ("validation_wells", constraints.min_validation_wells),
        ("validation_suffix_rows", constraints.min_validation_suffix_rows),
        ("training_wells", constraints.min_training_wells),
        ("training_suffix_rows", constraints.min_training_suffix_rows),
    )
    failures = [
        f"{name}={diagnostics[name]} < {minimum}"
        for name, minimum in checks
        if int(diagnostics[name]) < int(minimum)
    ]
    if failures:
        raise SplitConstructionError(
            f"fold {fold} violates frozen support constraints: {', '.join(failures)}"
        )


def _dataset_fingerprint(wells: Sequence[WellGeometry]) -> str:
    records = [
        {
            "well_id": well.well_id,
            "source_file": well.source_file,
            "typewell_file": well.typewell_file,
            "n_rows": well.n_rows,
            "n_suffix_rows": well.n_suffix_rows,
            "centroid_x": well.centroid_x,
            "centroid_y": well.centroid_y,
            "trajectory_sha256": well.trajectory_sha256,
            "typewell_sha256": well.typewell_sha256,
        }
        for well in wells
    ]
    return hashlib.sha256(_canonical_json_bytes({"wells": records})).hexdigest()


def _construction_packages() -> dict[str, str]:
    return {
        package: importlib_metadata.version(package)
        for package in ("numpy", "pandas", "scipy", "scikit-learn")
    }


def _base_manifest(
    mode: str,
    wells: Sequence[WellGeometry],
    excluded_ids: Iterable[str],
    hash_diagnostics: Mapping[str, Any],
    equality_group_by_index: Mapping[int, str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    well_ids = [well.well_id for well in wells]
    if len(well_ids) != len(set(well_ids)):
        raise SplitConstructionError("input contains duplicate well IDs")
    declared_exclusions = set(excluded_ids)
    leaked_exclusions = sorted(declared_exclusions & set(well_ids))
    if leaked_exclusions:
        raise SplitConstructionError(
            f"declared excluded IDs remain in the split population: {leaked_exclusions}"
        )
    spacings = {well.resample_spacing_ft for well in wells}
    if len(spacings) != 1:
        raise SplitConstructionError(
            "wells carry inconsistent trajectory resampling spacings"
        )
    resample_spacing_ft = float(next(iter(spacings)))
    well_records = []
    for index, well in enumerate(wells):
        well_records.append(
            {
                "well_id": well.well_id,
                "source_file": well.source_file,
                "typewell_file": well.typewell_file,
                "centroid_x": well.centroid_x,
                "centroid_y": well.centroid_y,
                "n_rows": well.n_rows,
                "n_suffix_rows": well.n_suffix_rows,
                "trajectory_sha256": well.trajectory_sha256,
                "typewell_sha256": well.typewell_sha256,
                "equality_group": equality_group_by_index[index],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "construction_columns": list(ALLOWED_HORIZONTAL_COLUMNS),
        "tvt_input_usage": "missingness mask only",
        "typewell_usage": "raw-file SHA256 only; values are not parsed",
        "resampled_polyline_spacing_ft": resample_spacing_ft,
        "excluded_ids": sorted(declared_exclusions),
        "dataset_sha256": _dataset_fingerprint(wells),
        "construction_provenance": {
            "source_file": "research/spatial_split.py",
            "source_sha256": _sha256_file(Path(__file__).resolve()),
            "packages": _construction_packages(),
        },
        "parameters": dict(parameters),
        "diagnostics": {
            "n_wells": len(wells),
            "n_suffix_rows": int(sum(well.n_suffix_rows for well in wells)),
            **dict(hash_diagnostics),
        },
        "wells": well_records,
        "folds": [],
    }


def _validate_hash_group_roles(
    equality_groups: Sequence[Sequence[int]],
    validation: set[int],
    training: set[int],
    embargo: set[int],
    fold: int,
) -> None:
    for group in equality_groups:
        roles = {
            "validation"
            if index in validation
            else "training"
            if index in training
            else "embargo"
            if index in embargo
            else "missing"
            for index in group
        }
        if len(roles) != 1:
            raise SplitConstructionError(
                f"fold {fold} splits an exact trajectory/typewell equality group: "
                f"{sorted(roles)}"
            )


def build_pad_out_manifest(
    wells: Sequence[WellGeometry],
    *,
    excluded_ids: Iterable[str] = DEFAULT_EXCLUDED_IDS,
    radius_ft: float = 1_500.0,
    n_folds: int = 5,
    constraints: SplitConstraints = SplitConstraints(),
) -> dict[str, Any]:
    """Build balanced, component-indivisible ``pad_out`` folds."""

    _require_positive_finite("radius_ft", radius_ft)
    if n_folds < 2:
        raise SplitConstructionError("n_folds must be at least 2")
    wells = sorted(wells, key=lambda well: well.well_id)
    equality_union, hash_diagnostics = _union_equal_hashes(wells)
    equality_groups = _groups_from_union(equality_union, len(wells))
    _, equality_group_by_index = _stable_group_ids(wells, equality_groups, "equality")

    xy = np.asarray([well.centroid for well in wells])
    for left, right in cKDTree(xy).query_pairs(radius_ft, output_type="ndarray"):
        equality_union.union(int(left), int(right))
    components = _groups_from_union(equality_union, len(wells))
    if len(components) < n_folds:
        raise SplitConstructionError(
            f"only {len(components)} indivisible pad components for {n_folds} folds"
        )
    _, component_by_index = _stable_group_ids(wells, components, "pad")

    ordered_components = sorted(
        components,
        key=lambda component: (
            -sum(wells[i].n_suffix_rows for i in component),
            min(wells[i].well_id for i in component),
        ),
    )
    fold_rows = [0] * n_folds
    fold_wells = [0] * n_folds
    fold_by_index: dict[int, int] = {}
    for component in ordered_components:
        fold = min(
            range(n_folds),
            key=lambda candidate: (
                fold_rows[candidate],
                fold_wells[candidate],
                candidate,
            ),
        )
        for index in component:
            fold_by_index[index] = fold
        fold_rows[fold] += sum(wells[i].n_suffix_rows for i in component)
        fold_wells[fold] += len(component)

    manifest = _base_manifest(
        "pad_out",
        wells,
        excluded_ids,
        hash_diagnostics,
        equality_group_by_index,
        {
            "centroid_component_radius_ft": radius_ft,
            "n_folds": n_folds,
            "trajectory_isolation": False,
            "interpretation": (
                "centroid-component pad-out only; cross-fold laterals may approach or cross"
            ),
            "component_assignment": (
                "descending suffix rows, tie min well ID; place in fold with "
                "fewest suffix rows, then wells, then fold number"
            ),
            "constraints": constraints.__dict__,
        },
    )
    for index, record in enumerate(manifest["wells"]):
        record["pad_component"] = component_by_index[index]
        record["validation_fold"] = fold_by_index[index]

    universe = set(range(len(wells)))
    fold_records = []
    for fold in range(n_folds):
        validation = {i for i in universe if fold_by_index[i] == fold}
        training = universe - validation
        embargo: set[int] = set()
        _validate_hash_group_roles(equality_groups, validation, training, embargo, fold)
        diagnostics = _fold_diagnostics(
            wells, sorted(validation), sorted(training), sorted(embargo)
        )
        _check_fold_support(diagnostics, constraints, fold)
        fold_records.append(
            {
                "fold": fold,
                "validation_ids": sorted(wells[i].well_id for i in validation),
                "training_ids": sorted(wells[i].well_id for i in training),
                "embargo_ids": [],
                "diagnostics": diagnostics,
            }
        )
    manifest["folds"] = fold_records
    manifest["diagnostics"].update(
        {
            "pad_components": len(components),
            "singleton_pad_components": int(
                sum(len(component) == 1 for component in components)
            ),
            "largest_pad_component": int(max(map(len, components))),
        }
    )
    return seal_manifest(manifest)


def _collective_region_labels(
    wells: Sequence[WellGeometry],
    equality_groups: Sequence[Sequence[int]],
    n_folds: int,
    seed: int,
    n_init: int,
    coordinate_scale_ft: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    xy = np.asarray([well.centroid for well in wells])
    center = np.median(xy, axis=0)
    normalized = (xy - center) / coordinate_scale_ft
    model = KMeans(
        n_clusters=n_folds,
        random_state=seed,
        n_init=n_init,
        algorithm="lloyd",
    )
    initial = model.fit_predict(normalized)
    labels = initial.copy()
    forced = 0
    for group in equality_groups:
        points = normalized[list(group)]
        costs = np.sum(
            np.sum(
                (points[:, None, :] - model.cluster_centers_[None, :, :]) ** 2, axis=2
            ),
            axis=0,
        )
        collective = int(np.argmin(costs))
        forced += int(np.sum(labels[list(group)] != collective))
        labels[list(group)] = collective

    # Canonical fold numbering by raw-space center X then Y.
    raw_centers = model.cluster_centers_ * coordinate_scale_ft + center
    order = sorted(range(n_folds), key=lambda label: tuple(raw_centers[label]))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    raw_centers = raw_centers[order]
    if set(labels) != set(range(n_folds)):
        raise SplitConstructionError(
            "exact equality constraints emptied a KMeans region; refusing split"
        )
    return labels, raw_centers, forced


def _region_embargo(
    wells: Sequence[WellGeometry],
    validation: set[int],
    candidates: set[int],
    equality_groups: Sequence[Sequence[int]],
    centroid_buffer_ft: float,
    polyline_buffer_ft: float,
) -> tuple[set[int], dict[str, int]]:
    xy = np.asarray([well.centroid for well in wells])
    validation_list = sorted(validation)
    candidate_list = sorted(candidates)
    centroid_tree = cKDTree(xy[validation_list])
    centroid_distance, _ = centroid_tree.query(xy[candidate_list], k=1)
    centroid_embargo = {
        index
        for index, distance in zip(candidate_list, centroid_distance)
        if float(distance) <= centroid_buffer_ft
    }

    validation_points = np.concatenate(
        [wells[index].resampled_xy for index in validation_list]
    )
    point_tree = cKDTree(validation_points)
    polyline_embargo: set[int] = set()
    for index in candidate_list:
        distances, _ = point_tree.query(wells[index].resampled_xy, k=1)
        if float(np.min(distances)) <= polyline_buffer_ft:
            polyline_embargo.add(index)

    embargo = centroid_embargo | polyline_embargo
    before_propagation = set(embargo)
    for group in equality_groups:
        group_set = set(group)
        if group_set & embargo:
            if group_set & validation:
                raise SplitConstructionError(
                    "an exact equality group crosses validation and embargo candidates"
                )
            embargo.update(group_set)
    if not embargo <= candidates:
        raise SplitConstructionError(
            "embargo propagation escaped candidate training set"
        )
    return embargo, {
        "centroid_embargo_wells": len(centroid_embargo),
        "polyline_embargo_wells": len(polyline_embargo),
        "equality_propagation_additional_wells": len(embargo - before_propagation),
    }


def build_region_out_manifest(
    wells: Sequence[WellGeometry],
    *,
    excluded_ids: Iterable[str] = DEFAULT_EXCLUDED_IDS,
    n_folds: int = 5,
    seed: int = 20260810,
    n_init: int = 50,
    coordinate_scale_ft: float = 10_000.0,
    centroid_buffer_ft: float = 5_000.0,
    polyline_buffer_ft: float = 1_500.0,
    constraints: SplitConstraints = SplitConstraints(),
) -> dict[str, Any]:
    """Build buffered, exact-group-indivisible ``region_out`` folds."""

    if n_folds < 2:
        raise SplitConstructionError("n_folds must be at least 2")
    if n_init < 1:
        raise SplitConstructionError("n_init must be at least 1")
    _require_positive_finite("coordinate_scale_ft", coordinate_scale_ft)
    _require_positive_finite("centroid_buffer_ft", centroid_buffer_ft)
    _require_positive_finite("polyline_buffer_ft", polyline_buffer_ft)
    wells = sorted(wells, key=lambda well: well.well_id)
    equality_union, hash_diagnostics = _union_equal_hashes(wells)
    equality_groups = _groups_from_union(equality_union, len(wells))
    _, equality_group_by_index = _stable_group_ids(wells, equality_groups, "equality")
    labels, raw_centers, forced = _collective_region_labels(
        wells,
        equality_groups,
        n_folds,
        seed,
        n_init,
        coordinate_scale_ft,
    )
    manifest = _base_manifest(
        "region_out",
        wells,
        excluded_ids,
        hash_diagnostics,
        equality_group_by_index,
        {
            "n_folds": n_folds,
            "kmeans_seed": seed,
            "kmeans_n_init": n_init,
            "kmeans_algorithm": "lloyd",
            "coordinate_metric": "raw Euclidean XY feet",
            "coordinate_center": "componentwise median",
            "coordinate_scale_ft": coordinate_scale_ft,
            "centroid_embargo_ft": centroid_buffer_ft,
            "resampled_polyline_embargo_ft": polyline_buffer_ft,
            "trajectory_isolation": True,
            "constraints": constraints.__dict__,
        },
    )
    for index, record in enumerate(manifest["wells"]):
        record["validation_fold"] = int(labels[index])

    fold_records = []
    universe = set(range(len(wells)))
    for fold in range(n_folds):
        validation = {i for i in universe if int(labels[i]) == fold}
        candidates = universe - validation
        embargo, embargo_diagnostics = _region_embargo(
            wells,
            validation,
            candidates,
            equality_groups,
            centroid_buffer_ft,
            polyline_buffer_ft,
        )
        training = candidates - embargo
        _validate_hash_group_roles(equality_groups, validation, training, embargo, fold)
        diagnostics = _fold_diagnostics(
            wells, sorted(validation), sorted(training), sorted(embargo)
        )
        diagnostics.update(embargo_diagnostics)
        centroid_min = diagnostics["nearest_training_centroid_ft"]["min"]
        polyline_min = diagnostics["nearest_training_resampled_polyline_ft"]["min"]
        if centroid_min is None or centroid_min <= centroid_buffer_ft:
            raise SplitConstructionError(
                f"fold {fold} failed the {centroid_buffer_ft:g}-ft centroid embargo"
            )
        if polyline_min is None or polyline_min <= polyline_buffer_ft:
            raise SplitConstructionError(
                f"fold {fold} failed the {polyline_buffer_ft:g}-ft polyline embargo"
            )
        _check_fold_support(diagnostics, constraints, fold)
        fold_records.append(
            {
                "fold": fold,
                "region_center_x": float(raw_centers[fold, 0]),
                "region_center_y": float(raw_centers[fold, 1]),
                "validation_ids": sorted(wells[i].well_id for i in validation),
                "training_ids": sorted(wells[i].well_id for i in training),
                "embargo_ids": sorted(wells[i].well_id for i in embargo),
                "diagnostics": diagnostics,
            }
        )
    manifest["folds"] = fold_records
    manifest["diagnostics"].update(
        {
            "collective_equality_label_reassignments": forced,
            "region_sizes": [int(np.sum(labels == fold)) for fold in range(n_folds)],
        }
    )
    return seal_manifest(manifest)


def _constraints_from_args(args: argparse.Namespace) -> SplitConstraints:
    return SplitConstraints(
        min_validation_wells=args.min_validation_wells,
        min_validation_suffix_rows=args.min_validation_suffix_rows,
        min_training_wells=args.min_training_wells,
        min_training_suffix_rows=args.min_training_suffix_rows,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("pad_out", "region_out", "both"), default="both"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON path for one mode, or output directory when --mode=both",
    )
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--no-default-exclusions", action="store_true")
    parser.add_argument("--min-validation-wells", type=int, default=100)
    parser.add_argument("--min-validation-suffix-rows", type=int, default=400_000)
    parser.add_argument("--min-training-wells", type=int, default=500)
    parser.add_argument("--min-training-suffix-rows", type=int, default=2_400_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    excluded = set(args.exclude_id)
    if not args.no_default_exclusions:
        excluded.update(DEFAULT_EXCLUDED_IDS)
    wells = load_well_geometries(args.train_dir, excluded_ids=excluded)
    constraints = _constraints_from_args(args)
    builders = {
        "pad_out": lambda: build_pad_out_manifest(
            wells, excluded_ids=excluded, constraints=constraints
        ),
        "region_out": lambda: build_region_out_manifest(
            wells, excluded_ids=excluded, constraints=constraints
        ),
    }
    modes = ("pad_out", "region_out") if args.mode == "both" else (args.mode,)
    for mode in modes:
        manifest = builders[mode]()
        output = args.output / f"{mode}.json" if args.mode == "both" else args.output
        write_manifest(manifest, output)
        print(f"{mode}: {output} sha256={manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
