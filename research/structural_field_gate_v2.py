"""Sealed fixed-physics v2 gate for the anchored structural field.

V2 is a fresh, write-once ``MEASURE_ONLY`` experiment descended from the
sealed v1 STOP record.  It does not select a field scale, learn amplitude
coefficients, refit the incumbent, or inspect validation TVT.  Four
geometry-only exact-profile GroupKFold roles provide derivative-dispersion
proposals, one all-training model provides the final field, and every target
uses the already-sealed role-exact base path as the core fallback policy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
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

from research import structural_field as field_core
from research import structural_field_gate as legacy


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "geosteern-anchored-structural-field-gate/2"
METHOD = "anchored_differential_field_fixed_identity_over_sealed_joint_v2"
STATUS_CEILING = "MEASURE_ONLY"
SHARD_STATUS = "MEASURE_ONLY_FIELD_V2_PREDICTIONS_SEALED_TRUTH_UNREAD"
BENCHMARK_STATUS = "MEASURE_ONLY_V2_TRUTH_QUARANTINED_BENCHMARK"
BARRIER_STATUS = "MEASURE_ONLY_ALL_FIELD_V2_SHARDS_AUDITED_TRUTH_STILL_UNREAD"

FIXED_H_FT = 15_000.0
FIXED_INDUCING_CELL_FT = 7_500.0
FIXED_LAPLACIAN = 3.0
THETA_FIELD = 1.0
THETA_BIAS = 1.0
INNER_FOLDS = 4
FIELD_FITS_PER_OUTER = 5
VALIDATION_PROPOSALS_PER_WELL = 5
OUTER_EXACT_REPEATS = 2
OUTER_FOLDS = 5
MODES = ("exact", "region")

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
INFERENCE_COLUMNS = legacy.INFERENCE_COLUMNS
TRAINING_COLUMNS = legacy.TRAINING_COLUMNS

V1_PROTOCOL_NAME = "anchored_structural_field_protocol.json"
V1_STOP_NAME = "anchored_structural_field_v1_STOP.json"
V1_STOP_SHA256 = "ea20d2104640a38b1de43e2e89dcb2a75f645481aaf3cb875737ea53db64f760"
V1_STOP_REASON_CODE = "EXTERNAL_TIMEOUT_BEFORE_BENCHMARK_ATOMIC_COMMIT"
V1_NONMUTATION_STATEMENT = (
    "V2 is a fresh method and protocol; it cannot alter, overwrite, erase, "
    "reinterpret, or promote the sealed v1 STOP lineage."
)
V1_ABSENT_DOWNSTREAM_NAMES = (
    "anchored_structural_field_benchmark.json",
    "anchored_structural_field_benchmark.json.sha256",
    "anchored_structural_field_pretruth_inventory.json",
    "anchored_structural_field_pretruth_inventory.json.sha256",
    "anchored_structural_field_protocol_pretruth_inventory.json",
    "anchored_structural_field_protocol_pretruth_inventory.json.sha256",
    "anchored_structural_field_score.json",
    "anchored_structural_field_score.json.sha256",
    "anchored_structural_field_score_per_well.csv",
    "anchored_structural_field_score_per_well.csv.sha256",
)
V1_ABSENT_DOWNSTREAM_DIRECTORIES = ("anchored_structural_field_protocol_folds",)

V1_SOURCE_SHA256 = {
    "research/structural_field.py": (
        "2c3203ba336bf30c501f1c5fdfb242412b3d4625a010eec3c729773c3dcc736e"
    ),
    "research/test_structural_field.py": (
        "ef8b081279705f5c7a2f1625d5a27e9769c0d6ad3f484684092bd5fc40f385e1"
    ),
    "research/structural_field_gate.py": (
        "3975dc1b83ea065d30de19f112dc2901687427307b3b980910c6684d1836399a"
    ),
    "research/test_structural_field_gate.py": (
        "661ed5fa0532ccf633402aeee33dc305efd68ebd116f6b57d027821517f9d059"
    ),
    "research/structural_field_score.py": (
        "6f5ea31f13181c63306e818764a7281aa04ae194a4a71b052ae5d59fcc8ed640"
    ),
    "research/test_structural_field_score.py": (
        "f68a39645b03789dfd9b5e36f916e7128f586e72f8f1c5b58f570a79ee6202e6"
    ),
}

RUNTIME_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *legacy.RUNTIME_SOURCE_FILES,
            "research/structural_field_gate_v2.py",
            "research/test_structural_field_gate_v2.py",
            "research/structural_field_score_v2.py",
            "research/test_structural_field_score_v2.py",
        )
    )
)
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "psutil",
)
LEGACY_HELPER_ALLOWLIST = frozenset(
    {
        "ArtifactDescriptor",
        "BOOTSTRAP_DRAWS",
        "BOOTSTRAP_SEED",
        "EXACT_MANIFEST_NAME",
        "EXACT_PROTOCOL_NAME",
        "EXACT_SHARD_DIR_NAME",
        "GateAudit",
        "GateError",
        "GateHold",
        "INFERENCE_COLUMNS",
        "IncumbentSuffix",
        "METHOD",
        "PAD_MANIFEST_NAME",
        "PROTOCOL_VERSION",
        "REGION_MANIFEST_NAME",
        "RUNTIME_SOURCE_FILES",
        "SPATIAL_INVENTORY_NAME",
        "SPATIAL_PROTOCOL_NAME",
        "SPATIAL_SHARD_DIR_NAME",
        "TRAINING_COLUMNS",
        "WellPath",
        "_as_training_wells",
        "_assert_no_coarsening",
        "_atomic_write_json",
        "_atomic_write_npz",
        "_audit_data_files",
        "_canonical_digest",
        "_compose_well",
        "_finite_nonnegative",
        "_incumbent_inventory",
        "_inventory_by_well",
        "_is_sha256",
        "_known_prefix",
        "_load_role_wells",
        "_logical_array_hash",
        "_model_metadata",
        "_outer_roles",
        "_prediction_diagnostics",
        "_read_json",
        "_read_sidecar",
        "_read_well_csv",
        "_reject_sensitive_field_names",
        "_require_exact_keys",
        "_safe_basename",
        "_sha_sidecar",
        "_support_query_truncation_count",
        "_validate_descriptor",
        "_validate_inventory_payload",
        "_validate_model_metadata_schema",
        "_well_file",
        "_write_sidecar",
        "audit_protocol",
        "incumbent_spatial",
        "jackknife_confidence",
        "sha256_file",
    }
)

EXACT_PROTOCOL_NAME = legacy.EXACT_PROTOCOL_NAME
EXACT_MANIFEST_NAME = legacy.EXACT_MANIFEST_NAME
EXACT_SHARD_DIR_NAME = legacy.EXACT_SHARD_DIR_NAME
SPATIAL_PROTOCOL_NAME = legacy.SPATIAL_PROTOCOL_NAME
SPATIAL_INVENTORY_NAME = legacy.SPATIAL_INVENTORY_NAME
SPATIAL_SHARD_DIR_NAME = legacy.SPATIAL_SHARD_DIR_NAME
REGION_MANIFEST_NAME = legacy.REGION_MANIFEST_NAME
PAD_MANIFEST_NAME = legacy.PAD_MANIFEST_NAME

JACKKNIFE_DEFINITION = (
    "The four geometry-only leave-one-role field models are evaluated on each "
    "validation suffix. At each raw suffix row, use the audited v1 persistent-"
    "support derivative definition for field_delta_without_prefix_bias: forward "
    "at suffix start, central in the interior, backward at TD. With at least "
    "three supported values, tau=1.4826*MAD; otherwise cJ=0. Sigma is the final "
    "all-outer-training derivative_residual_scale floored at 1e-8. "
    "cJ=1/(1+(tau/sigma)^2). Prefix-bias derivative is excluded."
)


# Reused v1 primitives are deliberately exported under the v2 namespace.  The
# v1 source and tests are six fixed hash identities in every v2 protocol.
GateError = legacy.GateError
GateHold = legacy.GateHold
ArtifactDescriptor = legacy.ArtifactDescriptor
GateAudit = legacy.GateAudit
IncumbentSuffix = legacy.IncumbentSuffix
WellPath = legacy.WellPath
sha256_file = legacy.sha256_file
_sha_sidecar = legacy._sha_sidecar
_is_sha256 = legacy._is_sha256
_canonical_digest = legacy._canonical_digest
_logical_array_hash = legacy._logical_array_hash
_atomic_write_json = legacy._atomic_write_json
_atomic_write_npz = legacy._atomic_write_npz
_write_sidecar = legacy._write_sidecar
_read_sidecar = legacy._read_sidecar
_safe_basename = legacy._safe_basename
_read_json = legacy._read_json
_require_exact_keys = legacy._require_exact_keys
_reject_sensitive_field_names = legacy._reject_sensitive_field_names
_finite_nonnegative = legacy._finite_nonnegative
_inventory_by_well = legacy._inventory_by_well
_well_file = legacy._well_file
_read_well_csv = legacy._read_well_csv
_known_prefix = legacy._known_prefix
_compose_well = legacy._compose_well
_load_role_wells = legacy._load_role_wells
_as_training_wells = legacy._as_training_wells
_model_metadata = legacy._model_metadata
_prediction_diagnostics = legacy._prediction_diagnostics


@dataclass(frozen=True)
class RolePartition:
    fold_by_well: dict[str, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FixedFieldFit:
    leave_one_models: tuple[field_core.StructuralFieldModel, ...]
    final_model: field_core.StructuralFieldModel
    metadata: dict[str, Any]


def fixed_configuration() -> dict[str, Any]:
    """Return the one physical configuration and identity coefficients."""

    return {
        "physical_h_ft": FIXED_H_FT,
        "effective_field_config": asdict(field_config()),
        "theta_field": THETA_FIELD,
        "theta_bias": THETA_BIAS,
    }


def field_config() -> field_core.FieldConfig:
    config = field_core.FieldConfig(
        resample_step_md=100.0,
        inducing_cell_ft=FIXED_INDUCING_CELL_FT,
        support_length_ft=FIXED_H_FT,
        graph_neighbors=6,
        graph_max_edge_ft=22_500.0,
        interpolation_neighbors=6,
        laplacian_strength=FIXED_LAPLACIAN,
        circulation_strength=0.1,
        ridge_strength=1.0e-6,
        huber_delta=1.5,
        discontinuity_mad_threshold=4.0,
        cut_fallback_radius_ft=500.0,
        min_effective_wells=1.5,
        min_directional_observability=0.05,
        max_distinct_support_wells=16,
        max_support_neighbors=4_096,
        blend_alpha=1.0,
    )
    if (
        config.inducing_cell_ft != FIXED_INDUCING_CELL_FT
        or config.support_length_ft != FIXED_H_FT
        or config.laplacian_strength != FIXED_LAPLACIAN
    ):
        raise GateError("fixed field configuration drift")
    return config


def _assert_no_coarsening(model: field_core.StructuralFieldModel) -> None:
    legacy._assert_no_coarsening(model)


def _geometry_vector(well: WellPath) -> tuple[float, ...]:
    arrays = [
        np.asarray(well.md, dtype=float),
        np.asarray(well.x, dtype=float),
        np.asarray(well.y, dtype=float),
        np.asarray(well.z, dtype=float),
    ]
    if (
        any(value.ndim != 1 for value in arrays)
        or len({len(value) for value in arrays}) != 1
    ):
        raise GateError("inner-role inference geometry shape drift")
    if not arrays[0].size or any(not np.isfinite(value).all() for value in arrays):
        raise GateError("inner-role inference geometry is empty or non-finite")
    md, x, y, z = arrays
    if not np.all(np.diff(md) > 0.0):
        raise GateError("inner-role MD is not strictly increasing")
    return (
        float(x[0]),
        float(y[0]),
        float(z[0]),
        float(x[-1]),
        float(y[-1]),
        float(z[-1]),
        float(np.mean(x)),
        float(np.mean(y)),
        float(md[-1] - md[0]),
    )


def _partition_from_geometry(
    well_ids: Sequence[str],
    group_by_well: Mapping[str, str],
    geometry_by_well: Mapping[str, Sequence[float]],
) -> RolePartition:
    ordered = sorted(map(str, well_ids))
    if (
        len(ordered) != len(set(ordered))
        or set(ordered) != set(map(str, group_by_well))
        or set(ordered) != set(map(str, geometry_by_well))
    ):
        raise GateError("inner-role group membership differs from outer training")
    groups = np.asarray([str(group_by_well[well_id]) for well_id in ordered])
    if len(set(groups.tolist())) < INNER_FOLDS:
        raise GateError("inner-role split requires at least four exact-profile groups")
    geometry = np.asarray(
        [geometry_by_well[well_id] for well_id in ordered], dtype=float
    )
    if geometry.ndim != 2 or not np.isfinite(geometry).all():
        raise GateError("inner-role geometry matrix is invalid")
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    fold_by_well: dict[str, int] = {}
    roles: list[dict[str, Any]] = []
    for fold, (fitting_index, excluded_index) in enumerate(
        splitter.split(geometry, groups=groups)
    ):
        fitting = [ordered[int(index)] for index in fitting_index]
        excluded = [ordered[int(index)] for index in excluded_index]
        if not fitting or not excluded or set(fitting) & set(excluded):
            raise GateError("inner-role leave-one membership is empty or aliased")
        for well_id in excluded:
            if well_id in fold_by_well:
                raise GateError("inner-role well was excluded more than once")
            fold_by_well[well_id] = fold
        roles.append(
            {
                "excluded_fold": fold,
                "fitting_well_count": len(fitting),
                "excluded_well_count": len(excluded),
                "fitting_ids_sha256": _id_digest(fitting),
                "excluded_ids_sha256": _id_digest(excluded),
            }
        )
    if set(fold_by_well) != set(ordered) or set(fold_by_well.values()) != set(
        range(INNER_FOLDS)
    ):
        raise GateError("inner-role split is incomplete")
    for group in set(groups.tolist()):
        folds = {
            fold_by_well[well_id]
            for well_id in ordered
            if str(group_by_well[well_id]) == group
        }
        if len(folds) != 1:
            raise GateError("exact-profile group crossed an inner boundary")
    metadata = {
        "partition_basis": (
            "deterministic GroupKFold(n_splits=4), one inference-geometry row per "
            "outer-training well, groups=exact-profile identity, y omitted"
        ),
        "inner_folds": INNER_FOLDS,
        "fold_by_well_sha256": _canonical_digest(
            [[well_id, fold_by_well[well_id]] for well_id in ordered]
        ),
        "leave_one_roles": roles,
    }
    return RolePartition(fold_by_well=fold_by_well, metadata=metadata)


def _inner_fold_roles(
    wells: Mapping[str, WellPath], group_by_well: Mapping[str, str]
) -> RolePartition:
    geometry = {well_id: _geometry_vector(well) for well_id, well in wells.items()}
    return _partition_from_geometry(list(wells), group_by_well, geometry)


def _fit_fixed_models(
    training_wells: Mapping[str, WellPath], partition: RolePartition
) -> FixedFieldFit:
    if set(training_wells) != set(partition.fold_by_well):
        raise GateError("fixed field roles differ from outer training membership")
    config = field_config()
    leave_one_models = []
    for fold in range(INNER_FOLDS):
        fitting = [
            training_wells[well_id]
            for well_id in sorted(training_wells)
            if partition.fold_by_well[well_id] != fold
        ]
        excluded = [
            well_id
            for well_id in sorted(training_wells)
            if partition.fold_by_well[well_id] == fold
        ]
        if not fitting or not excluded:
            raise GateError("fixed leave-one field role is empty")
        model = field_core.fit_structural_field(_as_training_wells(fitting), config)
        _assert_no_coarsening(model)
        leave_one_models.append(model)
    final_ids = sorted(training_wells)
    final_model = field_core.fit_structural_field(
        _as_training_wells(training_wells[well_id] for well_id in final_ids), config
    )
    _assert_no_coarsening(final_model)
    if len(leave_one_models) != INNER_FOLDS:
        raise GateError("fixed field fit did not produce four leave-one models")
    metadata = {
        "field_fit_count": FIELD_FITS_PER_OUTER,
        "validation_proposals_per_well": VALIDATION_PROPOSALS_PER_WELL,
        "role_partition": partition.metadata,
        "final_role": {
            "fitting_well_count": len(final_ids),
            "fitting_ids_sha256": _id_digest(final_ids),
        },
    }
    return FixedFieldFit(
        leave_one_models=tuple(leave_one_models),
        final_model=final_model,
        metadata=metadata,
    )


def _predict_core(
    model: field_core.StructuralFieldModel, well: WellPath
) -> field_core.StructuralPrediction:
    truncations = legacy._support_query_truncation_count(model, well)
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


def jackknife_confidence(
    md: NDArray[np.float64],
    proposals: Sequence[field_core.StructuralPrediction],
    sigma: float,
    suffix_start: int,
) -> NDArray[np.float64]:
    return legacy.jackknife_confidence(md, proposals, sigma, suffix_start)


def _predict_candidate(
    final_model: field_core.StructuralFieldModel,
    leave_one_models: Sequence[field_core.StructuralFieldModel],
    well: WellPath,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    if len(leave_one_models) != INNER_FOLDS:
        raise GateError("candidate requires exactly four leave-one field models")
    final = _predict_core(final_model, well)
    leave_one = [_predict_core(model, well) for model in leave_one_models]
    c_j = jackknife_confidence(
        well.md,
        leave_one,
        final_model.diagnostics.derivative_residual_scale,
        int(well.suffix_index[0]),
    )
    confidence = np.asarray(final.confidence, dtype=float) * c_j
    candidate = well.joint_full + confidence * (
        np.asarray(final.field_delta_without_prefix_bias_tvt, dtype=float)
        + np.asarray(final.prefix_bias_delta_tvt, dtype=float)
    )
    prefix = int(well.suffix_index[0])
    candidate[:prefix] = well.tvt_input[:prefix]
    idx = well.suffix_index
    arrays = {
        "row_index": idx.astype(np.int32),
        "base_prediction": well.base_full[idx].astype(np.float64),
        "joint_prediction": well.joint_full[idx].astype(np.float64),
        "candidate_prediction": candidate[idx].astype(np.float64),
        "field_confidence": confidence[idx].astype(np.float64),
        "field_delta_without_prefix_bias": np.asarray(
            final.field_delta_without_prefix_bias_tvt[idx], dtype=np.float64
        ),
        "prefix_bias_delta": np.asarray(
            final.prefix_bias_delta_tvt[idx], dtype=np.float64
        ),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise GateError(f"non-finite v2 candidate artifact for {well.well_id}")
    expected = arrays["joint_prediction"] + arrays["field_confidence"] * (
        arrays["field_delta_without_prefix_bias"] + arrays["prefix_bias_delta"]
    )
    if not np.allclose(
        arrays["candidate_prediction"], expected, rtol=0.0, atol=1.0e-10
    ):
        raise GateError("fixed identity candidate formula drift")
    diagnostics = {
        **_prediction_diagnostics(final),
        "mean_jackknife_confidence": float(np.mean(c_j[idx])),
        "mean_final_confidence": float(np.mean(confidence[idx])),
        "supported_fraction": float(np.mean(confidence[idx] > 0.0)),
    }
    return arrays, diagnostics


def _validate_prediction_arrays(
    arrays: Mapping[str, NDArray[Any]],
    validation_ids: Sequence[str],
    sealed_suffixes: Mapping[str, IncumbentSuffix],
) -> None:
    if set(arrays) != set(PREDICTION_ARRAYS):
        raise GateError("v2 field prediction array schema drift")
    lengths = {len(np.asarray(value)) for value in arrays.values()}
    if len(lengths) != 1:
        raise GateError("v2 field prediction array length drift")
    if (
        arrays["well_index"].dtype.kind not in "iu"
        or arrays["row_index"].dtype.kind not in "iu"
    ):
        raise GateError("v2 prediction identity arrays are not integers")
    for name in PREDICTION_ARRAYS[2:]:
        if not np.isfinite(arrays[name]).all():
            raise GateError(f"v2 field prediction contains non-finite {name}")
    confidence = np.asarray(arrays["field_confidence"], dtype=float)
    if np.any(confidence < 0.0) or np.any(confidence > 1.0 + 1.0e-12):
        raise GateError("v2 field confidence escaped [0, 1]")
    formula = arrays["joint_prediction"] + confidence * (
        arrays["field_delta_without_prefix_bias"] + arrays["prefix_bias_delta"]
    )
    if not np.allclose(arrays["candidate_prediction"], formula, rtol=0.0, atol=1.0e-10):
        raise GateError("v2 fixed identity candidate formula drift")
    if list(map(str, validation_ids)) != list(validation_ids):
        raise GateError("validation well identity type drift")
    for well_index, well_id in enumerate(validation_ids):
        if well_id not in sealed_suffixes:
            raise GateError("role-exact incumbent is missing a validation well")
        selected = np.asarray(arrays["well_index"]) == well_index
        expected = sealed_suffixes[well_id]
        if (
            not np.array_equal(arrays["row_index"][selected], expected.row_index)
            or not np.array_equal(arrays["base_prediction"][selected], expected.base)
            or not np.array_equal(arrays["joint_prediction"][selected], expected.joint)
        ):
            raise GateError("v2 comparator differs from role-exact incumbent")
    valid_indices = set(range(len(validation_ids)))
    if set(map(int, np.unique(arrays["well_index"]))) != valid_indices:
        raise GateError("v2 well-index coverage drift")


def _id_digest(values: Sequence[str]) -> str:
    return _canonical_digest(list(map(str, values)))


def _all_fold_identities() -> list[tuple[str, int, int]]:
    return [
        *(("exact", repeat, fold) for repeat in range(2) for fold in range(5)),
        *(("region", 0, fold) for fold in range(5)),
    ]


def _outer_roles(
    audit: GateAudit, mode: str, repeat: int, fold: int
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    training, validation, embargo, groups = legacy._outer_roles(
        audit, mode, repeat, fold
    )
    if mode == "region":
        matches = [
            row
            for row in audit.region_manifest.get("folds", [])
            if int(row.get("fold", -1)) == fold
        ]
        if len(matches) != 1 or training != [
            str(value) for value in matches[0]["training_ids"]
        ]:
            raise GateError("region training role differs from manifest.training_ids")
        if set(embargo) & (set(training) | set(validation)):
            raise GateError("region embargo entered a live v2 role")
    return training, validation, embargo, groups


def _load_incumbent_suffixes(
    audit: GateAudit,
    mode: str,
    repeat: int,
    requested_ids: Sequence[str] | None = None,
) -> dict[str, IncumbentSuffix]:
    """Load only requested sealed incumbent paths for an active outer role.

    The containing incumbent archives remain freshly byte/logically audited by
    ``audit_protocol``.  This loader does not retain entries for embargo or
    unrelated population wells when an explicit role is supplied.
    """

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
        raise GateError("v2 incumbent shard selection is incomplete")
    wanted = (
        set(map(str, requested_ids))
        if requested_ids is not None
        else set(_inventory_by_well(audit).index.astype(str))
    )
    if not wanted:
        raise GateError("v2 incumbent requested role is empty")
    result: dict[str, IncumbentSuffix] = {}
    for item in items:
        metadata_path = legacy._validate_descriptor(
            item["metadata"], audit.results_dir, audit.manifest_dir
        )
        prediction_path = legacy._validate_descriptor(
            item["prediction"], audit.results_dir, audit.manifest_dir
        )
        shard = _read_json(metadata_path, "incumbent shard")
        rows = shard.get("test_wells")
        if not isinstance(rows, list):
            raise GateError("incumbent shard lacks well metadata")
        requested_rows = [
            (index, row)
            for index, row in enumerate(rows)
            if str(row.get("well", "")) in wanted
        ]
        if not requested_rows:
            continue
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not {
                "well_index",
                "row_index",
                "base_prediction",
                "joint_prediction",
            }.issubset(archive.files):
                raise GateError("incumbent prediction channel schema drift")
            well_index_array = np.asarray(archive["well_index"])
            for expected_index, row in requested_rows:
                if int(row.get("well_index", -1)) != expected_index:
                    raise GateError("incumbent well-index metadata drift")
                well_id = str(row.get("well", ""))
                selected = well_index_array == expected_index
                if int(selected.sum()) != int(row.get("n_rows", -1)):
                    raise GateError("incumbent per-well row count drift")
                if well_id in result:
                    raise GateError("incumbent well appears in multiple folds")
                result[well_id] = IncumbentSuffix(
                    row_index=np.asarray(
                        archive["row_index"][selected], dtype=np.int64
                    ),
                    base=np.asarray(
                        archive["base_prediction"][selected], dtype=np.float64
                    ),
                    joint=np.asarray(
                        archive["joint_prediction"][selected], dtype=np.float64
                    ),
                )
    if set(result) != wanted:
        missing = sorted(wanted - set(result))
        extra = sorted(set(result) - wanted)
        raise GateError(
            f"v2 role-filtered incumbent coverage drift: missing={missing}, extra={extra}"
        )
    return result


def _work_proxy_well_stats(
    data_dir: Path, inventory: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    train_root = (data_dir / "train").resolve()
    config = field_config()
    stats: dict[str, dict[str, Any]] = {}
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
            md, config.resample_step_md, config.max_resampled_intervals_per_well
        )
        xk = np.interp(knots, md, x)
        yk = np.interp(knots, md, y)
        zk = np.interp(knots, md, z)
        delta_md = np.diff(knots)
        unit = np.column_stack((np.diff(xk) / delta_md, np.diff(yk) / delta_md))
        lateral = (np.abs(np.diff(zk) / delta_md) <= config.lateral_max_abs_dz_dmd) & (
            np.linalg.norm(unit, axis=1) >= config.min_horizontal_speed
        )
        midpoint = np.column_stack(
            ((xk[:-1] + xk[1:]) / 2.0, (yk[:-1] + yk[1:]) / 2.0)
        )[lateral]
        if len(midpoint) == 0:
            raise GateHold(f"work proxy well has no lateral observations: {well_id}")
        per_well_nodes, actual_cell = field_core._inducing_nodes(midpoint, config)
        if not math.isclose(
            actual_cell, FIXED_INDUCING_CELL_FT, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise GateHold("work proxy detected forbidden inducing-cell coarsening")
        prefix = _known_prefix(tvt_input)
        anchor_md = float(md[prefix - 1])
        prediction_knots = field_core._prediction_knots(md, anchor_md, config)
        prediction_prefix = int(
            np.searchsorted(prediction_knots, anchor_md, side="right")
        )
        bias_start = int(
            np.searchsorted(
                prediction_knots,
                anchor_md - config.prefix_bias_window_md,
                side="left",
            )
        )
        stats[well_id] = {
            "derivative_observations": int(lateral.sum()),
            "support_queries": int(
                len(prediction_knots) + max(0, prediction_prefix - 1 - bias_start)
            ),
            "fixed_inducing_cells": len(per_well_nodes),
            "midpoints": tuple(
                (float(point[0]), float(point[1])) for point in midpoint
            ),
            "geometry": (
                float(x[0]),
                float(y[0]),
                float(z[0]),
                float(x[-1]),
                float(y[-1]),
                float(z[-1]),
                float(np.mean(x)),
                float(np.mean(y)),
                float(md[-1] - md[0]),
            ),
        }
    return stats


def _work_proxy_definition() -> dict[str, Any]:
    return {
        "columns": list(INFERENCE_COLUMNS),
        "role_mass": (
            "sum independent derivative-observation and exact fixed-7500ft inducing-"
            "node "
            "mass over four exact-profile leave-one roles plus one all-training role"
        ),
        "derivative_weight": 8,
        "fixed_node_weight": 800,
        "validation_proposals_per_well": VALIDATION_PROPOSALS_PER_WELL,
        "validation_query_weight_per_proposal": 4_096,
        "formula": (
            "8*derivative_fit_mass + 800*fixed_node_fit_mass + "
            "4096*5*validation_support_queries"
        ),
        "selection": "maximum proxy_units with frozen deterministic tie break",
        "incumbent_work": "zero; sealed incumbent arrays are loaded, never refit",
    }


def _work_proxy_order_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["proxy_units"]),
        int(row["derivative_fit_mass"]),
        int(row["fixed_node_fit_mass"]),
        int(row["five_validation_proposal_queries"]),
        str(row["mode"]) == "region",
        -int(row["repeat"]),
        -int(row["fold"]),
    )


def _compute_work_proxy(
    data_dir: Path, results_dir: Path, manifest_dir: Path
) -> dict[str, Any]:
    exact_manifest = pd.read_csv(_safe_basename(results_dir, EXACT_MANIFEST_NAME))
    inventory = legacy.incumbent_spatial._validate_inventory(
        pd.read_csv(_safe_basename(results_dir, SPATIAL_INVENTORY_NAME))
    )
    region_manifest = _read_json(
        _safe_basename(manifest_dir, REGION_MANIFEST_NAME), "region manifest"
    )
    audit = GateAudit(
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
    inventory_index = inventory.copy()
    inventory_index["well"] = inventory_index["well"].astype(str)
    inventory_index = inventory_index.set_index("well", drop=False)
    rows: list[dict[str, Any]] = []
    for mode, repeat, fold in _all_fold_identities():
        training, validation, embargo, group_by_well = _outer_roles(
            audit, mode, repeat, fold
        )
        training_groups = {well_id: group_by_well[well_id] for well_id in training}
        geometry = {well_id: stats[well_id]["geometry"] for well_id in training}
        partition = _partition_from_geometry(training, training_groups, geometry)
        leave_obs: list[int] = []
        leave_nodes: list[int] = []
        leave_counts: list[int] = []
        leave_digests: list[str] = []
        for inner_fold in range(INNER_FOLDS):
            fitting = [
                well_id
                for well_id in sorted(training)
                if partition.fold_by_well[well_id] != inner_fold
            ]
            role_midpoints = np.vstack(
                [
                    np.asarray(stats[well_id]["midpoints"], dtype=float)
                    for well_id in fitting
                ]
            )
            role_nodes, actual_cell = field_core._inducing_nodes(
                role_midpoints, field_config()
            )
            if not math.isclose(
                actual_cell, FIXED_INDUCING_CELL_FT, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise GateHold("work proxy detected forbidden role coarsening")
            leave_obs.append(
                sum(stats[well_id]["derivative_observations"] for well_id in fitting)
            )
            leave_nodes.append(len(role_nodes))
            leave_counts.append(len(fitting))
            leave_digests.append(_id_digest(fitting))
        final_midpoints = np.vstack(
            [
                np.asarray(stats[well_id]["midpoints"], dtype=float)
                for well_id in training
            ]
        )
        final_nodes, final_actual_cell = field_core._inducing_nodes(
            final_midpoints, field_config()
        )
        if not math.isclose(
            final_actual_cell,
            FIXED_INDUCING_CELL_FT,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise GateHold("work proxy detected forbidden final-role coarsening")
        final_obs = sum(
            stats[well_id]["derivative_observations"] for well_id in training
        )
        derivative_mass = sum(leave_obs) + final_obs
        node_mass = sum(leave_nodes) + len(final_nodes)
        validation_queries = sum(
            stats[well_id]["support_queries"] for well_id in validation
        )
        validation_suffix_rows = sum(
            int(inventory_index.loc[well_id, "suffix_rows"]) for well_id in validation
        )
        five_queries = VALIDATION_PROPOSALS_PER_WELL * validation_queries
        proxy_units = 8 * derivative_mass + 800 * node_mass + 4_096 * five_queries
        rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "training_wells": len(training),
                "validation_wells": len(validation),
                "embargo_wells": len(embargo),
                "inner_role_partition_sha256": partition.metadata[
                    "fold_by_well_sha256"
                ],
                "leave_one_training_wells": leave_counts,
                "leave_one_role_sha256": leave_digests,
                "final_role_sha256": _id_digest(sorted(training)),
                "leave_one_derivative_observations": leave_obs,
                "final_derivative_observations": final_obs,
                "derivative_fit_mass": derivative_mass,
                "leave_one_fixed_node_count": leave_nodes,
                "final_fixed_node_count": len(final_nodes),
                "fixed_node_fit_mass": node_mass,
                "validation_support_queries": validation_queries,
                "validation_suffix_rows": validation_suffix_rows,
                "five_validation_proposal_queries": five_queries,
                "proxy_units": int(proxy_units),
            }
        )
    rows.sort(key=lambda item: (item["mode"], item["repeat"], item["fold"]))
    maximizing = max(rows, key=_work_proxy_order_key)
    payload = {
        "definition": _work_proxy_definition(),
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
        raise GateError("v2 benchmark work-proxy top-level schema drift")
    payload = {key: value for key, value in proxy.items() if key != "proxy_sha256"}
    if proxy.get("proxy_sha256") != _canonical_digest(payload):
        raise GateError("v2 benchmark work-proxy logical digest drift")
    if proxy.get("definition") != _work_proxy_definition():
        raise GateError("v2 benchmark work-proxy definition drift")
    rows = proxy.get("folds")
    if not isinstance(rows, list) or len(rows) != 15:
        raise GateError("v2 benchmark work-proxy fold inventory drift")
    keys = {
        "mode",
        "repeat",
        "fold",
        "training_wells",
        "validation_wells",
        "embargo_wells",
        "inner_role_partition_sha256",
        "leave_one_training_wells",
        "leave_one_role_sha256",
        "final_role_sha256",
        "leave_one_derivative_observations",
        "final_derivative_observations",
        "derivative_fit_mass",
        "leave_one_fixed_node_count",
        "final_fixed_node_count",
        "fixed_node_fit_mass",
        "validation_support_queries",
        "validation_suffix_rows",
        "five_validation_proposal_queries",
        "proxy_units",
    }
    identities = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != keys:
            raise GateError("v2 benchmark work-proxy fold schema drift")
        identity = (str(row["mode"]), int(row["repeat"]), int(row["fold"]))
        identities.append(identity)
        if identity not in set(_all_fold_identities()):
            raise GateError("v2 benchmark work-proxy contains an invalid fold")
        list_fields = (
            "leave_one_training_wells",
            "leave_one_role_sha256",
            "leave_one_derivative_observations",
            "leave_one_fixed_node_count",
        )
        if any(
            not isinstance(row[name], list) or len(row[name]) != 4
            for name in list_fields
        ):
            raise GateError("v2 work-proxy leave-one role inventory drift")
        if (
            not _is_sha256(row["inner_role_partition_sha256"])
            or any(not _is_sha256(value) for value in row["leave_one_role_sha256"])
            or not _is_sha256(row["final_role_sha256"])
        ):
            raise GateError("v2 work-proxy role digest drift")
        numeric = [
            row["training_wells"],
            row["validation_wells"],
            row["embargo_wells"],
            *row["leave_one_training_wells"],
            *row["leave_one_derivative_observations"],
            row["final_derivative_observations"],
            row["derivative_fit_mass"],
            *row["leave_one_fixed_node_count"],
            row["final_fixed_node_count"],
            row["fixed_node_fit_mass"],
            row["validation_support_queries"],
            row["validation_suffix_rows"],
            row["five_validation_proposal_queries"],
            row["proxy_units"],
        ]
        if any(
            isinstance(value, bool) or int(value) != value or int(value) < 0
            for value in numeric
        ):
            raise GateError("v2 work-proxy contains an invalid count")
        if any(
            int(value) >= int(row["training_wells"])
            for value in row["leave_one_training_wells"]
        ):
            raise GateError("v2 work-proxy leave-one role does not exclude a block")
        if (
            sum(map(int, row["leave_one_training_wells"]))
            != 3 * int(row["training_wells"])
            or len(set(map(str, row["leave_one_role_sha256"]))) != INNER_FOLDS
        ):
            raise GateError("v2 work-proxy leave-one roles do not form one partition")
        if int(row["validation_suffix_rows"]) <= 0:
            raise GateError("v2 work-proxy validation suffix is empty")
        derivative_mass = sum(map(int, row["leave_one_derivative_observations"])) + int(
            row["final_derivative_observations"]
        )
        node_mass = sum(map(int, row["leave_one_fixed_node_count"])) + int(
            row["final_fixed_node_count"]
        )
        five_queries = VALIDATION_PROPOSALS_PER_WELL * int(
            row["validation_support_queries"]
        )
        units = 8 * derivative_mass + 800 * node_mass + 4_096 * five_queries
        if (
            int(row["derivative_fit_mass"]) != derivative_mass
            or int(row["fixed_node_fit_mass"]) != node_mass
            or int(row["five_validation_proposal_queries"]) != five_queries
            or int(row["proxy_units"]) != units
        ):
            raise GateError("v2 benchmark work-proxy arithmetic drift")
    if set(identities) != set(_all_fold_identities()) or len(set(identities)) != 15:
        raise GateError("v2 benchmark work-proxy identities are incomplete")
    maximum = max(rows, key=_work_proxy_order_key)
    expected = {
        "mode": maximum["mode"],
        "repeat": maximum["repeat"],
        "fold": maximum["fold"],
    }
    if proxy.get("maximizing_identity") != expected:
        raise GateError("v2 benchmark work-proxy maximizing identity drift")


def _evaluation_contract() -> dict[str, Any]:
    return {
        "status_ceiling": STATUS_CEILING,
        "fresh_method": METHOD,
        "parent_v1_disposition": "STOP is immutable and cannot be promoted by v2",
        "fixed_physical_configuration": fixed_configuration(),
        "fixed_coefficients": {
            "theta_field": THETA_FIELD,
            "theta_bias": THETA_BIAS,
        },
        "candidate_formula": (
            "sealed_joint + final_core_confidence*jackknife_confidence*"
            "(field_delta_without_prefix_bias + prefix_bias_delta)"
        ),
        "outer_incumbent": (
            "already-sealed role-exact base and joint arrays; no incumbent model or "
            "path evidence is recomputed"
        ),
        "inner_partition": (
            "deterministic four-way GroupKFold over exact-profile groups using only "
            "outer-training IDs and inference geometry; y is omitted"
        ),
        "field_models_per_outer_split": {
            "leave_one_true_tvt_models": INNER_FOLDS,
            "all_outer_training_true_tvt_models": 1,
            "total": FIELD_FITS_PER_OUTER,
        },
        "validation_proposals_per_well": VALIDATION_PROPOSALS_PER_WELL,
        "jackknife_confidence": JACKKNIFE_DEFINITION,
        "validation_columns": list(INFERENCE_COLUMNS),
        "region_role": (
            "fit exactly manifest.training_ids; validation_ids are inference-only; "
            "embargo_ids enter no partition, fit, prediction, or diagnostic"
        ),
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
            "all 15 exact-allowlisted v2 JSON/NPZ shards must pass fresh byte, "
            "logical, membership, formula, comparator, lineage, and schema audit "
            "before the source-bound v2 scorer can open validation TVT"
        ),
        "bootstrap": {
            "exact": {
                "unit": "exact typewell profile group",
                "repeat_coupling": (
                    "one sampled group multiplicity is applied to all wells in both "
                    "repeats; each draw averages repeat-wise pooled RMSE gains"
                ),
                "draws": legacy.BOOTSTRAP_DRAWS,
                "seed": legacy.BOOTSTRAP_SEED,
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
            "point_worsening": (
                "candidate well-RMSE p90 minus sealed-joint well-RMSE p90"
            ),
            "one_sided_bound": (
                "95th percentile of bootstrap point-worsening values using the same "
                "mode-specific resampling multiplicities"
            ),
        },
        "runtime_acceptance": {
            "field_wall_seconds_at_most": 600.0,
            "field_peak_rss_gib_at_most": 6.0,
            "total_wall_seconds_at_most": 900.0,
            "total_peak_rss_gib_at_most": 10.0,
            "extrapolated_two_worker_fifteen_fold_seconds_at_most": 7_200.0,
            "extrapolation": (
                "measured worst-fold total including two live audits multiplied by "
                "ceil(15/2)"
            ),
            "caps_solver_or_coarsening": ("forbidden; inducing coarsening is STOP"),
        },
        "source_bound_scorer": (
            "research/structural_field_score_v2.py and its tests must exist and be "
            "hash-bound before freeze"
        ),
    }


def _packages() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def _source_hashes() -> dict[str, str]:
    paths = {name: ROOT / name for name in RUNTIME_SOURCE_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise GateError(
            "v2 runtime source is missing; scorer binding is required before freeze: "
            f"{missing}"
        )
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in V1_SOURCE_SHA256.items():
        if hashes.get(name) != expected:
            raise GateError(f"sealed v1 source/test identity drift: {name}")
    return hashes


def _artifact_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise GateError(f"required parent artifact is missing: {path}")
    sidecar = _sha_sidecar(path)
    if _read_sidecar(path).lower() != sha256_file(path):
        raise GateError(f"parent artifact sidecar drift: {path.name}")
    return {
        "name": path.name,
        "size_bytes": int(path.stat().st_size),
        "byte_sha256": sha256_file(path),
        "sidecar_name": sidecar.name,
        "sidecar_sha256": sha256_file(sidecar),
    }


def _parent_lineage(results_dir: Path) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    protocol_path = _safe_basename(results_dir, V1_PROTOCOL_NAME)
    stop_path = _safe_basename(results_dir, V1_STOP_NAME)
    protocol_binding = _artifact_binding(protocol_path)
    stop_binding = _artifact_binding(stop_path)
    if stop_binding["byte_sha256"] != V1_STOP_SHA256:
        raise GateError("sealed v1 STOP byte identity drift")
    stop = _read_json(stop_path, "sealed v1 STOP record")
    protocol = _read_json(protocol_path, "sealed v1 protocol")
    stop_protocol = stop.get("protocol")
    if not isinstance(stop_protocol, Mapping):
        raise GateError("sealed v1 STOP lacks protocol lineage")
    if (
        stop.get("record_version") != "geosteern-anchored-structural-field-stop/1"
        or stop.get("status") != "STOP"
        or stop.get("status_ceiling") != STATUS_CEILING
        or stop.get("reason_code") != V1_STOP_REASON_CODE
        or stop.get("method") != legacy.METHOD
        or stop.get("decision")
        != "V1 remains STOP; RUN, AGGREGATE, and SCORE remain closed."
        or stop_protocol.get("file") != V1_PROTOCOL_NAME
        or stop_protocol.get("version") != legacy.PROTOCOL_VERSION
        or stop_protocol.get("sha256") != protocol_binding["byte_sha256"]
        or stop_protocol.get("sidecar_verified") is not True
        or protocol.get("protocol_version") != legacy.PROTOCOL_VERSION
        or protocol.get("method") != legacy.METHOD
    ):
        raise GateError("sealed v1 STOP meaning or parent protocol drift")
    downstream = stop.get("downstream_state")
    if not isinstance(downstream, Mapping) or (
        downstream.get("prediction_shards") != 0
        or downstream.get("pretruth_inventory_exists") is not False
        or downstream.get("anchored_field_score_artifacts") != 0
        or downstream.get("validation_truth_scoring_performed") is not False
    ):
        raise GateError("sealed v1 STOP downstream state drift")
    current_v1_hashes = {
        name: sha256_file(ROOT / name) for name in sorted(V1_SOURCE_SHA256)
    }
    if current_v1_hashes != {
        name: V1_SOURCE_SHA256[name] for name in sorted(V1_SOURCE_SHA256)
    }:
        raise GateError("one of six sealed v1 source/test hashes drifted")
    downstream_paths = [
        _safe_basename(results_dir, name) for name in V1_ABSENT_DOWNSTREAM_NAMES
    ]
    downstream_directories = [
        (results_dir / name).resolve() for name in V1_ABSENT_DOWNSTREAM_DIRECTORIES
    ]
    present = [
        path.name
        for path in [*downstream_paths, *downstream_directories]
        if path.exists()
    ]
    if present:
        raise GateError(
            f"sealed v1 STOP now has forbidden downstream artifacts: {sorted(present)}"
        )
    payload = {
        "v1_protocol": protocol_binding,
        "v1_stop": stop_binding,
        "v1_stop_reason_code": V1_STOP_REASON_CODE,
        "v1_method": legacy.METHOD,
        "v1_source_and_test_sha256": current_v1_hashes,
        "live_absent_v1_downstream_names": [
            *V1_ABSENT_DOWNSTREAM_NAMES,
            *V1_ABSENT_DOWNSTREAM_DIRECTORIES,
        ],
        "fresh_method_nonmutation_statement": V1_NONMUTATION_STATEMENT,
    }
    return {**payload, "lineage_sha256": _canonical_digest(payload)}


def _descriptor_paths(
    value: object, results_dir: Path, manifest_dir: Path
) -> set[Path]:
    found: set[Path] = set()
    if isinstance(value, Mapping):
        if set(value) == {
            "scope",
            "name",
            "size_bytes",
            "byte_sha256",
            "logical_sha256",
        }:
            found.add(legacy._validate_descriptor(value, results_dir, manifest_dir))
        else:
            for nested in value.values():
                found.update(_descriptor_paths(nested, results_dir, manifest_dir))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_descriptor_paths(nested, results_dir, manifest_dir))
    return found


def _reserved_frozen_paths(
    results_dir: Path,
    manifest_dir: Path,
    incumbent_inventory: Mapping[str, Any] | None = None,
) -> set[Path]:
    reserved = {
        _safe_basename(results_dir, V1_PROTOCOL_NAME),
        _sha_sidecar(_safe_basename(results_dir, V1_PROTOCOL_NAME)),
        _safe_basename(results_dir, V1_STOP_NAME),
        _sha_sidecar(_safe_basename(results_dir, V1_STOP_NAME)),
    }
    if incumbent_inventory is not None:
        descriptors = _descriptor_paths(
            incumbent_inventory, results_dir.resolve(), manifest_dir.resolve()
        )
        reserved.update(descriptors)
        reserved.update(
            _sha_sidecar(path) for path in descriptors if _sha_sidecar(path).exists()
        )
    return {path.resolve() for path in reserved}


def _assert_write_target(path: Path, reserved: Iterable[Path]) -> None:
    resolved = path.resolve()
    sidecar = _sha_sidecar(resolved)
    frozen = {item.resolve() for item in reserved}
    if resolved in frozen or sidecar in frozen:
        raise GateError(
            f"v2 output aliases a frozen parent/incumbent artifact: {resolved}"
        )
    if resolved.exists() or sidecar.exists():
        raise GateError(f"refusing to overwrite v2 artifact: {resolved}")


def _assert_outside_tree(path: Path, tree: Path, label: str) -> None:
    resolved = path.resolve()
    root = tree.resolve()
    if resolved == root or root in resolved.parents:
        raise GateError(f"{label} cannot enter reserved tree: {root}")


def _assert_artifact_outside_frozen_trees(path: Path, results_dir: Path) -> None:
    for tree in (
        results_dir / EXACT_SHARD_DIR_NAME,
        results_dir / SPATIAL_SHARD_DIR_NAME,
        results_dir / "anchored_structural_field_protocol_folds",
    ):
        _assert_outside_tree(path, tree, "v2 artifact")


def freeze_protocol(
    data_dir: Path,
    results_dir: Path,
    manifest_dir: Path,
    protocol_path: Path,
) -> tuple[Path, Path]:
    """Freeze the fresh v2 method without opening any validation suffix TVT."""

    data_dir = data_dir.resolve()
    results_dir = results_dir.resolve()
    manifest_dir = manifest_dir.resolve()
    protocol_path = protocol_path.resolve()
    sources = _source_hashes()
    incumbent_inventory = legacy._incumbent_inventory(results_dir, manifest_dir)
    reserved = _reserved_frozen_paths(results_dir, manifest_dir, incumbent_inventory)
    _assert_artifact_outside_frozen_trees(protocol_path, results_dir)
    _assert_write_target(protocol_path, reserved)
    parent = _parent_lineage(results_dir)
    parent_audit = legacy.audit_protocol(
        _safe_basename(results_dir, V1_PROTOCOL_NAME),
        verify_data=True,
        data_dir=data_dir,
        results_dir=results_dir,
        manifest_dir=manifest_dir,
    )
    if parent_audit.protocol_sha256 != parent["v1_protocol"]["byte_sha256"]:
        raise GateError("live v1 protocol audit differs from bound parent lineage")
    spatial_inventory = legacy.incumbent_spatial._validate_inventory(
        pd.read_csv(_safe_basename(results_dir, SPATIAL_INVENTORY_NAME))
    )
    legacy._audit_data_files(spatial_inventory, data_dir)
    work_proxy = _compute_work_proxy(data_dir, results_dir, manifest_dir)
    _validate_work_proxy_payload(work_proxy)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_V2_FIELD_SCORING_MEASURE_ONLY",
        "status_ceiling": STATUS_CEILING,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
            "manifest_dir": str(manifest_dir),
        },
        "source_sha256": sources,
        "packages": _packages(),
        "parent_v1_lineage": parent,
        "fixed_configuration": fixed_configuration(),
        "evaluation": _evaluation_contract(),
        "benchmark_work_proxy": work_proxy,
        "incumbent_pretruth_inventory": incumbent_inventory,
        "notes": [
            V1_NONMUTATION_STATEMENT,
            "Validation and benchmark reads use MD/X/Y/Z/TVT_input only.",
            "Training TVT may be opened only after the exact outer role is fixed.",
            "No field scale or amplitude coefficient is selected from labels.",
            "No incumbent model, typewell evidence, or ordered transport is recomputed.",
            "No external confirmation is present; production OPEN is impossible.",
        ],
    }
    _atomic_write_json(protocol_path, protocol)
    _write_sidecar(protocol_path)
    return protocol_path, _sha_sidecar(protocol_path)


def audit_protocol(
    protocol_path: Path,
    *,
    verify_data: bool = True,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    manifest_dir: Path | None = None,
) -> GateAudit:
    """Freshly audit v2, immutable v1 ancestry, incumbents, and live data."""

    protocol_path = protocol_path.resolve()
    expected = _read_sidecar(protocol_path)
    actual = sha256_file(protocol_path)
    if expected != actual:
        raise GateError("v2 structural-field protocol hash drift")
    protocol = _read_json(protocol_path, "v2 structural-field protocol")
    required = {
        "protocol_version",
        "status",
        "status_ceiling",
        "method",
        "created_at_utc",
        "roots",
        "source_sha256",
        "packages",
        "parent_v1_lineage",
        "fixed_configuration",
        "evaluation",
        "benchmark_work_proxy",
        "incumbent_pretruth_inventory",
        "notes",
    }
    if set(protocol) != required:
        raise GateError("v2 structural-field protocol schema drift")
    if (
        protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "FROZEN_BEFORE_V2_FIELD_SCORING_MEASURE_ONLY"
        or protocol.get("status_ceiling") != STATUS_CEILING
        or protocol.get("method") != METHOD
        or protocol.get("fixed_configuration") != fixed_configuration()
        or protocol.get("evaluation") != _evaluation_contract()
    ):
        raise GateError("v2 structural-field method/configuration drift")
    roots = protocol.get("roots")
    if not isinstance(roots, Mapping) or set(roots) != {
        "data_dir",
        "results_dir",
        "manifest_dir",
    }:
        raise GateError("v2 protocol root schema drift")
    live_data = (data_dir or Path(str(roots["data_dir"]))).resolve()
    live_results = (results_dir or Path(str(roots["results_dir"]))).resolve()
    live_manifests = (manifest_dir or Path(str(roots["manifest_dir"]))).resolve()
    if protocol.get("source_sha256") != _source_hashes():
        raise GateError("v2 bound source identity drift")
    if protocol.get("packages") != _packages():
        raise GateError("v2 package environment drift")
    parent = _parent_lineage(live_results)
    if protocol.get("parent_v1_lineage") != parent:
        raise GateError("v2 parent STOP lineage drift")
    parent_audit = legacy.audit_protocol(
        _safe_basename(live_results, V1_PROTOCOL_NAME),
        verify_data=verify_data,
        data_dir=live_data,
        results_dir=live_results,
        manifest_dir=live_manifests,
    )
    if parent_audit.protocol_sha256 != parent["v1_protocol"]["byte_sha256"]:
        raise GateError("live v1 parent audit drift")
    incumbent_inventory = protocol.get("incumbent_pretruth_inventory")
    if not isinstance(incumbent_inventory, Mapping):
        raise GateError("v2 protocol lacks incumbent pretruth inventory")
    legacy._validate_inventory_payload(
        incumbent_inventory, live_results, live_manifests
    )
    fresh_inventory = legacy._incumbent_inventory(live_results, live_manifests)
    if fresh_inventory != incumbent_inventory:
        raise GateError("fresh incumbent inventory differs from v2 frozen inventory")
    exact_manifest = pd.read_csv(_safe_basename(live_results, EXACT_MANIFEST_NAME))
    spatial_inventory = legacy.incumbent_spatial._validate_inventory(
        pd.read_csv(_safe_basename(live_results, SPATIAL_INVENTORY_NAME))
    )
    region_manifest = _read_json(
        _safe_basename(live_manifests, REGION_MANIFEST_NAME), "region manifest"
    )
    proxy = protocol.get("benchmark_work_proxy")
    if not isinstance(proxy, Mapping):
        raise GateError("v2 protocol lacks benchmark work proxy")
    _validate_work_proxy_payload(proxy)
    if verify_data:
        legacy._audit_data_files(spatial_inventory, live_data)
        live_proxy = _compute_work_proxy(live_data, live_results, live_manifests)
        if live_proxy != proxy:
            raise GateError("v2 inference-visible benchmark work proxy drift")
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


def _field_shard_name(mode: str, repeat: int, fold: int) -> str:
    if (mode, repeat, fold) not in set(_all_fold_identities()):
        raise GateError("invalid v2 field shard identity")
    return (
        f"exact_repeat_{repeat}_fold_{fold}.json"
        if mode == "exact"
        else f"region_fold_{fold}.json"
    )


def _field_prediction_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".npz")


def _validate_role_partition_schema(value: object) -> Mapping[str, Any]:
    partition = _require_exact_keys(
        value,
        {
            "partition_basis",
            "inner_folds",
            "fold_by_well_sha256",
            "leave_one_roles",
        },
        "v2 role partition",
    )
    if int(partition["inner_folds"]) != INNER_FOLDS or not _is_sha256(
        partition["fold_by_well_sha256"]
    ):
        raise GateError("v2 role partition identity drift")
    roles = partition["leave_one_roles"]
    if not isinstance(roles, list) or len(roles) != INNER_FOLDS:
        raise GateError("v2 leave-one role inventory drift")
    for fold, role in enumerate(roles):
        role = _require_exact_keys(
            role,
            {
                "excluded_fold",
                "fitting_well_count",
                "excluded_well_count",
                "fitting_ids_sha256",
                "excluded_ids_sha256",
            },
            f"v2 leave-one role {fold}",
        )
        if (
            int(role["excluded_fold"]) != fold
            or int(role["fitting_well_count"]) <= 0
            or int(role["excluded_well_count"]) <= 0
            or not _is_sha256(role["fitting_ids_sha256"])
            or not _is_sha256(role["excluded_ids_sha256"])
        ):
            raise GateError("v2 leave-one role metadata drift")
    return partition


def _validate_field_shard_schema(shard: Mapping[str, Any]) -> None:
    """Apply a recursive exact allowlist to a metric-silent v2 shard."""

    _reject_sensitive_field_names(shard)
    _require_exact_keys(
        shard,
        {
            "status",
            "status_ceiling",
            "protocol_version",
            "method",
            "protocol_sha256",
            "benchmark_file",
            "benchmark_sha256",
            "incumbent_inventory_sha256",
            "parent_v1_lineage_sha256",
            "mode",
            "repeat",
            "fold",
            "outer_role_sha256",
            "training_well_count",
            "validation_well_count",
            "embargo_well_count",
            "fixed_method_from_outer_training_only",
            "validation_diagnostics",
            "prediction_file",
            "prediction_sha256",
            "prediction_logical_sha256",
            "prediction_rows",
            "prediction_channels",
            "validation_wells",
            "runtime_seconds",
        },
        "v2 field shard",
    )
    _require_exact_keys(
        shard["outer_role_sha256"],
        {"training_ids", "validation_ids", "embargo_ids"},
        "v2 field shard outer roles",
    )
    learned = _require_exact_keys(
        shard["fixed_method_from_outer_training_only"],
        {
            "fixed_configuration",
            "field_fit_count",
            "validation_proposals_per_well",
            "role_partition",
            "leave_one_field_models",
            "final_role",
            "final_field_model",
            "support_query_truncation_count",
        },
        "v2 fixed method",
    )
    _validate_role_partition_schema(learned["role_partition"])
    final_role = _require_exact_keys(
        learned["final_role"],
        {"fitting_well_count", "fitting_ids_sha256"},
        "v2 final field role",
    )
    if int(final_role["fitting_well_count"]) <= 0 or not _is_sha256(
        final_role["fitting_ids_sha256"]
    ):
        raise GateError("v2 final field role drift")
    leave_models = learned["leave_one_field_models"]
    if not isinstance(leave_models, list) or len(leave_models) != INNER_FOLDS:
        raise GateError("v2 leave-one field model inventory drift")
    for fold, item in enumerate(leave_models):
        item = _require_exact_keys(
            item,
            {"excluded_fold", "model"},
            f"v2 leave-one field model {fold}",
        )
        if int(item["excluded_fold"]) != fold:
            raise GateError("v2 leave-one model role order drift")
        legacy._validate_model_metadata_schema(
            item["model"], f"v2 leave-one field model {fold}"
        )
    legacy._validate_model_metadata_schema(
        learned["final_field_model"], "v2 final field model"
    )
    diagnostics = shard["validation_diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise GateError("v2 validation diagnostic schema drift")
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
            diagnostic,
            diagnostic_keys,
            f"v2 validation diagnostic {well_id}",
        )
    rows = shard["validation_wells"]
    if not isinstance(rows, list):
        raise GateError("v2 validation-well schema drift")
    for index, row in enumerate(rows):
        _require_exact_keys(
            row,
            {"well", "well_index", "equality_group", "n_rows"},
            f"v2 validation well {index}",
        )


def _frozen_worst_proxy_row(audit: GateAudit) -> Mapping[str, Any]:
    proxy = audit.protocol.get("benchmark_work_proxy")
    if not isinstance(proxy, Mapping):
        raise GateError("v2 protocol lacks a benchmark work proxy")
    _validate_work_proxy_payload(proxy)
    identity = proxy["maximizing_identity"]
    matches = [
        row
        for row in proxy["folds"]
        if (str(row["mode"]), int(row["repeat"]), int(row["fold"]))
        == (
            str(identity["mode"]),
            int(identity["repeat"]),
            int(identity["fold"]),
        )
    ]
    if len(matches) != 1:
        raise GateError("v2 frozen worst-work identity is ambiguous")
    return matches[0]


def _valid_solver_codes(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    for code in value:
        if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
            return False
        if int(code) not in {0, 1, 2, 4, 5}:
            return False
    return True


def _benchmark_acceptance(
    timing: Mapping[str, Any],
    memory: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> dict[str, bool]:
    thresholds = _evaluation_contract()["runtime_acceptance"]
    observations = shape.get("model_observations")
    healthy_models = isinstance(observations, list) and len(observations) == 5
    if healthy_models:
        for observation in observations:
            if not isinstance(observation, Mapping):
                healthy_models = False
                break
            codes = observation.get("solver_stop_codes")
            try:
                requested = float(observation.get("requested_inducing_cell_ft"))
                actual = float(observation.get("actual_inducing_cell_ft"))
            except (TypeError, ValueError):
                healthy_models = False
                break
            if (
                not _valid_solver_codes(codes)
                or not math.isclose(
                    requested,
                    FIXED_INDUCING_CELL_FT,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    actual,
                    requested,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                healthy_models = False
                break
    return {
        "field_wall": float(timing["field"])
        <= thresholds["field_wall_seconds_at_most"],
        "field_peak_rss": float(memory["field_peak_rss"])
        <= thresholds["field_peak_rss_gib_at_most"],
        "total_wall": float(timing["total"])
        <= thresholds["total_wall_seconds_at_most"],
        "total_peak_rss": float(memory["total_peak_rss"])
        <= thresholds["total_peak_rss_gib_at_most"],
        "two_worker_fifteen_fold": float(timing["extrapolated_two_worker_fifteen_fold"])
        <= thresholds["extrapolated_two_worker_fifteen_fold_seconds_at_most"],
        "no_caps_solver_or_coarsening": not bool(shape["solver_caps_changed"])
        and not bool(shape["coarsening_allowed"])
        and healthy_models,
        "no_support_query_truncation": int(shape["support_query_truncation_count"])
        == 0,
    }


def _validate_benchmark_payload(
    benchmark: Mapping[str, Any],
    audit: GateAudit,
    *,
    require_acceptance: bool = True,
) -> None:
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
            "fixed_configuration",
            "timing_seconds",
            "memory_gib",
            "acceptance",
            "all_acceptance_pass",
            "validation_metrics",
        },
        "v2 benchmark",
    )
    if (
        benchmark.get("status") != BENCHMARK_STATUS
        or benchmark.get("status_ceiling") != STATUS_CEILING
        or benchmark.get("method") != METHOD
        or benchmark.get("protocol_sha256") != audit.protocol_sha256
        or benchmark.get("fixed_configuration") != fixed_configuration()
        or benchmark.get("validation_metrics")
        != "withheld; validation TVT was not parsed"
    ):
        raise GateError("v2 benchmark protocol/configuration identity drift")
    proxy = audit.protocol.get("benchmark_work_proxy")
    if not isinstance(proxy, Mapping):
        raise GateError("v2 benchmark protocol lacks work proxy")
    _validate_work_proxy_payload(proxy)
    if benchmark.get("benchmark_work_proxy_sha256") != proxy["proxy_sha256"]:
        raise GateError("v2 benchmark work-proxy digest drift")
    row = _frozen_worst_proxy_row(audit)
    expected_identity = {
        "mode": str(row["mode"]),
        "repeat": int(row["repeat"]),
        "fold": int(row["fold"]),
        "training_wells": int(row["training_wells"]),
        "validation_wells": int(row["validation_wells"]),
        "embargo_wells": int(row["embargo_wells"]),
        "proxy_units": int(row["proxy_units"]),
    }
    identity = _require_exact_keys(
        benchmark["worst_work_identity"],
        set(expected_identity),
        "v2 benchmark identity",
    )
    if dict(identity) != expected_identity:
        raise GateError("v2 benchmark did not measure the frozen worst-work identity")
    shape = _require_exact_keys(
        benchmark["work_shape"],
        {
            "lineage_audits",
            "leave_one_field_fits",
            "final_field_fits",
            "field_fits_total",
            "validation_proposals_per_well",
            "prediction_rows",
            "solver_caps_changed",
            "coarsening_allowed",
            "support_query_truncation_count",
            "model_observations",
        },
        "v2 benchmark work shape",
    )
    if (
        int(shape["lineage_audits"]) != 2
        or int(shape["leave_one_field_fits"]) != INNER_FOLDS
        or int(shape["final_field_fits"]) != 1
        or int(shape["field_fits_total"]) != FIELD_FITS_PER_OUTER
        or int(shape["validation_proposals_per_well"]) != VALIDATION_PROPOSALS_PER_WELL
        or int(shape["prediction_rows"]) != int(row["validation_suffix_rows"])
        or shape["solver_caps_changed"] is not False
        or shape["coarsening_allowed"] is not False
        or int(shape["support_query_truncation_count"]) != 0
    ):
        raise GateError("v2 benchmark work shape/cap contract drift")
    observations = shape["model_observations"]
    if not isinstance(observations, list) or len(observations) != 5:
        raise GateError("v2 benchmark model-observation inventory drift")
    expected_roles = [f"leave_one_{fold}" for fold in range(4)] + ["final"]
    expected_counts = [
        *map(int, row["leave_one_training_wells"]),
        int(row["training_wells"]),
    ]
    expected_digests = [
        *map(str, row["leave_one_role_sha256"]),
        str(row["final_role_sha256"]),
    ]
    stop_codes: list[int] = []
    for index, observation in enumerate(observations):
        observation = _require_exact_keys(
            observation,
            {
                "role",
                "fitting_well_count",
                "fitting_ids_sha256",
                "requested_inducing_cell_ft",
                "actual_inducing_cell_ft",
                "solver_stop_codes",
            },
            f"v2 benchmark model observation {index}",
        )
        codes = observation["solver_stop_codes"]
        if (
            observation["role"] != expected_roles[index]
            or int(observation["fitting_well_count"]) != expected_counts[index]
            or observation["fitting_ids_sha256"] != expected_digests[index]
            or not _valid_solver_codes(codes)
            or not math.isclose(
                float(observation["requested_inducing_cell_ft"]),
                FIXED_INDUCING_CELL_FT,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not math.isclose(
                float(observation["actual_inducing_cell_ft"]),
                FIXED_INDUCING_CELL_FT,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise GateError("v2 benchmark observed model role/solver/coarsening drift")
        stop_codes.extend(map(int, codes))
    if len(stop_codes) != 10:
        raise GateError("v2 benchmark must bind exactly ten solver stop codes")
    timing = _require_exact_keys(
        benchmark["timing_seconds"],
        {
            "initial_live_audit",
            "field",
            "final_live_audit",
            "total",
            "extrapolated_two_worker_fifteen_fold",
        },
        "v2 benchmark timing",
    )
    values = {
        key: _finite_nonnegative(value, f"v2 benchmark time {key}")
        for key, value in timing.items()
    }
    if values["initial_live_audit"] <= 0.0 or values["final_live_audit"] <= 0.0:
        raise GateError("v2 benchmark omitted one of two live audits")
    component_sum = (
        values["initial_live_audit"] + values["field"] + values["final_live_audit"]
    )
    if component_sum > values["total"] + 1.0e-6:
        raise GateError("v2 benchmark components exceed total timing")
    expected_projection = values["total"] * math.ceil(15 / 2)
    if not math.isclose(
        values["extrapolated_two_worker_fifteen_fold"],
        expected_projection,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise GateError("v2 benchmark projection arithmetic drift")
    memory = _require_exact_keys(
        benchmark["memory_gib"],
        {"field_peak_rss", "total_peak_rss"},
        "v2 benchmark memory",
    )
    memory_values = {
        key: _finite_nonnegative(value, f"v2 benchmark memory {key}")
        for key, value in memory.items()
    }
    if memory_values["field_peak_rss"] > memory_values["total_peak_rss"] + 1.0e-12:
        raise GateError("v2 field RSS exceeds total RSS")
    expected_acceptance = _benchmark_acceptance(values, memory_values, shape)
    acceptance = _require_exact_keys(
        benchmark["acceptance"], set(expected_acceptance), "v2 benchmark acceptance"
    )
    if any(not isinstance(value, bool) for value in acceptance.values()):
        raise GateError("v2 benchmark acceptance values are not booleans")
    if dict(acceptance) != expected_acceptance or benchmark.get(
        "all_acceptance_pass"
    ) is not all(expected_acceptance.values()):
        raise GateError("v2 benchmark acceptance was not recomputed")
    if require_acceptance and not all(expected_acceptance.values()):
        raise GateHold("v2 benchmark acceptance failed; run is STOP")


def _validate_benchmark(path: Path, audit: GateAudit) -> str:
    path = path.resolve()
    if _read_sidecar(path).lower() != sha256_file(path):
        raise GateError("v2 benchmark sidecar drift")
    benchmark = _read_json(path, "v2 field benchmark")
    _validate_benchmark_payload(benchmark, audit)
    return sha256_file(path)


def _assert_output_directory_separate(output_dir: Path, audit: GateAudit) -> None:
    resolved = output_dir.resolve()
    forbidden_exact = {
        audit.results_dir.resolve(),
        audit.manifest_dir.resolve(),
    }
    protected_trees = {
        (audit.results_dir / EXACT_SHARD_DIR_NAME).resolve(),
        (audit.results_dir / SPATIAL_SHARD_DIR_NAME).resolve(),
        (audit.results_dir / "anchored_structural_field_protocol_folds").resolve(),
    }
    if resolved in forbidden_exact or any(
        resolved == protected
        or protected in resolved.parents
        or resolved in protected.parents
        for protected in protected_trees
    ):
        raise GateError("v2 output directory aliases a frozen v1/incumbent directory")


def _assert_existing_not_hardlinked(path: Path, reserved: Iterable[Path]) -> None:
    if not path.exists():
        return
    for frozen in reserved:
        if frozen.exists() and os.path.samefile(path, frozen):
            raise GateError("v2 output is hard-linked to a frozen parent artifact")


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
    shard_path = shard_path.resolve()
    _assert_output_directory_separate(shard_path.parent, audit)
    reserved = _reserved_frozen_paths(
        audit.results_dir,
        audit.manifest_dir,
        audit.protocol["incumbent_pretruth_inventory"],
    ) | {
        audit.protocol_path.resolve(),
        _sha_sidecar(audit.protocol_path).resolve(),
        benchmark_path.resolve(),
        _sha_sidecar(benchmark_path).resolve(),
    }
    _assert_existing_not_hardlinked(shard_path, reserved)
    if (
        shard.get("status") != SHARD_STATUS
        or shard.get("status_ceiling") != STATUS_CEILING
        or shard.get("protocol_version") != PROTOCOL_VERSION
        or shard.get("method") != METHOD
    ):
        raise GateError("v2 field shard status/method drift")
    if _read_sidecar(shard_path).lower() != sha256_file(shard_path):
        raise GateError("v2 field shard metadata hash drift")
    if shard.get("protocol_sha256") != audit.protocol_sha256:
        raise GateError("v2 field shard protocol identity drift")
    benchmark_sha256 = _validate_benchmark(benchmark_path, audit)
    if (
        shard.get("benchmark_file") != benchmark_path.name
        or shard.get("benchmark_sha256") != benchmark_sha256
    ):
        raise GateError("v2 field shard benchmark lineage drift")
    incumbent_inventory = audit.protocol["incumbent_pretruth_inventory"]
    parent = audit.protocol["parent_v1_lineage"]
    if (
        shard.get("incumbent_inventory_sha256")
        != incumbent_inventory["inventory_sha256"]
        or shard.get("parent_v1_lineage_sha256") != parent["lineage_sha256"]
    ):
        raise GateError("v2 field shard parent/incumbent lineage drift")
    if (
        shard.get("mode") != mode
        or int(shard.get("repeat", -1)) != repeat
        or int(shard.get("fold", -1)) != fold
    ):
        raise GateError("v2 field shard identity drift")
    training_ids, validation_ids, embargo_ids, group_by_well = _outer_roles(
        audit, mode, repeat, fold
    )
    role_digests = {
        "training_ids": _id_digest(training_ids),
        "validation_ids": _id_digest(validation_ids),
        "embargo_ids": _id_digest(embargo_ids),
    }
    if shard.get("outer_role_sha256") != role_digests:
        raise GateError("v2 field shard outer-role digest drift")
    if (
        int(shard.get("training_well_count", -1)) != len(training_ids)
        or int(shard.get("validation_well_count", -1)) != len(validation_ids)
        or int(shard.get("embargo_well_count", -1)) != len(embargo_ids)
    ):
        raise GateError("v2 field shard outer-role count drift")
    _finite_nonnegative(shard.get("runtime_seconds"), "v2 field shard runtime")
    learned = shard["fixed_method_from_outer_training_only"]
    if (
        learned.get("fixed_configuration") != fixed_configuration()
        or int(learned.get("field_fit_count", -1)) != FIELD_FITS_PER_OUTER
        or int(learned.get("validation_proposals_per_well", -1))
        != VALIDATION_PROPOSALS_PER_WELL
        or int(learned.get("support_query_truncation_count", -1)) != 0
    ):
        raise GateError("v2 fixed-method identity/work shape drift")

    # Recompute the exact geometry-only partition without opening training TVT.
    training_suffixes = _load_incumbent_suffixes(audit, mode, repeat, training_ids)
    geometry_wells = _load_role_wells(
        audit, training_ids, training_suffixes, "benchmark"
    )
    partition = _inner_fold_roles(
        geometry_wells,
        {well_id: group_by_well[well_id] for well_id in training_ids},
    )
    if learned.get("role_partition") != partition.metadata:
        raise GateError("v2 field shard geometry-only role partition drift")
    roles = partition.metadata["leave_one_roles"]
    leave_models = learned["leave_one_field_models"]
    for fold_index, (role, model_item) in enumerate(
        zip(roles, leave_models, strict=True)
    ):
        metadata = model_item["model"]
        if (
            int(model_item["excluded_fold"]) != fold_index
            or int(metadata["training_wells"]) != int(role["fitting_well_count"])
            or not math.isclose(
                float(metadata["requested_inducing_cell_ft"]),
                FIXED_INDUCING_CELL_FT,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise GateError("v2 leave-one model membership/configuration drift")
    final_role = learned["final_role"]
    final_model = learned["final_field_model"]
    if (
        final_role
        != {
            "fitting_well_count": len(training_ids),
            "fitting_ids_sha256": _id_digest(sorted(training_ids)),
        }
        or int(final_model["training_wells"]) != len(training_ids)
        or not math.isclose(
            float(final_model["requested_inducing_cell_ft"]),
            FIXED_INDUCING_CELL_FT,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise GateError("v2 final model membership/configuration drift")
    if shard.get("prediction_channels") != list(PREDICTION_ARRAYS[2:]):
        raise GateError("v2 prediction channel schema drift")
    diagnostics = shard["validation_diagnostics"]
    if list(map(str, diagnostics)) != validation_ids:
        raise GateError("v2 validation diagnostic order/membership drift")
    for well_id, diagnostic in diagnostics.items():
        for key, value in diagnostic.items():
            if key != "status" and not math.isfinite(float(value)):
                raise GateError(f"non-finite v2 diagnostic: {well_id}:{key}")
        if diagnostic["status"] != "anchored_field_100ft_knots_with_policy_fallback":
            raise GateError("v2 prediction diagnostic status drift")
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
                raise GateError(f"v2 diagnostic count drift: {well_id}:{key}")
        for key in (
            "fallback_fraction",
            "mean_core_confidence",
            "mean_jackknife_confidence",
            "mean_final_confidence",
            "supported_fraction",
            "query_direction_observability_mean",
        ):
            if not 0.0 <= float(diagnostic[key]) <= 1.0 + 1.0e-12:
                raise GateError(f"v2 diagnostic range drift: {well_id}:{key}")
        for key in (
            "mean_training_midpoint_distance_ft",
            "max_training_midpoint_distance_ft",
            "effective_well_support_mean",
        ):
            if float(diagnostic[key]) < 0.0:
                raise GateError(f"v2 diagnostic sign drift: {well_id}:{key}")

    prediction_path = _safe_basename(
        shard_path.parent, str(shard.get("prediction_file", ""))
    )
    if prediction_path != _field_prediction_path(shard_path).resolve():
        raise GateError("v2 prediction basename drift")
    _assert_existing_not_hardlinked(prediction_path, reserved)
    if sha256_file(prediction_path) != shard.get("prediction_sha256"):
        raise GateError("v2 prediction byte hash drift")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_ARRAYS):
            raise GateError("v2 prediction NPZ schema drift")
        arrays = {name: archive[name].copy() for name in PREDICTION_ARRAYS}
    if len(arrays["row_index"]) != int(shard.get("prediction_rows", -1)):
        raise GateError("v2 prediction row count drift")
    if _logical_array_hash(arrays) != shard.get("prediction_logical_sha256"):
        raise GateError("v2 prediction logical hash drift")
    validation_suffixes = _load_incumbent_suffixes(audit, mode, repeat, validation_ids)
    _validate_prediction_arrays(arrays, validation_ids, validation_suffixes)
    rows = shard["validation_wells"]
    wells = [str(row.get("well", "")) for row in rows]
    if wells != validation_ids:
        raise GateError("v2 validation well ordering drift")
    inventory = _inventory_by_well(audit)
    for well_index, row in enumerate(rows):
        well_id = wells[well_index]
        selected = arrays["well_index"] == well_index
        expected_count = int(inventory.loc[well_id, "suffix_rows"])
        if (
            int(row.get("well_index", -1)) != well_index
            or str(row.get("equality_group", "")) != group_by_well[well_id]
            or int(row.get("n_rows", -1)) != expected_count
            or int(selected.sum()) != expected_count
        ):
            raise GateError("v2 validation well metadata drift")
        prefix = int(inventory.loc[well_id, "prefix_rows"])
        total = int(inventory.loc[well_id, "rows"])
        diagnostic = diagnostics[well_id]
        if (
            int(diagnostic["evaluation_rows"]) <= 0
            or int(diagnostic["prefix_rows"]) != prefix
            or int(diagnostic["suffix_rows"]) != expected_count
            or not np.array_equal(
                arrays["row_index"][selected], np.arange(prefix, total)
            )
        ):
            raise GateError("v2 validation row/diagnostic alignment drift")
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
    _assert_output_directory_separate(output_dir, audit)
    shard_path = (output_dir / _field_shard_name(mode, repeat, fold)).resolve()
    prediction_path = _field_prediction_path(shard_path)
    if shard_path.exists():
        if not resume:
            raise GateError(f"v2 field shard already exists: {shard_path}")
        shard = _read_json(shard_path, "v2 field shard")
        _validate_field_shard(
            shard, audit, mode, repeat, fold, shard_path, benchmark_path
        )
        return shard_path
    if prediction_path.exists() or _sha_sidecar(shard_path).exists():
        raise GateError(f"incomplete/stale v2 field artifacts: {shard_path}")
    reserved = _reserved_frozen_paths(
        audit.results_dir,
        audit.manifest_dir,
        audit.protocol["incumbent_pretruth_inventory"],
    ) | {
        audit.protocol_path.resolve(),
        _sha_sidecar(audit.protocol_path).resolve(),
        benchmark_path.resolve(),
        _sha_sidecar(benchmark_path).resolve(),
    }
    _assert_write_target(shard_path, reserved)
    _assert_write_target(prediction_path, reserved)
    started = time.perf_counter()
    training_ids, validation_ids, embargo_ids, group_by_well = _outer_roles(
        audit, mode, repeat, fold
    )
    active_ids = [*training_ids, *validation_ids]
    if set(active_ids) & set(embargo_ids):
        raise GateError("v2 embargo entered an active outer role")
    suffixes = _load_incumbent_suffixes(audit, mode, repeat, active_ids)
    training_wells = _load_role_wells(
        audit,
        training_ids,
        {well_id: suffixes[well_id] for well_id in training_ids},
        "training",
    )
    partition = _inner_fold_roles(
        training_wells,
        {well_id: group_by_well[well_id] for well_id in training_ids},
    )
    fitted = _fit_fixed_models(training_wells, partition)
    validation_wells = _load_role_wells(
        audit,
        validation_ids,
        {well_id: suffixes[well_id] for well_id in validation_ids},
        "validation",
    )
    array_blocks: dict[str, list[NDArray[Any]]] = {
        name: [] for name in PREDICTION_ARRAYS
    }
    rows = []
    diagnostics: dict[str, Any] = {}
    for well_index, well_id in enumerate(validation_ids):
        arrays, diagnostic = _predict_candidate(
            fitted.final_model, fitted.leave_one_models, validation_wells[well_id]
        )
        count = len(arrays["row_index"])
        array_blocks["well_index"].append(np.full(count, well_index, dtype=np.int32))
        for name in PREDICTION_ARRAYS[1:]:
            array_blocks[name].append(arrays[name])
        rows.append(
            {
                "well": well_id,
                "well_index": well_index,
                "equality_group": group_by_well[well_id],
                "n_rows": count,
            }
        )
        diagnostics[well_id] = diagnostic
    if not validation_ids or any(not blocks for blocks in array_blocks.values()):
        raise GateError("v2 validation prediction role is empty")
    sealed_arrays = {
        name: np.concatenate(blocks) for name, blocks in array_blocks.items()
    }
    _validate_prediction_arrays(
        sealed_arrays,
        validation_ids,
        {well_id: suffixes[well_id] for well_id in validation_ids},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npz(prediction_path, sealed_arrays)
    leave_models = [
        {
            "excluded_fold": fold_index,
            "model": _model_metadata(model),
        }
        for fold_index, model in enumerate(fitted.leave_one_models)
    ]
    learned = {
        "fixed_configuration": fixed_configuration(),
        "field_fit_count": FIELD_FITS_PER_OUTER,
        "validation_proposals_per_well": VALIDATION_PROPOSALS_PER_WELL,
        "role_partition": partition.metadata,
        "leave_one_field_models": leave_models,
        "final_role": fitted.metadata["final_role"],
        "final_field_model": _model_metadata(fitted.final_model),
        "support_query_truncation_count": 0,
    }
    shard = {
        "status": SHARD_STATUS,
        "status_ceiling": STATUS_CEILING,
        "protocol_version": PROTOCOL_VERSION,
        "method": METHOD,
        "protocol_sha256": audit.protocol_sha256,
        "benchmark_file": benchmark_path.name,
        "benchmark_sha256": benchmark_sha256,
        "incumbent_inventory_sha256": audit.protocol["incumbent_pretruth_inventory"][
            "inventory_sha256"
        ],
        "parent_v1_lineage_sha256": audit.protocol["parent_v1_lineage"][
            "lineage_sha256"
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
        "fixed_method_from_outer_training_only": learned,
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


def run_folds(
    protocol_path: Path,
    benchmark_path: Path,
    output_dir: Path | None,
    folds: Sequence[tuple[str, int, int]],
    resume: bool,
) -> list[Path]:
    """Write v2 metric-silent shards; validation suffix TVT remains unread."""

    audit = audit_protocol(protocol_path, verify_data=True)
    benchmark_path = benchmark_path.resolve()
    benchmark_sha256 = _validate_benchmark(benchmark_path, audit)
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    _assert_output_directory_separate(output_dir, audit)
    valid = set(_all_fold_identities())
    normalized: list[tuple[str, int, int]] = []
    for identity in folds:
        if identity not in valid:
            raise GateError(f"invalid v2 field fold identity: {identity}")
        if identity not in normalized:
            normalized.append(identity)
    completed = []
    for mode, repeat, fold in normalized:
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
        current = audit_protocol(protocol_path, verify_data=True)
        if current.protocol_sha256 != audit.protocol_sha256:
            raise GateError("v2 protocol/source/data drifted after fold commit")
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
    row = _frozen_worst_proxy_row(audit)
    return str(row["mode"]), int(row["repeat"]), int(row["fold"])


def benchmark_gate(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    """Measure the frozen worst v2 fold with validation TVT quarantined."""

    protocol_path = protocol_path.resolve()
    output_path = output_path.resolve()
    default_shard_dir = protocol_path.with_name(protocol_path.stem + "_folds")
    _assert_outside_tree(output_path, default_shard_dir, "v2 benchmark output")
    total_start = time.perf_counter()
    with _RssSampler() as total_memory:
        initial_start = time.perf_counter()
        audit = audit_protocol(protocol_path, verify_data=True)
        initial_audit_seconds = time.perf_counter() - initial_start
        print(
            json.dumps(
                {
                    "v2_benchmark_phase": "initial_live_audit_complete",
                    "seconds": initial_audit_seconds,
                    "validation_metrics": "withheld",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        reserved = _reserved_frozen_paths(
            audit.results_dir,
            audit.manifest_dir,
            audit.protocol["incumbent_pretruth_inventory"],
        ) | {audit.protocol_path.resolve(), _sha_sidecar(audit.protocol_path).resolve()}
        _assert_artifact_outside_frozen_trees(output_path, audit.results_dir)
        _assert_write_target(output_path, reserved)
        mode, repeat, fold = _largest_outer_identity(audit)
        training_ids, validation_ids, embargo_ids, group_by_well = _outer_roles(
            audit, mode, repeat, fold
        )
        active_ids = [*training_ids, *validation_ids]
        if set(active_ids) & set(embargo_ids):
            raise GateError("v2 benchmark embargo entered an active role")
        field_start = time.perf_counter()
        with _RssSampler() as field_memory:
            suffixes = _load_incumbent_suffixes(audit, mode, repeat, active_ids)
            training_wells = _load_role_wells(
                audit,
                training_ids,
                {well_id: suffixes[well_id] for well_id in training_ids},
                "training",
            )
            partition = _inner_fold_roles(
                training_wells,
                {well_id: group_by_well[well_id] for well_id in training_ids},
            )
            print(
                json.dumps(
                    {
                        "v2_benchmark_phase": "outer_roles_loaded",
                        "training_wells": len(training_ids),
                        "validation_wells": len(validation_ids),
                        "embargo_wells": len(embargo_ids),
                        "validation_metrics": "withheld",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            fitted = _fit_fixed_models(training_wells, partition)
            print(
                json.dumps(
                    {
                        "v2_benchmark_phase": "five_field_fits_complete",
                        "field_fits": FIELD_FITS_PER_OUTER,
                        "validation_metrics": "withheld",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            validation_wells = _load_role_wells(
                audit,
                validation_ids,
                {well_id: suffixes[well_id] for well_id in validation_ids},
                "benchmark",
            )
            prediction_rows = 0
            for well_id in validation_ids:
                arrays, _ = _predict_candidate(
                    fitted.final_model,
                    fitted.leave_one_models,
                    validation_wells[well_id],
                )
                prediction_rows += len(arrays["row_index"])
        field_seconds = time.perf_counter() - field_start
        print(
            json.dumps(
                {
                    "v2_benchmark_phase": "validation_proposals_complete",
                    "seconds": field_seconds,
                    "prediction_rows": prediction_rows,
                    "validation_metrics": "withheld",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        final_start = time.perf_counter()
        final_audit = audit_protocol(protocol_path, verify_data=True)
        final_audit_seconds = time.perf_counter() - final_start
        print(
            json.dumps(
                {
                    "v2_benchmark_phase": "final_live_audit_complete",
                    "seconds": final_audit_seconds,
                    "validation_metrics": "withheld",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            final_audit.protocol_sha256 != audit.protocol_sha256
            or final_audit.protocol != audit.protocol
        ):
            raise GateError("v2 protocol/source/data drifted before benchmark commit")
    total_seconds = time.perf_counter() - total_start
    projection = total_seconds * math.ceil(15 / 2)
    gib = float(1024**3)
    timing = {
        "initial_live_audit": initial_audit_seconds,
        "field": field_seconds,
        "final_live_audit": final_audit_seconds,
        "total": total_seconds,
        "extrapolated_two_worker_fifteen_fold": projection,
    }
    memory = {
        "field_peak_rss": field_memory.peak_bytes / gib,
        "total_peak_rss": total_memory.peak_bytes / gib,
    }
    role_rows = partition.metadata["leave_one_roles"]
    model_observations = []
    for inner_fold, (model, role) in enumerate(
        zip(fitted.leave_one_models, role_rows, strict=True)
    ):
        metadata = _model_metadata(model)
        model_observations.append(
            {
                "role": f"leave_one_{inner_fold}",
                "fitting_well_count": int(role["fitting_well_count"]),
                "fitting_ids_sha256": str(role["fitting_ids_sha256"]),
                "requested_inducing_cell_ft": float(
                    metadata["requested_inducing_cell_ft"]
                ),
                "actual_inducing_cell_ft": float(metadata["actual_inducing_cell_ft"]),
                "solver_stop_codes": list(metadata["solver_stop_codes"]),
            }
        )
    final_metadata = _model_metadata(fitted.final_model)
    model_observations.append(
        {
            "role": "final",
            "fitting_well_count": len(training_ids),
            "fitting_ids_sha256": _id_digest(sorted(training_ids)),
            "requested_inducing_cell_ft": float(
                final_metadata["requested_inducing_cell_ft"]
            ),
            "actual_inducing_cell_ft": float(final_metadata["actual_inducing_cell_ft"]),
            "solver_stop_codes": list(final_metadata["solver_stop_codes"]),
        }
    )
    shape = {
        "lineage_audits": 2,
        "leave_one_field_fits": INNER_FOLDS,
        "final_field_fits": 1,
        "field_fits_total": FIELD_FITS_PER_OUTER,
        "validation_proposals_per_well": VALIDATION_PROPOSALS_PER_WELL,
        "prediction_rows": prediction_rows,
        "solver_caps_changed": False,
        "coarsening_allowed": False,
        "support_query_truncation_count": 0,
        "model_observations": model_observations,
    }
    acceptance = _benchmark_acceptance(timing, memory, shape)
    proxy_row = _frozen_worst_proxy_row(audit)
    result = {
        "status": BENCHMARK_STATUS,
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
            "proxy_units": int(proxy_row["proxy_units"]),
        },
        "work_shape": shape,
        "fixed_configuration": fixed_configuration(),
        "timing_seconds": timing,
        "memory_gib": memory,
        "acceptance": acceptance,
        "all_acceptance_pass": all(acceptance.values()),
        "validation_metrics": "withheld; validation TVT was not parsed",
    }
    _validate_benchmark_payload(result, audit, require_acceptance=False)
    _atomic_write_json(output_path, result)
    _write_sidecar(output_path)
    # This validates the sealed artifact and raises GateHold after commit when a
    # measured resource threshold fails, preserving the STOP evidence.
    _validate_benchmark(output_path, audit)
    return result


def _validate_exact_shard_directory(shard_dir: Path) -> None:
    directory = shard_dir.resolve()
    if not directory.is_dir():
        raise GateError(f"v2 shard directory is missing: {directory}")
    json_names = {
        _field_shard_name(mode, repeat, fold)
        for mode, repeat, fold in _all_fold_identities()
    }
    npz_names = {Path(name).with_suffix(".npz").name for name in json_names}
    sidecar_names = {name + ".sha256" for name in json_names}
    expected = json_names | npz_names | sidecar_names
    entries = list(directory.iterdir())
    if {path.name for path in entries} != expected or any(
        not path.is_file() for path in entries
    ):
        raise GateError(
            "v2 shard directory is partial, padded, duplicated, or contains extras"
        )
    resolved = [
        str((directory / name).resolve()).casefold() for name in sorted(expected)
    ]
    if len(resolved) != len(set(resolved)):
        raise GateError("v2 shard artifact paths alias after resolution")
    file_ids = []
    for name in sorted(expected):
        stat = (directory / name).stat()
        file_ids.append((int(stat.st_dev), int(stat.st_ino)))
    if all(inode != 0 for _, inode in file_ids) and len(file_ids) != len(set(file_ids)):
        raise GateError("v2 shard artifacts alias hard-linked bytes")


def build_pretruth_field_inventory(
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path | None,
) -> dict[str, Any]:
    """Freshly audit and inventory exactly all 15 v2 shards before truth."""

    audit = audit_protocol(protocol_path, verify_data=True)
    benchmark_path = benchmark_path.resolve()
    benchmark_sha256 = _validate_benchmark(benchmark_path, audit)
    shard_dir = (
        shard_dir.resolve()
        if shard_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    _assert_output_directory_separate(shard_dir, audit)
    _validate_exact_shard_directory(shard_dir)
    items = []
    for mode, repeat, fold in _all_fold_identities():
        shard_path = shard_dir / _field_shard_name(mode, repeat, fold)
        shard = _read_json(shard_path, "v2 field shard")
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
    """Seal the exact 15-shard v2 inventory; scorer performs a fresh audit."""

    protocol_path = protocol_path.resolve()
    benchmark_path = benchmark_path.resolve()
    resolved_shard_dir = (
        shard_dir.resolve()
        if shard_dir is not None
        else protocol_path.with_name(protocol_path.stem + "_folds")
    )
    inventory_output = inventory_output.resolve()
    _assert_outside_tree(inventory_output, resolved_shard_dir, "v2 pretruth barrier")
    audit = audit_protocol(protocol_path, verify_data=True)
    reserved = _reserved_frozen_paths(
        audit.results_dir,
        audit.manifest_dir,
        audit.protocol["incumbent_pretruth_inventory"],
    ) | {
        protocol_path,
        _sha_sidecar(protocol_path),
        benchmark_path,
        _sha_sidecar(benchmark_path),
    }
    _assert_artifact_outside_frozen_trees(inventory_output, audit.results_dir)
    _assert_write_target(inventory_output, reserved)
    inventory = build_pretruth_field_inventory(
        protocol_path, benchmark_path, resolved_shard_dir
    )
    sealed = {
        "status": BARRIER_STATUS,
        "status_ceiling": STATUS_CEILING,
        "pretruth_field_inventory": inventory,
        "next_phase": (
            "HOLD: source-bound v2 scorer must freshly re-audit protocol, parent "
            "STOP lineage, benchmark, exact shard directory, and this inventory "
            "before opening validation TVT"
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
        raise argparse.ArgumentTypeError(f"fold is outside the v2 set: {value}")
    return identity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--data-dir", type=Path, required=True)
    freeze.add_argument("--results-dir", type=Path, required=True)
    freeze.add_argument("--manifest-dir", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path)
    audit.add_argument("--results-dir", type=Path)
    audit.add_argument("--manifest-dir", type=Path)
    audit.add_argument("--no-data", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--fold", type=_parse_fold, action="append")
    run.add_argument("--resume", action="store_true")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--protocol", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
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
        print(f"froze MEASURE_ONLY v2 protocol {protocol}")
        print(f"wrote v2 protocol identity {sidecar}")
    elif args.command == "audit":
        bundle = audit_protocol(
            args.protocol,
            verify_data=not args.no_data,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            manifest_dir=args.manifest_dir,
        )
        print(f"v2 field audit passed: {bundle.protocol_sha256}")
    elif args.command == "run":
        run_folds(
            args.protocol,
            args.benchmark,
            args.output_dir,
            args.fold or _all_fold_identities(),
            args.resume,
        )
    elif args.command == "benchmark":
        print(json.dumps(benchmark_gate(args.protocol, args.output), indent=2))
    elif args.command == "aggregate":
        print(
            json.dumps(
                aggregate_barrier(
                    args.protocol,
                    args.benchmark,
                    args.shard_dir,
                    args.inventory_output,
                ),
                indent=2,
            )
        )
    else:  # pragma: no cover
        raise GateError(f"unsupported v2 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
