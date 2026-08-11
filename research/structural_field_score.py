"""Sealed post-prediction scorecard for the structural-field gate.

This module consumes, but never creates or selects among, the fifteen frozen
prediction shards emitted by :mod:`research.structural_field_gate`.  The gate
protocol, accepted no-truth benchmark, incumbent lineage, manifests, source
bytes, data bytes, every shard, and the separately sealed aggregate barrier
must all validate before the first validation ``TVT`` value can be read.

The exact-profile tail statistic is deliberately distributional: it is
``q0.90(candidate well RMSE) - q0.90(joint well RMSE)`` over both repeat
records.  A bootstrap draw applies one exact-profile-group multiplicity vector
to both repeats and computes weighted q0.90 values over that coupled repeated-
record population.  The reported conservative bound is the 95th percentile
of those differences.  It is not q0.90 of paired RMSE differences.

The status ceiling is ``MEASURE_ONLY``.  Results can only be ``MEASURE_ONLY``
or ``STOP`` and can never promote production ``OPEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from research import structural_field_gate as gate


SCORE_VERSION = "geosteern-anchored-structural-field-score/1"
STATUS_CEILING = "MEASURE_ONLY_OR_STOP_NEVER_OPEN"
EXACT_MODE = "exact"
REGION_MODE = "region"
EXPECTED_SHARDS = 15
BOOTSTRAP_DRAWS = 4_000
BOOTSTRAP_SEED = 20260811
TOP_WELLS_TO_REMOVE = 10

EXACT_MEAN_GAIN_FT = 1.0
EXACT_MEDIAN_GAIN_FT = 0.5
REGION_GAIN_FT = 0.75
REGION_FOLD_GAIN_FT = 0.20
P90_POINT_MAX_WORSENING_FT = 0.20
P90_UPPER_MAX_WORSENING_FT = 0.50
REGION_FOLD_P90_MAX_WORSENING_FT = 1.0

SCORER_SOURCE_FILES = (
    "research/structural_field_score.py",
    "research/test_structural_field_score.py",
)

SCORED_COLUMNS = (
    "mode",
    "repeat",
    "fold",
    "well",
    "equality_group",
    "n_rows",
    "joint_sse",
    "candidate_sse",
    "joint_rmse",
    "candidate_rmse",
    "rmse_gain_ft",
    "mean_field_confidence",
    "support_fraction",
)


class ScoreError(RuntimeError):
    """Fail-closed protocol, prediction, truth, or score error."""


TruthLoader = Callable[[Path], ArrayLike]


@dataclass(frozen=True)
class ScoreAudit:
    """Everything that must validate before the truth boundary."""

    gate_audit: gate.GateAudit
    protocol_path: Path
    benchmark_path: Path
    shard_dir: Path
    barrier_path: Path
    barrier_sha256: str
    barrier_sidecar_sha256: str
    shard_inventory: dict[str, Any]

    @property
    def shard_inventory_sha256(self) -> str:
        return str(self.shard_inventory["inventory_sha256"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ScoreError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScoreError(f"could not parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ScoreError(f"{label} must be a JSON object")
    return payload


def _read_sidecar(path: Path) -> str:
    sidecar = gate._sha_sidecar(path)
    if not sidecar.is_file():
        raise ScoreError(f"missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name or not _is_sha256(fields[0]):
        raise ScoreError(f"malformed SHA-256 sidecar: {sidecar}")
    return fields[0]


def _score_contract() -> dict[str, Any]:
    """The fixed, no-selection measurement contract recorded in the result."""

    return {
        "candidate": "single_frozen_anchored_structural_field_candidate",
        "comparator": "sealed_joint_prediction",
        "selection": "forbidden",
        "pad_out": "absent_and_forbidden",
        "support_confidence": "descriptive_only_no_gate_effect",
        "truth_boundary": (
            "gate protocol, benchmark, lineage, sealed aggregate barrier, and all "
            "15 candidate JSON/NPZ shards validate before any TVT loader call"
        ),
        "exact": {
            "primary": "mean_repeat_wise_pooled_rmse_gain_ft",
            "mean_gain_at_least_ft": EXACT_MEAN_GAIN_FT,
            "each_repeat_gain_positive": True,
            "coupled_exact_profile_group_gain_ci95_low_positive": True,
            "paired_median_well_gain_at_least_ft": EXACT_MEDIAN_GAIN_FT,
            "paired_well_definition": (
                "for each well sum SSE and n_rows across both repeats, compute "
                "sqrt(sum SSE/sum n_rows) for each arm, subtract candidate from "
                "joint, then take the median across wells"
            ),
            "top_positive_sse_wells_removed": TOP_WELLS_TO_REMOVE,
            "top_removal_gain_positive": True,
            "p90_population": "both_repeat_records_per_well",
            "p90_worsening": (
                "q0.90(candidate well RMSE)-q0.90(joint well RMSE); not q0.90 "
                "of paired differences"
            ),
            "p90_point_worsening_at_most_ft": P90_POINT_MAX_WORSENING_FT,
            "p90_one_sided_group_bound_below_ft": P90_UPPER_MAX_WORSENING_FT,
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "coupling": (
                    "one exact-profile-group multiplicity vector is shared by "
                    "both repeats; tail quantiles use the coupled repeated-record "
                    "population"
                ),
            },
        },
        "region": {
            "pooled_gain_at_least_ft": REGION_GAIN_FT,
            "each_fold_gain_at_least_ft": REGION_FOLD_GAIN_FT,
            "exhaustive_fold_resamples": 5**5,
            "exhaustive_gain_ci95_low_positive": True,
            "top_positive_sse_wells_removed": TOP_WELLS_TO_REMOVE,
            "top_removal_gain_positive": True,
            "p90_worsening": ("q0.90(candidate well RMSE)-q0.90(joint well RMSE)"),
            "global_p90_point_worsening_at_most_ft": P90_POINT_MAX_WORSENING_FT,
            "global_p90_one_sided_fold_bound_below_ft": P90_UPPER_MAX_WORSENING_FT,
            "each_fold_p90_worsening_at_most_ft": REGION_FOLD_P90_MAX_WORSENING_FT,
        },
        "mode_interaction": "neither mode can rescue the other",
        "status_ceiling": STATUS_CEILING,
    }


def _validate_scorer_source_binding(audit: gate.GateAudit) -> dict[str, str]:
    sources = audit.protocol.get("source_sha256")
    if not isinstance(sources, Mapping):
        raise ScoreError("gate protocol lacks source lineage")
    bound: dict[str, str] = {}
    for relative in SCORER_SOURCE_FILES:
        path = (gate.ROOT / relative).resolve()
        try:
            path.relative_to(gate.ROOT.resolve())
        except ValueError as exc:
            raise ScoreError("scorer source escaped repository root") from exc
        if not path.is_file():
            raise ScoreError(f"bound scorer source is missing: {relative}")
        actual = sha256_file(path)
        if sources.get(relative) != actual:
            raise ScoreError(f"gate protocol does not bind scorer bytes: {relative}")
        bound[relative] = actual
    return bound


def _validate_inventory_shape(inventory: Mapping[str, Any]) -> None:
    expected_keys = {
        "protocol_sha256",
        "benchmark_file",
        "benchmark_sha256",
        "incumbent_inventory_sha256",
        "shard_count",
        "items",
        "inventory_sha256",
    }
    if set(inventory) != expected_keys:
        raise ScoreError("pretruth shard inventory schema drift")
    payload = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if inventory.get("inventory_sha256") != gate._canonical_digest(payload):
        raise ScoreError("pretruth shard inventory digest drift")
    items = inventory.get("items")
    if not isinstance(items, list) or len(items) != EXPECTED_SHARDS:
        raise ScoreError("pretruth shard inventory is incomplete")
    expected = set(gate._all_fold_identities())
    identities: list[tuple[str, int, int]] = []
    metadata_names: list[str] = []
    prediction_names: list[str] = []
    required_item = {
        "mode",
        "repeat",
        "fold",
        "metadata_name",
        "metadata_sha256",
        "metadata_sidecar_sha256",
        "prediction_name",
        "prediction_sha256",
        "prediction_logical_sha256",
    }
    for item in items:
        if not isinstance(item, Mapping) or set(item) != required_item:
            raise ScoreError("pretruth shard inventory item schema drift")
        identity = (str(item["mode"]), int(item["repeat"]), int(item["fold"]))
        identities.append(identity)
        metadata_names.append(str(item["metadata_name"]))
        prediction_names.append(str(item["prediction_name"]))
        for key in (
            "metadata_sha256",
            "metadata_sidecar_sha256",
            "prediction_sha256",
            "prediction_logical_sha256",
        ):
            if not _is_sha256(item[key]):
                raise ScoreError(f"pretruth shard inventory has invalid {key}")
    if set(identities) != expected or len(identities) != len(set(identities)):
        raise ScoreError("pretruth shard inventory identities drift")
    if len(metadata_names) != len(set(metadata_names)) or len(prediction_names) != len(
        set(prediction_names)
    ):
        raise ScoreError("pretruth shard inventory aliases artifact names")


def _validate_exact_shard_directory(shard_dir: Path) -> None:
    directory = shard_dir.resolve()
    if not directory.is_dir():
        raise ScoreError(f"candidate shard directory is missing: {directory}")
    json_names = {
        gate._field_shard_name(mode, repeat, fold)
        for mode, repeat, fold in gate._all_fold_identities()
    }
    npz_names = {Path(name).with_suffix(".npz").name for name in json_names}
    sidecar_names = {name + ".sha256" for name in json_names}
    expected = json_names | npz_names | sidecar_names
    entries = list(directory.iterdir())
    present = {path.name for path in entries}
    if present != expected or any(not path.is_file() for path in entries):
        raise ScoreError(
            "candidate shard directory is partial, padded, or contains extra artifacts"
        )
    resolved = [
        str((directory / name).resolve()).casefold() for name in sorted(expected)
    ]
    if len(resolved) != len(set(resolved)):
        raise ScoreError("candidate shard artifacts alias resolved paths")
    file_ids = []
    for name in sorted(expected):
        stat = (directory / name).stat()
        file_ids.append((int(stat.st_dev), int(stat.st_ino)))
    if all(inode != 0 for _, inode in file_ids) and len(file_ids) != len(set(file_ids)):
        raise ScoreError("candidate shard artifacts alias hard-linked bytes")


def _load_sealed_barrier(path: Path) -> tuple[dict[str, Any], str, str]:
    barrier_path = path.resolve()
    barrier_hash = sha256_file(barrier_path) if barrier_path.is_file() else ""
    if not barrier_hash or _read_sidecar(barrier_path) != barrier_hash:
        raise ScoreError("sealed aggregate barrier sidecar drift")
    payload = _read_json(barrier_path, "sealed aggregate barrier")
    if set(payload) != {
        "status",
        "status_ceiling",
        "pretruth_field_inventory",
        "next_phase",
    }:
        raise ScoreError("sealed aggregate barrier schema drift")
    if (
        payload.get("status")
        != ("MEASURE_ONLY_ALL_FIELD_SHARDS_AUDITED_TRUTH_STILL_UNREAD")
        or payload.get("status_ceiling") != gate.STATUS_CEILING
    ):
        raise ScoreError("sealed aggregate barrier status drift")
    if not isinstance(payload.get("next_phase"), str) or not str(
        payload["next_phase"]
    ).startswith("HOLD:"):
        raise ScoreError("sealed aggregate barrier next-phase marker drift")
    inventory = payload.get("pretruth_field_inventory")
    if not isinstance(inventory, dict):
        raise ScoreError("sealed aggregate barrier lacks shard inventory")
    _validate_inventory_shape(inventory)
    sidecar_hash = sha256_file(gate._sha_sidecar(barrier_path))
    return inventory, barrier_hash, sidecar_hash


def audit_before_truth(
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path,
    barrier_path: Path,
) -> ScoreAudit:
    """Deep-audit gate lineage and all shards without reading validation TVT."""

    try:
        gate_audit = gate.audit_protocol(protocol_path, verify_data=True)
        _validate_scorer_source_binding(gate_audit)
        benchmark = benchmark_path.resolve()
        directory = shard_dir.resolve()
        barrier = barrier_path.resolve()
        _validate_exact_shard_directory(directory)
        sealed_inventory, barrier_hash, barrier_sidecar_hash = _load_sealed_barrier(
            barrier
        )
        fresh_inventory = gate.build_pretruth_field_inventory(
            protocol_path, benchmark, directory
        )
    except (gate.GateError, OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, ScoreError):
            raise
        raise ScoreError("pretruth gate or lineage audit failed") from exc
    _validate_inventory_shape(fresh_inventory)
    if fresh_inventory != sealed_inventory:
        raise ScoreError("sealed aggregate barrier differs from fresh shard inventory")
    if fresh_inventory["protocol_sha256"] != gate_audit.protocol_sha256:
        raise ScoreError("shard inventory protocol binding drift")
    if fresh_inventory["benchmark_file"] != benchmark.name or fresh_inventory[
        "benchmark_sha256"
    ] != sha256_file(benchmark):
        raise ScoreError("shard inventory benchmark binding drift")
    incumbent = gate_audit.protocol.get("incumbent_pretruth_inventory", {})
    if fresh_inventory["incumbent_inventory_sha256"] != incumbent.get(
        "inventory_sha256"
    ):
        raise ScoreError("shard inventory incumbent binding drift")
    protected = [
        gate_audit.protocol_path.resolve(),
        gate._sha_sidecar(gate_audit.protocol_path).resolve(),
        benchmark,
        gate._sha_sidecar(benchmark).resolve(),
        barrier,
        gate._sha_sidecar(barrier).resolve(),
    ]
    protected.extend(path.resolve() for path in directory.iterdir())
    if len({str(path).casefold() for path in protected}) != len(protected):
        raise ScoreError("protocol, benchmark, barrier, or shard paths alias")
    protected_ids = [
        (int(path.stat().st_dev), int(path.stat().st_ino)) for path in protected
    ]
    if all(inode != 0 for _, inode in protected_ids) and len(protected_ids) != len(
        set(protected_ids)
    ):
        raise ScoreError("protocol, benchmark, barrier, or shard bytes are hard-linked")
    return ScoreAudit(
        gate_audit=gate_audit,
        protocol_path=gate_audit.protocol_path.resolve(),
        benchmark_path=benchmark,
        shard_dir=directory,
        barrier_path=barrier,
        barrier_sha256=barrier_hash,
        barrier_sidecar_sha256=barrier_sidecar_hash,
        shard_inventory=fresh_inventory,
    )


def _default_truth_loader(path: Path) -> NDArray[np.float64]:
    """Read only the labeled TVT column after the global barrier has passed."""

    try:
        frame = pd.read_csv(path, usecols=["TVT"])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ScoreError(f"could not read validation TVT: {path.name}") from exc
    return frame["TVT"].to_numpy(dtype=np.float64)


def _load_truth(
    audit: ScoreAudit,
    well: str,
    truth_loader: TruthLoader,
) -> NDArray[np.float64]:
    inventory = gate._inventory_by_well(audit.gate_audit)
    if well not in inventory.index:
        raise ScoreError(f"scored well is absent from data inventory: {well}")
    path = gate._well_file(audit.gate_audit, well)
    try:
        truth = np.asarray(truth_loader(path), dtype=np.float64)
    except ScoreError:
        raise
    except Exception as exc:
        raise ScoreError(f"truth loader failed for {well}") from exc
    expected_rows = int(inventory.loc[well, "rows"])
    if truth.ndim != 1 or len(truth) != expected_rows:
        raise ScoreError(f"truth loader returned wrong shape for {well}")
    if not np.isfinite(truth).all():
        raise ScoreError(f"truth loader returned nonfinite TVT for {well}")
    return truth


def _score_shards(
    audit: ScoreAudit,
    truth_loader: TruthLoader,
) -> pd.DataFrame:
    truth_cache: dict[str, NDArray[np.float64]] = {}
    inventory = gate._inventory_by_well(audit.gate_audit)
    scored: list[dict[str, Any]] = []
    for mode, repeat, fold in gate._all_fold_identities():
        shard_path = audit.shard_dir / gate._field_shard_name(mode, repeat, fold)
        shard = gate._read_json(shard_path, "field shard")
        try:
            arrays = gate._validate_field_shard(
                shard,
                audit.gate_audit,
                mode,
                repeat,
                fold,
                shard_path,
                audit.benchmark_path,
            )
            _, validation_ids, _, group_by_well = gate._outer_roles(
                audit.gate_audit, mode, repeat, fold
            )
        except (gate.GateError, KeyError, TypeError, ValueError) as exc:
            raise ScoreError("candidate shard drifted after pretruth barrier") from exc
        for well_index, well in enumerate(validation_ids):
            if well not in truth_cache:
                truth_cache[well] = _load_truth(audit, well, truth_loader)
            truth = truth_cache[well]
            selected = np.asarray(arrays["well_index"] == well_index, dtype=bool)
            row_index = np.asarray(arrays["row_index"][selected], dtype=np.int64)
            prefix = int(inventory.loc[well, "prefix_rows"])
            total = int(inventory.loc[well, "rows"])
            expected = np.arange(prefix, total, dtype=np.int64)
            if not np.array_equal(row_index, expected):
                raise ScoreError(f"sealed suffix identity drift for {well}")
            suffix_truth = truth[row_index]
            joint = np.asarray(arrays["joint_prediction"][selected], dtype=float)
            candidate = np.asarray(
                arrays["candidate_prediction"][selected], dtype=float
            )
            confidence = np.asarray(arrays["field_confidence"][selected], dtype=float)
            joint_sse = float(np.sum(np.square(joint - suffix_truth), dtype=np.float64))
            candidate_sse = float(
                np.sum(np.square(candidate - suffix_truth), dtype=np.float64)
            )
            n_rows = len(row_index)
            numeric = np.asarray((joint_sse, candidate_sse, n_rows), dtype=float)
            if not np.isfinite(numeric).all() or n_rows <= 0:
                raise ScoreError(f"nonfinite score for {well}")
            joint_rmse = float(np.sqrt(joint_sse / n_rows))
            candidate_rmse = float(np.sqrt(candidate_sse / n_rows))
            scored.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "fold": fold,
                    "well": well,
                    "equality_group": str(group_by_well[well]),
                    "n_rows": n_rows,
                    "joint_sse": joint_sse,
                    "candidate_sse": candidate_sse,
                    "joint_rmse": joint_rmse,
                    "candidate_rmse": candidate_rmse,
                    "rmse_gain_ft": joint_rmse - candidate_rmse,
                    "mean_field_confidence": float(np.mean(confidence)),
                    "support_fraction": float(np.mean(confidence > 0.0)),
                }
            )
    frame = pd.DataFrame(scored, columns=SCORED_COLUMNS)
    _validate_scored_rows(frame, audit.gate_audit)
    return frame


def _validate_scored_rows(rows: pd.DataFrame, audit: gate.GateAudit) -> None:
    if tuple(rows.columns) != SCORED_COLUMNS:
        raise ScoreError("per-well score schema drift")
    numeric = rows[
        [
            "repeat",
            "fold",
            "n_rows",
            "joint_sse",
            "candidate_sse",
            "joint_rmse",
            "candidate_rmse",
            "rmse_gain_ft",
            "mean_field_confidence",
            "support_fraction",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ScoreError("per-well scores contain nonfinite data")
    n_rows_raw = rows["n_rows"].to_numpy(dtype=float)
    if np.any(n_rows_raw <= 0.0) or not np.array_equal(n_rows_raw, np.rint(n_rows_raw)):
        raise ScoreError("per-well n_rows must be positive integers")
    for column in ("joint_sse", "candidate_sse", "joint_rmse", "candidate_rmse"):
        if np.any(rows[column].to_numpy(dtype=float) < 0.0):
            raise ScoreError(f"per-well scores contain negative {column}")
    n_rows = rows["n_rows"].to_numpy(dtype=float)
    expected_joint = np.sqrt(rows["joint_sse"].to_numpy(dtype=float) / n_rows)
    expected_candidate = np.sqrt(rows["candidate_sse"].to_numpy(dtype=float) / n_rows)
    if not np.allclose(
        rows["joint_rmse"], expected_joint, rtol=1.0e-13, atol=1.0e-13
    ) or not np.allclose(
        rows["candidate_rmse"], expected_candidate, rtol=1.0e-13, atol=1.0e-13
    ):
        raise ScoreError("per-well RMSE does not reconstruct from SSE and row count")
    if not np.allclose(
        rows["rmse_gain_ft"],
        expected_joint - expected_candidate,
        rtol=1.0e-13,
        atol=1.0e-13,
    ):
        raise ScoreError("per-well gain does not reconstruct from arm RMSEs")
    if np.any(rows["mean_field_confidence"].to_numpy(dtype=float) < 0.0) or np.any(
        rows["mean_field_confidence"].to_numpy(dtype=float) > 1.0
    ):
        raise ScoreError("descriptive confidence escaped [0, 1]")
    if np.any(rows["support_fraction"].to_numpy(dtype=float) < 0.0) or np.any(
        rows["support_fraction"].to_numpy(dtype=float) > 1.0
    ):
        raise ScoreError("descriptive support fraction escaped [0, 1]")
    universe = set(gate._inventory_by_well(audit).index.astype(str))
    exact = rows.loc[rows["mode"] == EXACT_MODE]
    region = rows.loc[rows["mode"] == REGION_MODE]
    if len(exact) != 2 * len(universe) or len(region) != len(universe):
        raise ScoreError("scored mode coverage is incomplete")
    if exact.duplicated(["repeat", "well"]).any() or region["well"].duplicated().any():
        raise ScoreError("scored mode coverage contains duplicate wells")
    for repeat in range(2):
        if set(exact.loc[exact["repeat"] == repeat, "well"].astype(str)) != universe:
            raise ScoreError("exact repeat does not score every well once")
    if set(region["well"].astype(str)) != universe:
        raise ScoreError("region mode does not score every well once")
    inventory = gate._inventory_by_well(audit)
    expected_identities = set(gate._all_fold_identities())
    observed_identities = {
        (str(row.mode), int(row.repeat), int(row.fold))
        for row in rows[["mode", "repeat", "fold"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_identities != expected_identities:
        raise ScoreError("per-well score fold identities drift")
    for identity in gate._all_fold_identities():
        mode, repeat, fold = identity
        _, validation_ids, _, group_by_well = gate._outer_roles(
            audit, mode, repeat, fold
        )
        frame = rows.loc[
            (rows["mode"] == mode) & (rows["repeat"] == repeat) & (rows["fold"] == fold)
        ]
        if set(frame["well"].astype(str)) != set(validation_ids):
            raise ScoreError("per-well score membership differs from frozen roles")
        for row in frame.itertuples(index=False):
            well = str(row.well)
            if str(row.equality_group) != str(group_by_well[well]):
                raise ScoreError("per-well score equality-group lineage drift")
            if int(row.n_rows) != int(inventory.loc[well, "suffix_rows"]):
                raise ScoreError("per-well score suffix-row lineage drift")


def _pooled_gain(rows: pd.DataFrame) -> float:
    if rows.empty or np.any(rows["n_rows"].to_numpy(dtype=float) <= 0.0):
        raise ScoreError("pooled gain received empty or invalid rows")
    n_rows = float(rows["n_rows"].sum())
    return float(
        np.sqrt(float(rows["joint_sse"].sum()) / n_rows)
        - np.sqrt(float(rows["candidate_sse"].sum()) / n_rows)
    )


def _repeat_pooled_gains(rows: pd.DataFrame) -> dict[str, float]:
    repeats = sorted(rows["repeat"].astype(int).unique().tolist())
    if repeats != [0, 1]:
        raise ScoreError("exact score requires repeats 0 and 1")
    return {
        str(repeat): _pooled_gain(rows.loc[rows["repeat"] == repeat])
        for repeat in repeats
    }


def _mean_repeat_pooled_gain(rows: pd.DataFrame) -> float:
    return float(np.mean(list(_repeat_pooled_gains(rows).values())))


def _linear_quantile(values: NDArray[np.float64], quantile: float) -> float:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ScoreError("quantile received invalid values")
    return float(np.quantile(values, quantile, method="linear"))


def _replicated_linear_quantile(
    values: NDArray[np.float64],
    multiplicity: NDArray[np.float64],
    quantile: float,
) -> float:
    """Match NumPy linear quantiles after integer cluster replication."""

    if (
        values.ndim != 1
        or multiplicity.ndim != 1
        or len(values) != len(multiplicity)
        or not np.isfinite(values).all()
        or not np.isfinite(multiplicity).all()
        or np.any(multiplicity < 0.0)
        or not np.array_equal(multiplicity, np.rint(multiplicity))
    ):
        raise ScoreError("replicated quantile received invalid values")
    weights = np.rint(multiplicity).astype(np.int64)
    total = int(weights.sum())
    if total <= 0:
        raise ScoreError("replicated quantile has zero mass")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    position = (total - 1) * quantile
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    lower_index = min(
        int(np.searchsorted(cumulative, lower, side="right")), len(values) - 1
    )
    upper_index = min(
        int(np.searchsorted(cumulative, upper, side="right")), len(values) - 1
    )
    fraction = position - lower
    return float(
        ordered_values[lower_index]
        + fraction * (ordered_values[upper_index] - ordered_values[lower_index])
    )


def _p90_worsening(rows: pd.DataFrame) -> dict[str, float]:
    joint = _linear_quantile(rows["joint_rmse"].to_numpy(dtype=float), 0.9)
    candidate = _linear_quantile(rows["candidate_rmse"].to_numpy(dtype=float), 0.9)
    return {
        "population_records": int(len(rows)),
        "joint_q0_90_well_rmse_ft": joint,
        "candidate_q0_90_well_rmse_ft": candidate,
        "worsening_ft": candidate - joint,
    }


def _paired_exact_wells(rows: pd.DataFrame) -> pd.DataFrame:
    counts = rows.groupby("well", sort=True)["repeat"].nunique()
    if not (counts == 2).all():
        raise ScoreError("paired exact well metric lacks both repeats")
    paired = (
        rows.groupby(["well", "equality_group"], sort=True)
        .agg(
            n_rows=("n_rows", "sum"),
            joint_sse=("joint_sse", "sum"),
            candidate_sse=("candidate_sse", "sum"),
        )
        .reset_index()
    )
    paired["joint_rmse"] = np.sqrt(paired["joint_sse"] / paired["n_rows"])
    paired["candidate_rmse"] = np.sqrt(paired["candidate_sse"] / paired["n_rows"])
    paired["rmse_gain_ft"] = paired["joint_rmse"] - paired["candidate_rmse"]
    return paired


def _exact_group_bootstrap(
    rows: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply one group multiplicity vector to both exact repeat populations."""

    if draws < 100:
        raise ScoreError("exact bootstrap requires at least 100 draws")
    groups = sorted(rows["equality_group"].astype(str).unique().tolist())
    if not groups:
        raise ScoreError("exact bootstrap has no equality groups")
    group_index = {group: index for index, group in enumerate(groups)}
    row_group = rows["equality_group"].astype(str).map(group_index).to_numpy(dtype=int)
    joint_well = rows["joint_rmse"].to_numpy(dtype=float)
    candidate_well = rows["candidate_rmse"].to_numpy(dtype=float)
    contributions: dict[int, dict[str, NDArray[np.float64]]] = {}
    for repeat in range(2):
        grouped = (
            rows.loc[rows["repeat"] == repeat]
            .groupby("equality_group", sort=True)
            .agg(
                n_rows=("n_rows", "sum"),
                joint_sse=("joint_sse", "sum"),
                candidate_sse=("candidate_sse", "sum"),
            )
            .reindex(groups)
        )
        if grouped.isna().any().any():
            raise ScoreError("exact groups do not appear in both repeats")
        contributions[repeat] = {
            name: grouped[name].to_numpy(dtype=float)
            for name in ("n_rows", "joint_sse", "candidate_sse")
        }
    rng = np.random.default_rng(seed)
    gain_samples = np.empty(draws, dtype=float)
    p90_samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        multiplicity = np.bincount(sampled, minlength=len(groups)).astype(float)
        repeat_gains = []
        for repeat in range(2):
            part = contributions[repeat]
            count = float(multiplicity @ part["n_rows"])
            repeat_gains.append(
                np.sqrt(float(multiplicity @ part["joint_sse"]) / count)
                - np.sqrt(float(multiplicity @ part["candidate_sse"]) / count)
            )
        gain_samples[draw] = float(np.mean(repeat_gains))
        weights = multiplicity[row_group]
        p90_samples[draw] = _replicated_linear_quantile(
            candidate_well, weights, 0.9
        ) - _replicated_linear_quantile(joint_well, weights, 0.9)
    return {
        "unit": "exact_profile_equality_group",
        "repeat_coupling": "same_group_multiplicity_vector_for_both_repeats",
        "p90_population": "coupled_two_repeat_well_records",
        "groups": len(groups),
        "draws": draws,
        "seed": seed,
        "gain_ci95_low_ft": float(np.quantile(gain_samples, 0.025)),
        "gain_ci95_high_ft": float(np.quantile(gain_samples, 0.975)),
        "p90_worsening_one_sided_95_upper_ft": float(np.quantile(p90_samples, 0.95)),
    }


def _region_exhaustive_bootstrap(rows: pd.DataFrame) -> dict[str, Any]:
    """Enumerate all 5^5 ordered region-fold resamples with replacement."""

    folds = sorted(rows["fold"].astype(int).unique().tolist())
    if folds != list(range(5)):
        raise ScoreError("region bootstrap requires exactly folds 0..4")
    grouped = rows.groupby("fold", sort=True).agg(
        n_rows=("n_rows", "sum"),
        joint_sse=("joint_sse", "sum"),
        candidate_sse=("candidate_sse", "sum"),
    )
    n_rows = grouped["n_rows"].to_numpy(dtype=float)
    joint_sse = grouped["joint_sse"].to_numpy(dtype=float)
    candidate_sse = grouped["candidate_sse"].to_numpy(dtype=float)
    row_fold = rows["fold"].to_numpy(dtype=int)
    joint_well = rows["joint_rmse"].to_numpy(dtype=float)
    candidate_well = rows["candidate_rmse"].to_numpy(dtype=float)
    draws = 5**5
    gain_samples = np.empty(draws, dtype=float)
    p90_samples = np.empty(draws, dtype=float)
    for draw, sample in enumerate(itertools.product(range(5), repeat=5)):
        multiplicity = np.bincount(sample, minlength=5).astype(float)
        count = float(multiplicity @ n_rows)
        gain_samples[draw] = np.sqrt(float(multiplicity @ joint_sse) / count) - np.sqrt(
            float(multiplicity @ candidate_sse) / count
        )
        weights = multiplicity[row_fold]
        p90_samples[draw] = _replicated_linear_quantile(
            candidate_well, weights, 0.9
        ) - _replicated_linear_quantile(joint_well, weights, 0.9)
    return {
        "unit": "sealed_region_fold",
        "method": "exhaustive_ordered_resampling_with_replacement",
        "draws": draws,
        "gain_ci95_low_ft": float(np.quantile(gain_samples, 0.025)),
        "gain_ci95_high_ft": float(np.quantile(gain_samples, 0.975)),
        "p90_worsening_one_sided_95_upper_ft": float(np.quantile(p90_samples, 0.95)),
    }


def _top_positive_removal(
    rows: pd.DataFrame,
    *,
    exact_repeats: bool,
) -> dict[str, Any]:
    contribution = (
        rows.groupby("well", sort=True)
        .agg(joint_sse=("joint_sse", "sum"), candidate_sse=("candidate_sse", "sum"))
        .reset_index()
    )
    contribution["positive_sse_reduction"] = (
        contribution["joint_sse"] - contribution["candidate_sse"]
    )
    positive = contribution.loc[
        contribution["positive_sse_reduction"] > 0.0
    ].sort_values(
        ["positive_sse_reduction", "well"],
        ascending=[False, True],
        kind="mergesort",
    )
    removed = positive.head(TOP_WELLS_TO_REMOVE)["well"].astype(str).tolist()
    retained = rows.loc[~rows["well"].astype(str).isin(removed)]
    if retained.empty:
        raise ScoreError("top-positive removal left no scored wells")
    gain = (
        _mean_repeat_pooled_gain(retained) if exact_repeats else _pooled_gain(retained)
    )
    return {
        "requested_removal": TOP_WELLS_TO_REMOVE,
        "removed_positive_wells": removed,
        "removed_count": len(removed),
        "remaining_gain_ft": gain,
        "passed": bool(gain > 0.0),
    }


def _descriptive_support(rows: pd.DataFrame) -> dict[str, Any]:
    weights = rows["n_rows"].to_numpy(dtype=float)
    return {
        "row_weighted_mean_field_confidence": float(
            np.average(rows["mean_field_confidence"], weights=weights)
        ),
        "row_weighted_support_fraction": float(
            np.average(rows["support_fraction"], weights=weights)
        ),
        "gate_effect": "none",
    }


def _exact_result(rows: pd.DataFrame) -> dict[str, Any]:
    repeat_gains = _repeat_pooled_gains(rows)
    mean_gain = float(np.mean(list(repeat_gains.values())))
    paired = _paired_exact_wells(rows)
    median_gain = float(np.median(paired["rmse_gain_ft"].to_numpy(dtype=float)))
    bootstrap = _exact_group_bootstrap(rows)
    p90 = _p90_worsening(rows)
    influence = _top_positive_removal(rows, exact_repeats=True)
    gates = {
        "mean_repeat_pooled_gain_at_least_1ft": bool(mean_gain >= EXACT_MEAN_GAIN_FT),
        "both_repeat_gains_positive": bool(
            all(value > 0.0 for value in repeat_gains.values())
        ),
        "coupled_group_gain_ci95_low_positive": bool(
            bootstrap["gain_ci95_low_ft"] > 0.0
        ),
        "paired_median_well_gain_at_least_0_5ft": bool(
            median_gain >= EXACT_MEDIAN_GAIN_FT
        ),
        "top10_positive_sse_removal_gain_positive": bool(influence["passed"]),
        "p90_point_worsening_at_most_0_2ft": bool(
            p90["worsening_ft"] <= P90_POINT_MAX_WORSENING_FT
        ),
        "p90_one_sided_group_bound_below_0_5ft": bool(
            bootstrap["p90_worsening_one_sided_95_upper_ft"]
            < P90_UPPER_MAX_WORSENING_FT
        ),
    }
    return {
        "primary_metric": "mean_repeat_wise_pooled_rmse_gain_ft",
        "repeat_pooled_gain_ft": repeat_gains,
        "mean_repeat_pooled_gain_ft": mean_gain,
        "paired_median_well_gain_ft": median_gain,
        "paired_well_definition": (
            "per well sqrt(sum repeat SSE/sum repeat rows) for each arm, then "
            "joint minus candidate; median is across wells"
        ),
        "coupled_exact_profile_group_bootstrap": bootstrap,
        "p90_distribution_guard": {
            **p90,
            "definition": (
                "q0.90(candidate well RMSE)-q0.90(joint well RMSE) over both "
                "repeat records; not q0.90 of paired differences"
            ),
        },
        "top10_positive_sse_removal": influence,
        "descriptive_support_confidence": _descriptive_support(rows),
        "fixed_gates": {**gates, "passed": bool(all(gates.values()))},
    }


def _region_result(rows: pd.DataFrame) -> dict[str, Any]:
    pooled_gain = _pooled_gain(rows)
    fold_gains = {
        str(fold): _pooled_gain(rows.loc[rows["fold"] == fold]) for fold in range(5)
    }
    fold_p90 = {
        str(fold): _p90_worsening(rows.loc[rows["fold"] == fold])["worsening_ft"]
        for fold in range(5)
    }
    bootstrap = _region_exhaustive_bootstrap(rows)
    p90 = _p90_worsening(rows)
    influence = _top_positive_removal(rows, exact_repeats=False)
    gates = {
        "pooled_gain_at_least_0_75ft": bool(pooled_gain >= REGION_GAIN_FT),
        "every_fold_gain_at_least_0_2ft": bool(
            all(value >= REGION_FOLD_GAIN_FT for value in fold_gains.values())
        ),
        "exhaustive_fold_gain_ci95_low_positive": bool(
            bootstrap["gain_ci95_low_ft"] > 0.0
        ),
        "top10_positive_sse_removal_gain_positive": bool(influence["passed"]),
        "global_p90_point_worsening_at_most_0_2ft": bool(
            p90["worsening_ft"] <= P90_POINT_MAX_WORSENING_FT
        ),
        "global_p90_one_sided_fold_bound_below_0_5ft": bool(
            bootstrap["p90_worsening_one_sided_95_upper_ft"]
            < P90_UPPER_MAX_WORSENING_FT
        ),
        "every_fold_p90_worsening_at_most_1ft": bool(
            all(
                value <= REGION_FOLD_P90_MAX_WORSENING_FT for value in fold_p90.values()
            )
        ),
    }
    return {
        "primary_metric": "pooled_rmse_gain_ft",
        "pooled_gain_ft": pooled_gain,
        "fold_pooled_gain_ft": fold_gains,
        "exhaustive_5_to_5_fold_cluster_bootstrap": bootstrap,
        "global_p90_distribution_guard": {
            **p90,
            "definition": (
                "q0.90(candidate well RMSE)-q0.90(joint well RMSE) over the "
                "region-fold/well population"
            ),
        },
        "fold_p90_worsening_ft": fold_p90,
        "top10_positive_sse_removal": influence,
        "descriptive_support_confidence": _descriptive_support(rows),
        "descriptive_support_confidence_by_region_fold": {
            str(fold): _descriptive_support(rows.loc[rows["fold"] == fold])
            for fold in range(5)
        },
        "fixed_gates": {**gates, "passed": bool(all(gates.values()))},
    }


def _per_well_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_per_well.csv")


def _output_targets(output_path: Path) -> tuple[Path, Path, Path, Path]:
    per_well = _per_well_path(output_path)
    return (
        per_well,
        gate._sha_sidecar(per_well),
        output_path,
        gate._sha_sidecar(output_path),
    )


def _assert_write_once(output_path: Path) -> None:
    existing = [str(path) for path in _output_targets(output_path) if path.exists()]
    if existing:
        raise ScoreError(f"refusing to overwrite score artifacts: {existing}")


def _validate_output_location(
    output_path: Path,
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path,
    barrier_path: Path,
) -> None:
    """Keep result writes outside every sealed input and shard namespace."""

    output = output_path.resolve()
    directory = shard_dir.resolve()
    targets = tuple(path.resolve() for path in _output_targets(output))
    if len({str(path).casefold() for path in targets}) != len(targets):
        raise ScoreError("score output targets alias one another")
    for target in targets:
        try:
            target.relative_to(directory)
        except ValueError:
            pass
        else:
            raise ScoreError(
                "score output targets must be outside candidate shard directory"
            )
    protected = {
        path.resolve()
        for item in (protocol_path, benchmark_path, barrier_path)
        for path in (Path(item), gate._sha_sidecar(Path(item)))
    }
    if directory.is_dir():
        protected.update(path.resolve() for path in directory.iterdir())
    protected_keys = {str(path).casefold() for path in protected}
    if any(str(target).casefold() in protected_keys for target in targets):
        raise ScoreError("score output target aliases sealed lineage")


def _persist_write_once(
    output_path: Path,
    rows: pd.DataFrame,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    output = output_path.resolve()
    _assert_write_once(output)
    per_well, per_well_sidecar, result_path, result_sidecar = _output_targets(output)
    ordered = rows.loc[:, SCORED_COLUMNS].sort_values(
        ["mode", "repeat", "fold", "well"], kind="mergesort"
    )
    csv_bytes = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    sealed = dict(result)
    sealed["per_well_artifact"] = {
        "name": per_well.name,
        "rows": len(ordered),
        "columns": list(SCORED_COLUMNS),
        "byte_sha256": csv_hash,
        "sidecar": per_well_sidecar.name,
    }
    sealed["result_sidecar"] = result_sidecar.name
    try:
        result_bytes = (
            json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScoreError("result is not finite canonical JSON") from exc
    result_hash = hashlib.sha256(result_bytes).hexdigest()
    payloads = (
        csv_bytes,
        f"{csv_hash}  {per_well.name}\n".encode("ascii"),
        result_bytes,
        f"{result_hash}  {result_path.name}\n".encode("ascii"),
    )
    targets = (per_well, per_well_sidecar, result_path, result_sidecar)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tuple(
        path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in targets
    )
    if any(path.exists() for path in temporary):
        raise ScoreError("stale temporary score artifact exists")
    try:
        for path, payload in zip(temporary, payloads, strict=True):
            path.write_bytes(payload)
        _assert_write_once(output)
        for source, target in zip(temporary, targets, strict=True):
            os.replace(source, target)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()
    if _read_sidecar(per_well) != sha256_file(per_well):
        raise ScoreError("per-well CSV failed postwrite seal audit")
    if _read_sidecar(result_path) != sha256_file(result_path):
        raise ScoreError("final JSON failed postwrite seal audit")
    return sealed


def _same_audit(before: ScoreAudit, after: ScoreAudit) -> bool:
    return bool(
        before.gate_audit.protocol_sha256 == after.gate_audit.protocol_sha256
        and before.barrier_sha256 == after.barrier_sha256
        and before.barrier_sidecar_sha256 == after.barrier_sidecar_sha256
        and before.shard_inventory == after.shard_inventory
        and before.shard_inventory_sha256 == after.shard_inventory_sha256
        and before.benchmark_path == after.benchmark_path
        and sha256_file(before.benchmark_path) == sha256_file(after.benchmark_path)
    )


def _build_result(audit: ScoreAudit, scored: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct the complete metric and lineage payload from per-well rows."""

    exact_rows = scored.loc[scored["mode"] == EXACT_MODE].copy()
    region_rows = scored.loc[scored["mode"] == REGION_MODE].copy()
    exact_result = _exact_result(exact_rows)
    region_result = _region_result(region_rows)
    both_pass = bool(
        exact_result["fixed_gates"]["passed"] and region_result["fixed_gates"]["passed"]
    )
    bound_sources = _validate_scorer_source_binding(audit.gate_audit)
    return {
        "score_version": SCORE_VERSION,
        "status": "MEASURE_ONLY" if both_pass else "STOP",
        "status_ceiling": STATUS_CEILING,
        "production_open": False,
        "score_contract": _score_contract(),
        "lineage": {
            "protocol": {
                "name": audit.protocol_path.name,
                "byte_sha256": audit.gate_audit.protocol_sha256,
                "sidecar_sha256": sha256_file(gate._sha_sidecar(audit.protocol_path)),
            },
            "benchmark": {
                "name": audit.benchmark_path.name,
                "byte_sha256": sha256_file(audit.benchmark_path),
                "sidecar_sha256": sha256_file(gate._sha_sidecar(audit.benchmark_path)),
            },
            "sealed_pretruth_barrier": {
                "name": audit.barrier_path.name,
                "byte_sha256": audit.barrier_sha256,
                "sidecar_sha256": audit.barrier_sidecar_sha256,
            },
            "incumbent_inventory_sha256": audit.gate_audit.protocol[
                "incumbent_pretruth_inventory"
            ]["inventory_sha256"],
            "bound_scorer_sources": bound_sources,
        },
        "candidate_shard_inventory": audit.shard_inventory,
        "candidate_shard_inventory_sha256": audit.shard_inventory_sha256,
        "exact_gate": exact_result,
        "region_gate": region_result,
        "mode_interaction": {
            "neither_mode_can_rescue_the_other": True,
            "both_passed": both_pass,
        },
        "interpretation": (
            "Fixed grouped and region-out measurement only; external confirmation "
            "is absent and production OPEN is impossible."
        ),
    }


def score_structural_field(
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path,
    barrier_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Audit, read TVT only, score fixed gates, re-audit, and seal once.

    The public sealing path intentionally exposes no truth-loader injection.
    Tests may spy on the private ``_default_truth_loader`` boundary, but every
    durable result is sourced from the on-disk ``TVT`` column bound by the gate
    data inventory.
    """

    output = output_path.resolve()
    _validate_output_location(
        output, protocol_path, benchmark_path, shard_dir, barrier_path
    )
    _assert_write_once(output)
    audit = audit_before_truth(protocol_path, benchmark_path, shard_dir, barrier_path)
    scored = _score_shards(audit, _default_truth_loader)
    result = _build_result(audit, scored)

    # This is deliberately the last operation before the write-once commit.
    # Any protocol, source, data, benchmark, barrier, or shard drift after the
    # truth boundary prevents all score artifacts from being written.
    commit_audit = audit_before_truth(
        protocol_path, benchmark_path, shard_dir, barrier_path
    )
    _validate_output_location(
        output, protocol_path, benchmark_path, shard_dir, barrier_path
    )
    if not _same_audit(audit, commit_audit):
        raise ScoreError(
            "protocol, data, benchmark, barrier, or shard drifted after score"
        )
    return _persist_write_once(output, scored, result)


def audit_score_artifacts(
    result_path: Path,
    protocol_path: Path,
    benchmark_path: Path,
    shard_dir: Path,
    barrier_path: Path,
) -> dict[str, Any]:
    """Independently audit sealed scores and reconstruct every fixed gate.

    This audit never reads validation truth.  It validates current protocol,
    source, data, benchmark, barrier, and shard lineage, verifies the JSON and
    CSV seals, then recomputes all metrics and gate booleans from the per-well
    CSV and requires exact agreement with the stored result.
    """

    result_file = result_path.resolve()
    _validate_output_location(
        result_file, protocol_path, benchmark_path, shard_dir, barrier_path
    )
    if _read_sidecar(result_file) != sha256_file(result_file):
        raise ScoreError("sealed result JSON sidecar drift")
    payload = _read_json(result_file, "sealed score result")
    if payload.get("production_open") is not False or payload.get("status") not in {
        "MEASURE_ONLY",
        "STOP",
    }:
        raise ScoreError("sealed score result escaped its status ceiling")
    audit = audit_before_truth(protocol_path, benchmark_path, shard_dir, barrier_path)
    descriptor = payload.get("per_well_artifact")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "name",
        "rows",
        "columns",
        "byte_sha256",
        "sidecar",
    }:
        raise ScoreError("sealed result lacks a valid per-well descriptor")
    expected_csv = _per_well_path(result_file)
    if (
        descriptor.get("name") != expected_csv.name
        or descriptor.get("sidecar") != gate._sha_sidecar(expected_csv).name
        or descriptor.get("columns") != list(SCORED_COLUMNS)
        or not _is_sha256(descriptor.get("byte_sha256"))
    ):
        raise ScoreError("per-well descriptor lineage drift")
    if (
        _read_sidecar(expected_csv) != sha256_file(expected_csv)
        or sha256_file(expected_csv) != descriptor["byte_sha256"]
    ):
        raise ScoreError("per-well CSV byte seal drift")
    try:
        rows = pd.read_csv(expected_csv)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ScoreError("could not parse sealed per-well CSV") from exc
    if tuple(rows.columns) != SCORED_COLUMNS or len(rows) != int(descriptor["rows"]):
        raise ScoreError("per-well CSV schema or row count drift")
    _validate_scored_rows(rows, audit.gate_audit)
    reconstructed = _build_result(audit, rows)
    expected_payload = {
        **reconstructed,
        "per_well_artifact": dict(descriptor),
        "result_sidecar": gate._sha_sidecar(result_file).name,
    }
    if payload != expected_payload:
        raise ScoreError("sealed result metrics or lineage do not reconstruct from CSV")
    return {
        "status": "MEASURE_ONLY_SCORE_ARTIFACTS_AUDITED",
        "production_open": False,
        "result_name": result_file.name,
        "result_sha256": sha256_file(result_file),
        "per_well_name": expected_csv.name,
        "per_well_sha256": sha256_file(expected_csv),
        "candidate_shard_inventory_sha256": audit.shard_inventory_sha256,
        "measured_status": payload["status"],
    }


def _add_lineage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="open bound TVT and seal fixed scores")
    _add_lineage_arguments(score)
    score.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit", help="audit sealed scores without reading TVT")
    _add_lineage_arguments(audit)
    audit.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "score":
        result = score_structural_field(
            args.protocol,
            args.benchmark,
            args.shard_dir,
            args.barrier,
            args.output,
        )
    else:
        result = audit_score_artifacts(
            args.result,
            args.protocol,
            args.benchmark,
            args.shard_dir,
            args.barrier,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
