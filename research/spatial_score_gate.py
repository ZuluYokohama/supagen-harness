"""Sealed prediction-before-truth gate for frozen spatial well splits.

This module evaluates exactly one already-selected candidate,
``equal_ordered_joint_v2``, on both predeclared spatial split modes.  The modes
have deliberately different roles:

``region_out``
    Primary spatial falsification gate.  Its manifest-provided embargoed wells
    are excluded from training rather than reassigned.
``pad_out``
    Secondary centroid-component sensitivity diagnostic.  It cannot rescue,
    replace, or veto the primary result and is never selected as a winner.

The lifecycle is intentionally split into four commands.  ``freeze`` seals the
two input manifests, an inference-safe data inventory, executable source, and
the full evaluation contract.  ``run`` fits every learned quantity within one
spatial training fold and writes predictions only.  ``aggregate`` validates all
ten prediction shards before it first opens any validation suffix truth.  All
outputs remain MEASURE_ONLY research artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research import repeated_group_gate as group_gate
from research import spatial_split


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "geosteern-spatial-falsification-gate/1"
METHOD = "equal_ordered_joint_v2"
MODES = ("region_out", "pad_out")
PRIMARY_MODE = "region_out"
SECONDARY_MODE = "pad_out"
N_FOLDS = 5

EXPECTED_ELIGIBLE_WELLS = group_gate.EXPECTED_ELIGIBLE_WELLS
EXPECTED_TYPEWELL_GROUPS = group_gate.EXPECTED_TYPEWELL_GROUPS
EXPECTED_EQUALITY_GROUPS = 749
EXPECTED_SUFFIX_ROWS = 3_769_838
EXPECTED_SPATIAL_DATASET_SHA256 = (
    "9a9e36ded16a390263bf8c9123422a79baefd21779ede2b90ab0693951310465"
)
EXPECTED_MANIFEST_SHA256 = {
    "region_out": {
        "logical": "70af52ed2ca2ed3e77e218471836869ea74771e08eacfbd4e7f93c9067bda622",
        "byte": "af0160c67397c18d4e6cb70da18ba5ea76429973b5b0c8d2a9d4b1f0752ac5df",
    },
    "pad_out": {
        "logical": "f17ec3ec67b1490b7efab6250f7c674b24a7de080e563310dfc8b5ef9a3c4b8c",
        "byte": "c9a80bbdf5acb28d34f542381ddba53a1f50c7b243dee234183a3f01cb5d82e6",
    },
}
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_DRAWS = 4000
EXPECTED_PAD_COMPONENTS = 183
REGION_EXHAUSTIVE_RESAMPLES = N_FOLDS**N_FOLDS
MATERIALITY_FT = 0.2
TOP_POSITIVE_SSE_REMOVAL_WELLS = 10
COEFFICIENT_BOUND_TOLERANCE = group_gate.COEFFICIENT_BOUND_TOLERANCE
ESTIMATED_RUNTIME_HOURS = (2.0, 2.5)

PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
SOURCE_FILES = tuple(
    sorted(
        set(group_gate.SOURCE_FILES)
        | {
            "research/spatial_split.py",
            "research/test_spatial_split.py",
            "research/spatial_score_gate.py",
            "research/test_spatial_score_gate.py",
        }
    )
)

INVENTORY_COLUMNS = (
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
PREDICTION_ARRAYS = (
    "well_index",
    "row_index",
    "base_prediction",
    "typewell_prediction",
    "ordered_prediction",
    "joint_prediction",
)
SCORED_SSE_COLUMNS = (
    "mode",
    "fold",
    "well",
    "typewell_profile_hash",
    "spatial_cluster_id",
    "n_rows",
    "base_sse",
    "typewell_sse",
    "ordered_sse",
    "joint_sse",
)

ProtocolError = group_gate.ProtocolError


@dataclass(frozen=True)
class SpatialAuditBundle:
    """Validated protocol, data inventory, and both spatial manifests."""

    protocol: dict
    inventory: pd.DataFrame
    manifests: dict[str, dict]
    protocol_sha256: str
    protocol_path: Path
    inventory_path: Path
    manifest_paths: dict[str, Path]


def _packages() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ProtocolError(f"required source is missing: {path}")
        hashes[relative] = group_gate.sha256_file(path)
    return hashes


def _inventory_path(protocol_path: Path) -> Path:
    return protocol_path.with_name(protocol_path.stem + "_data_inventory.csv")


def _scored_sse_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_scored_sse.csv")


def _fold_shard_name(mode: str, fold: int) -> str:
    if mode not in MODES:
        raise ProtocolError(f"unsupported spatial mode: {mode}")
    return f"{mode}_fold_{fold}.json"


def _prediction_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".npz")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _id_digest(ids: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(str(value) for value in ids),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_protocol() -> dict:
    """Return the complete predeclared spatial falsification contract."""

    return {
        "mode_policy": {
            "primary": PRIMARY_MODE,
            "primary_role": (
                "buffered spatial-isolation falsification gate using only retained "
                "manifest training IDs"
            ),
            "secondary": SECONDARY_MODE,
            "secondary_role": (
                "centroid-component sensitivity diagnostic only; cross-fold laterals "
                "may approach or cross"
            ),
            "winner_selection": "forbidden",
            "secondary_can_rescue_primary": False,
            "secondary_can_veto_primary": False,
        },
        "primary_metric": (
            "region_out pooled full-suffix-row RMSE gain of frozen joint candidate "
            "versus its fold-local base"
        ),
        "bootstrap": {
            "region_out": {
                "unit": "sealed validation region fold",
                "method": "deterministic exhaustive ordered resampling with replacement",
                "clusters": N_FOLDS,
                "draws": REGION_EXHAUSTIVE_RESAMPLES,
            },
            "pad_out": {
                "unit": "sealed pad component",
                "method": "seeded cluster bootstrap with replacement",
                "expected_clusters": EXPECTED_PAD_COMPONENTS,
                "seed": BOOTSTRAP_SEED,
                "draws": BOOTSTRAP_DRAWS,
            },
            "interval": "percentile 95%",
            "channel_coupling": (
                "each resample applies identical cluster multiplicities to base, "
                "typewell, ordered, and joint SSE and row counts"
            ),
        },
        "interim_metrics": "forbidden",
        "truth_boundary": (
            "all ten JSON/NPZ prediction shards must validate before any outer "
            "validation suffix TVT is opened"
        ),
        "prediction_artifact": {
            "format": "compressed NPZ plus hash-sealed JSON metadata",
            "truth_columns": "forbidden",
            "arrays": list(PREDICTION_ARRAYS),
        },
        "primary_support_criteria": {
            "pooled_rmse_gain_at_least_ft": MATERIALITY_FT,
            "all_five_region_fold_gains_positive": True,
            "sealed_region_fold_bootstrap_ci95_low_positive": True,
            "joint_vs_resample_best_component_ci95_low_positive": True,
            "paired_median_well_gain_at_least_ft": MATERIALITY_FT,
            "top_positive_sse_removal": {
                "unit": "well",
                "remove": TOP_POSITIVE_SSE_REMOVAL_WELLS,
                "remaining_pooled_rmse_gain_positive": True,
            },
            "coefficient_stability": {
                "no_sign_flips_across_region_folds": True,
                "repeated_bound_hit_definition": "two or more folds at one bound",
                "repeated_bound_hits_allowed": False,
                "shrink_bounds": [0.0, 1.5],
                "joint_coefficient_bounds": [-1.0, 2.0],
                "absolute_tolerance": COEFFICIENT_BOUND_TOLERANCE,
            },
        },
        "components": (
            "typewell and ordered channels are reported as predeclared diagnostics; "
            "they are never used to select a candidate or a mode"
        ),
        "aggregate_artifacts": {
            "scored_rows": "per-mode/per-well SSE CSV plus SHA-256 sidecar",
            "result": "final JSON plus SHA-256 sidecar",
            "inventory": (
                "protocol, data inventory, both manifests, all ten shard JSON files "
                "and sidecars, all ten prediction NPZ files, scored SSE, and final "
                "result identity"
            ),
            "overwrite": "forbidden",
        },
        "status_ceiling": "MEASURE_ONLY; cannot promote production OPEN",
    }


def _fit_contract() -> dict:
    if METHOD != group_gate.METHOD:
        raise ProtocolError("spatial gate method drifted from grouped candidate")
    return {
        "method": METHOD,
        "implementation": (
            "imports fold-local base/evidence/shrink/joint primitives from "
            "research/repeated_group_gate.py"
        ),
        "inner_group": (
            "sealed manifest equality group, proven one-to-one with the exact parsed "
            "typewell TVT/GR profile partition on this frozen corpus"
        ),
        "inner_splits": group_gate.INNER_SPLITS,
        "train_stride": group_gate.TRAIN_STRIDE,
        "evaluation_stride": group_gate.EVALUATION_STRIDE,
        "base_model": group_gate.frozen_research_params(),
        "ordered_transport": group_gate._frozen_ordered_settings(),
        "refit_scope": (
            "every model weight, base shrink, typewell shrink, ordered shrink, and "
            "joint coefficient is refit from each manifest-retained spatial "
            "training set using inner exact-profile grouped OOF"
        ),
        "region_embargo_policy": (
            "embargo IDs are excluded and never repurposed as training or validation"
        ),
        "state_isolation": (
            "no fitted model, OOF prediction, coefficient, transform, matrix, or cache "
            "is shared across folds or modes"
        ),
    }


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"could not parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} is not a JSON object: {path}")
    return payload


def _manifest_fold(manifest: Mapping[str, Any], fold: int) -> dict:
    matches = [
        row for row in manifest.get("folds", []) if int(row.get("fold", -1)) == fold
    ]
    if len(matches) != 1:
        raise ProtocolError(f"manifest has no unique fold {fold}")
    return matches[0]


def _validate_spatial_manifest(
    manifest: dict,
    mode: str,
    inventory: pd.DataFrame | None = None,
) -> None:
    """Fail closed on seal, role, provenance, and parsed-profile drift."""

    if mode not in MODES or manifest.get("mode") != mode:
        raise ProtocolError(f"spatial manifest mode mismatch for {mode}")
    if not spatial_split.verify_manifest_sha256(manifest):
        raise ProtocolError(f"{mode} manifest logical SHA-256 drift")
    if manifest.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256[mode]["logical"]:
        raise ProtocolError(f"{mode} is not the predeclared sealed manifest")
    if int(manifest.get("schema_version", -1)) != spatial_split.SCHEMA_VERSION:
        raise ProtocolError(f"{mode} manifest schema drift")
    if manifest.get("construction_columns") != list(
        spatial_split.ALLOWED_HORIZONTAL_COLUMNS
    ):
        raise ProtocolError(f"{mode} construction-column drift")
    if manifest.get("tvt_input_usage") != "missingness mask only":
        raise ProtocolError(f"{mode} TVT_input usage drift")
    if manifest.get("typewell_usage") != "raw-file SHA256 only; values are not parsed":
        raise ProtocolError(f"{mode} typewell construction usage drift")

    provenance = manifest.get("construction_provenance")
    if not isinstance(provenance, dict):
        raise ProtocolError(f"{mode} manifest lacks construction provenance")
    if provenance.get("source_file") != "research/spatial_split.py":
        raise ProtocolError(f"{mode} split source identity drift")
    if provenance.get("source_sha256") != group_gate.sha256_file(
        ROOT / "research/spatial_split.py"
    ):
        raise ProtocolError(f"{mode} split source hash drift")
    if provenance.get("packages") != spatial_split._construction_packages():
        raise ProtocolError(f"{mode} split package drift")

    parameters = manifest.get("parameters")
    if (
        not isinstance(parameters, dict)
        or int(parameters.get("n_folds", -1)) != N_FOLDS
    ):
        raise ProtocolError(f"{mode} split-count drift")
    if float(manifest.get("resampled_polyline_spacing_ft", np.nan)) != 100.0:
        raise ProtocolError(f"{mode} trajectory-resampling spacing drift")
    trajectory_isolation = parameters.get("trajectory_isolation")
    if mode == PRIMARY_MODE and trajectory_isolation is not True:
        raise ProtocolError(
            "region_out must retain resampled-point trajectory isolation"
        )
    if mode == SECONDARY_MODE and trajectory_isolation is not False:
        raise ProtocolError("pad_out must remain a non-isolated sensitivity split")
    if mode == PRIMARY_MODE and (
        float(parameters.get("centroid_embargo_ft", np.nan)) != 5_000.0
        or float(parameters.get("resampled_polyline_embargo_ft", np.nan)) != 1_500.0
    ):
        raise ProtocolError("region_out frozen embargo parameters drifted")
    if (
        mode == SECONDARY_MODE
        and float(parameters.get("centroid_component_radius_ft", np.nan)) != 1_500.0
    ):
        raise ProtocolError("pad_out frozen centroid-component radius drifted")

    wells = manifest.get("wells")
    folds = manifest.get("folds")
    if not isinstance(wells, list) or not wells:
        raise ProtocolError(f"{mode} manifest has no wells")
    if len(wells) != EXPECTED_ELIGIBLE_WELLS:
        raise ProtocolError(f"{mode} eligible-well population drifted")
    if not isinstance(folds, list) or len(folds) != N_FOLDS:
        raise ProtocolError(f"{mode} manifest must contain five folds")
    well_ids = [str(row.get("well_id", "")) for row in wells]
    if any(not wid for wid in well_ids) or len(well_ids) != len(set(well_ids)):
        raise ProtocolError(f"{mode} manifest has blank or duplicate well IDs")
    universe = set(well_ids)
    excluded = {str(value) for value in manifest.get("excluded_ids", [])}
    if excluded != set(group_gate.EXCLUDED_TEST_OVERLAP):
        raise ProtocolError(f"{mode} declared exclusion set drift")
    if universe & excluded:
        raise ProtocolError(f"{mode} manifest includes a declared excluded well")
    if not _is_sha256(manifest.get("dataset_sha256")):
        raise ProtocolError(f"{mode} manifest dataset fingerprint is invalid")

    equality_group = {
        str(row.get("well_id")): str(row.get("equality_group", "")) for row in wells
    }
    if any(not value for value in equality_group.values()):
        raise ProtocolError(f"{mode} manifest has a blank equality group")
    if len(set(equality_group.values())) != EXPECTED_EQUALITY_GROUPS:
        raise ProtocolError(f"{mode} exact equality-group population drifted")
    if int(manifest.get("diagnostics", {}).get("n_suffix_rows", -1)) != (
        EXPECTED_SUFFIX_ROWS
    ):
        raise ProtocolError(f"{mode} suffix-row population drifted")
    validation_fold = {
        str(row.get("well_id")): int(row.get("validation_fold", -1)) for row in wells
    }
    if any(value not in range(N_FOLDS) for value in validation_fold.values()):
        raise ProtocolError(f"{mode} manifest has an invalid validation fold")
    if mode == SECONDARY_MODE:
        pad_components = [str(row.get("pad_component", "")) for row in wells]
        if any(not value for value in pad_components):
            raise ProtocolError("pad_out manifest has a blank pad component")
        observed_components = len(set(pad_components))
        if observed_components != EXPECTED_PAD_COMPONENTS:
            raise ProtocolError(
                f"expected {EXPECTED_PAD_COMPONENTS} sealed pad components, "
                f"got {observed_components}"
            )
        if int(manifest.get("diagnostics", {}).get("pad_components", -1)) != (
            EXPECTED_PAD_COMPONENTS
        ):
            raise ProtocolError("pad_out component diagnostic drift")
        by_component: dict[str, set[int]] = {}
        for row in wells:
            by_component.setdefault(str(row["pad_component"]), set()).add(
                int(row["validation_fold"])
            )
        if any(
            len(folds_for_component) != 1
            for folds_for_component in by_component.values()
        ):
            raise ProtocolError("pad_out component crosses validation folds")

    observed_folds = []
    for row in folds:
        fold = int(row.get("fold", -1))
        observed_folds.append(fold)
        role_lists = {}
        for role in ("validation_ids", "training_ids", "embargo_ids"):
            values = [str(value) for value in row.get(role, [])]
            if len(values) != len(set(values)) or values != sorted(values):
                raise ProtocolError(f"{mode} fold {fold} has duplicate/unsorted {role}")
            role_lists[role] = set(values)
        validation = role_lists["validation_ids"]
        training = role_lists["training_ids"]
        embargo = role_lists["embargo_ids"]
        if validation & training or validation & embargo or training & embargo:
            raise ProtocolError(f"{mode} fold {fold} has overlapping roles")
        if validation | training | embargo != universe:
            raise ProtocolError(f"{mode} fold {fold} does not partition all wells")
        if {validation_fold[wid] for wid in validation} != {fold}:
            raise ProtocolError(f"{mode} fold {fold} validation assignment drift")
        for equality in set(equality_group.values()):
            members = {
                wid for wid, value in equality_group.items() if value == equality
            }
            roles = {
                "validation"
                if wid in validation
                else "training"
                if wid in training
                else "embargo"
                for wid in members
            }
            if len(roles) != 1:
                raise ProtocolError(
                    f"{mode} fold {fold} splits an exact trajectory/typewell group"
                )
        if mode == PRIMARY_MODE:
            centroid_buffer = float(parameters.get("centroid_embargo_ft", np.nan))
            polyline_buffer = float(
                parameters.get("resampled_polyline_embargo_ft", np.nan)
            )
            diagnostics = row.get("diagnostics", {})
            centroid_min = diagnostics.get("nearest_training_centroid_ft", {}).get(
                "min"
            )
            polyline_min = diagnostics.get(
                "nearest_training_resampled_polyline_ft", {}
            ).get("min")
            if (
                not np.isfinite(centroid_buffer)
                or not np.isfinite(polyline_buffer)
                or centroid_min is None
                or polyline_min is None
                or float(centroid_min) <= centroid_buffer
                or float(polyline_min) <= polyline_buffer
            ):
                raise ProtocolError(f"region_out fold {fold} violates its embargo")
        elif embargo:
            raise ProtocolError("pad_out cannot carry embargo wells")
    if sorted(observed_folds) != list(range(N_FOLDS)):
        raise ProtocolError(f"{mode} fold identities drifted")

    if inventory is None:
        return
    if set(inventory["well"].astype(str)) != universe:
        raise ProtocolError(f"{mode} manifest/data-inventory population mismatch")
    inventory_by_well = inventory.set_index("well", drop=False)
    for row in wells:
        wid = str(row["well_id"])
        data_row = inventory_by_well.loc[wid]
        comparisons = {
            "source_file": str(data_row["horizontal_file"]),
            "typewell_file": str(data_row["typewell_file"]),
            "n_rows": int(data_row["rows"]),
            "n_suffix_rows": int(data_row["suffix_rows"]),
            "typewell_sha256": str(data_row["typewell_sha256"]),
        }
        for key, expected in comparisons.items():
            actual = row.get(key)
            if isinstance(expected, int):
                actual = int(actual)
            else:
                actual = str(actual)
            if actual != expected:
                raise ProtocolError(f"{mode} manifest data drift for {wid}: {key}")

    # The split constructor preserves raw-file equality.  The model's inner
    # gate uses parsed TVT/GR equality, which is slightly broader in principle;
    # explicitly prove those exact profiles never cross any spatial role.
    parsed_group = dict(
        zip(
            inventory["well"].astype(str),
            inventory["typewell_profile_hash"].astype(str),
            strict=True,
        )
    )
    groups: dict[str, set[str]] = {}
    for wid, profile_hash in parsed_group.items():
        groups.setdefault(profile_hash, set()).add(wid)
    equality_to_profiles: dict[str, set[str]] = {}
    profile_to_equalities: dict[str, set[str]] = {}
    for wid in universe:
        equality_to_profiles.setdefault(equality_group[wid], set()).add(
            parsed_group[wid]
        )
        profile_to_equalities.setdefault(parsed_group[wid], set()).add(
            equality_group[wid]
        )
    if any(len(values) != 1 for values in equality_to_profiles.values()) or any(
        len(values) != 1 for values in profile_to_equalities.values()
    ):
        raise ProtocolError(
            f"{mode} sealed equality groups are not one-to-one with parsed profiles"
        )
    for fold_row in folds:
        fold = int(fold_row["fold"])
        validation = set(map(str, fold_row["validation_ids"]))
        training = set(map(str, fold_row["training_ids"]))
        for members in groups.values():
            roles = {
                "validation"
                if wid in validation
                else "training"
                if wid in training
                else "embargo"
                for wid in members
            }
            if len(roles) != 1:
                raise ProtocolError(
                    f"{mode} fold {fold} splits an exact parsed typewell profile"
                )


def _load_manifest(
    path: Path, mode: str, inventory: pd.DataFrame | None = None
) -> dict:
    if group_gate.sha256_file(path) != EXPECTED_MANIFEST_SHA256[mode]["byte"]:
        raise ProtocolError(f"{mode} is not the predeclared manifest byte artifact")
    manifest = _read_json(path, f"{mode} manifest")
    _validate_spatial_manifest(manifest, mode, inventory)
    return manifest


def _validate_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    missing = set(INVENTORY_COLUMNS) - set(inventory.columns)
    if missing:
        raise ProtocolError(f"data inventory is missing columns: {sorted(missing)}")
    inventory = inventory.loc[:, INVENTORY_COLUMNS].copy()
    if inventory.isna().any().any():
        raise ProtocolError("data inventory contains missing values")
    inventory["well"] = inventory["well"].astype(str)
    if inventory["well"].duplicated().any():
        raise ProtocolError("data inventory contains duplicate well IDs")
    if len(inventory) != EXPECTED_ELIGIBLE_WELLS:
        raise ProtocolError(
            f"expected {EXPECTED_ELIGIBLE_WELLS} eligible wells, got {len(inventory)}"
        )
    if inventory["typewell_profile_hash"].nunique() != EXPECTED_TYPEWELL_GROUPS:
        raise ProtocolError(
            "data inventory exact parsed typewell-profile population drifted"
        )
    if int(inventory["suffix_rows"].sum()) != EXPECTED_SUFFIX_ROWS:
        raise ProtocolError("data inventory suffix-row population drifted")
    for column in ("horizontal_sha256", "typewell_sha256", "typewell_profile_hash"):
        if not inventory[column].map(_is_sha256).all():
            raise ProtocolError(f"data inventory has an invalid SHA-256 in {column}")
    if set(inventory["well"]) & group_gate.EXCLUDED_TEST_OVERLAP:
        raise ProtocolError("excluded train/test-overlap ID entered data inventory")
    return inventory.sort_values("well").reset_index(drop=True)


def _validate_cross_manifest_population(manifests: Mapping[str, dict]) -> None:
    if set(manifests) != set(MODES):
        raise ProtocolError("both predeclared spatial manifests are required")
    populations = {
        mode: {str(row["well_id"]) for row in manifest["wells"]}
        for mode, manifest in manifests.items()
    }
    if populations[PRIMARY_MODE] != populations[SECONDARY_MODE]:
        raise ProtocolError("spatial manifests do not cover the same wells")
    if set(manifests[PRIMARY_MODE].get("excluded_ids", [])) != set(
        manifests[SECONDARY_MODE].get("excluded_ids", [])
    ):
        raise ProtocolError("spatial manifests do not share the exclusion set")
    if manifests[PRIMARY_MODE].get("dataset_sha256") != manifests[SECONDARY_MODE].get(
        "dataset_sha256"
    ):
        raise ProtocolError("spatial manifests do not share a dataset fingerprint")
    if manifests[PRIMARY_MODE].get("dataset_sha256") != EXPECTED_SPATIAL_DATASET_SHA256:
        raise ProtocolError("frozen spatial dataset fingerprint drifted")
    by_mode = {
        mode: {str(row["well_id"]): row for row in manifest["wells"]}
        for mode, manifest in manifests.items()
    }
    # Both split plans must bind the exact same inference-safe geometry and raw
    # typewell identities; only their spatial role assignment may differ.
    shared_fields = (
        "source_file",
        "typewell_file",
        "centroid_x",
        "centroid_y",
        "n_rows",
        "n_suffix_rows",
        "trajectory_sha256",
        "typewell_sha256",
        "equality_group",
    )
    for wid in sorted(populations[PRIMARY_MODE]):
        for field in shared_fields:
            if by_mode[PRIMARY_MODE][wid].get(field) != by_mode[SECONDARY_MODE][
                wid
            ].get(field):
                raise ProtocolError(
                    f"cross-mode manifest identity drift for {wid}: {field}"
                )


def freeze_protocol(
    data_dir: Path,
    manifest_dir: Path,
    protocol_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Seal both split inputs and a write-once inference-safe protocol."""

    data_dir = data_dir.resolve()
    manifest_dir = manifest_dir.resolve()
    protocol_path = protocol_path.resolve()
    inventory_path = _inventory_path(protocol_path)
    protocol_sidecar = group_gate._protocol_sidecar(protocol_path)
    inventory_sidecar = group_gate._protocol_sidecar(inventory_path)
    for path in (protocol_path, protocol_sidecar, inventory_path, inventory_sidecar):
        if path.exists():
            raise ProtocolError(
                f"freeze is write-once; artifact already exists: {path}"
            )

    source_hashes = _source_hashes()
    packages = _packages()
    fit_contract = _fit_contract()
    group_gate._assert_inference_safe_feature_surface()
    inventory = _validate_inventory(group_gate._base_manifest(data_dir))
    manifest_paths = {mode: (manifest_dir / f"{mode}.json").resolve() for mode in MODES}
    if len(set(manifest_paths.values())) != len(MODES):
        raise ProtocolError("spatial manifest paths alias one another")
    manifests = {
        mode: _load_manifest(path, mode, inventory)
        for mode, path in manifest_paths.items()
    }
    _validate_cross_manifest_population(manifests)

    inventory_bytes = inventory.to_csv(index=False, lineterminator="\n").encode("utf-8")
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    manifest_contract = {
        mode: {
            "file": path.name,
            "path": str(path),
            "byte_sha256": group_gate.sha256_file(path),
            "logical_manifest_sha256": manifests[mode]["manifest_sha256"],
            "schema_version": manifests[mode]["schema_version"],
            "dataset_sha256": manifests[mode]["dataset_sha256"],
            "construction_columns": manifests[mode]["construction_columns"],
            "construction_provenance": manifests[mode]["construction_provenance"],
            "resampled_polyline_spacing_ft": manifests[mode][
                "resampled_polyline_spacing_ft"
            ],
            "parameters": manifests[mode]["parameters"],
            "eligible_wells": len(manifests[mode]["wells"]),
            "equality_groups": len(
                {row["equality_group"] for row in manifests[mode]["wells"]}
            ),
            "suffix_rows": manifests[mode]["diagnostics"]["n_suffix_rows"],
            "excluded_ids": manifests[mode]["excluded_ids"],
            "role": "primary spatial falsification gate"
            if mode == PRIMARY_MODE
            else "secondary centroid-component sensitivity only",
        }
        for mode, path in manifest_paths.items()
    }
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_SPATIAL_SCORING_MEASURE_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "data": {
            "data_dir": str(data_dir),
            "eligible_wells": int(len(inventory)),
            "unique_typewell_groups": int(inventory["typewell_profile_hash"].nunique()),
            "suffix_rows": int(inventory["suffix_rows"].sum()),
            "excluded_test_overlap_ids": sorted(group_gate.EXCLUDED_TEST_OVERLAP),
            "inventory_file": inventory_path.name,
            "inventory_sha256": inventory_hash,
            "inventory_columns": list(INVENTORY_COLUMNS),
        },
        "spatial_manifests": manifest_contract,
        "spatial_run": {
            "modes": list(MODES),
            "folds_per_mode": N_FOLDS,
            "required_prediction_shards": len(MODES) * N_FOLDS,
            "single_process_estimated_wall_clock_hours": list(ESTIMATED_RUNTIME_HOURS),
        },
        "fit": fit_contract,
        "evaluation": _evaluation_protocol(),
        "source_sha256": source_hashes,
        "packages": packages,
        "notes": [
            "Both spatial modes are frozen and run; no mode or component selection is allowed.",
            "region_out is primary; pad_out is a secondary non-isolated sensitivity diagnostic.",
            "Region trajectory separation is measured on 100-MD-ft resampled points, not exact continuous segments.",
            "Pad cross-fold laterals may approach or cross and its observed distance may be zero.",
            "Region embargo wells are excluded, not added to the training population.",
            "Every learned quantity is refit on each retained spatial training fold.",
            "Freeze inventories inference columns and raw bytes without opening suffix TVT values.",
            "Run may open labels for retained training fits only; validation truth is aggregate-only.",
            "Spatial results remain MEASURE_ONLY and cannot promote production OPEN.",
        ],
    }
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    group_gate._atomic_write_bytes(inventory_path, inventory_bytes)
    group_gate._write_hash_sidecar(inventory_path)
    group_gate._atomic_write_bytes(protocol_path, protocol_bytes)
    group_gate._atomic_write_bytes(
        protocol_sidecar,
        f"{protocol_hash}  {protocol_path.name}\n".encode("ascii"),
    )
    return protocol_path, inventory_path, protocol_sidecar, inventory_sidecar


def _safe_manifest_path(contract: Mapping[str, Any], mode: str) -> Path:
    raw_path = str(contract.get("path", ""))
    path = Path(raw_path).resolve()
    if path.name != contract.get("file") or path.name != f"{mode}.json":
        raise ProtocolError(f"unsafe or unexpected {mode} manifest path")
    return path


def _audit_dataset(inventory: pd.DataFrame, data_dir: Path) -> None:
    for number, row in enumerate(inventory.itertuples(index=False), 1):
        horizontal = group_gate._safe_train_file(data_dir, str(row.horizontal_file))
        typewell = group_gate._safe_train_file(data_dir, str(row.typewell_file))
        if not horizontal.is_file() or not typewell.is_file():
            raise ProtocolError(f"frozen dataset file is missing for well {row.well}")
        if group_gate.sha256_file(horizontal) != str(row.horizontal_sha256):
            raise ProtocolError(f"horizontal dataset hash drift for well {row.well}")
        if group_gate.sha256_file(typewell) != str(row.typewell_sha256):
            raise ProtocolError(f"typewell dataset hash drift for well {row.well}")
        if number % 200 == 0:
            print(
                f"spatial audit inventory: {number}/{len(inventory)} wells", flush=True
            )


def audit_protocol(
    protocol_path: Path,
    verify_data: bool = True,
) -> SpatialAuditBundle:
    """Verify protocol, inventory, both manifests, source, packages, and data."""

    protocol_path = protocol_path.resolve()
    if group_gate._read_sidecar(protocol_path) != group_gate.sha256_file(protocol_path):
        raise ProtocolError("spatial protocol hash drift")
    protocol = _read_json(protocol_path, "spatial protocol")
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported spatial protocol version")
    if protocol.get("method") != METHOD or METHOD != group_gate.METHOD:
        raise ProtocolError("frozen spatial method identity changed")
    if protocol.get("fit") != _fit_contract():
        raise ProtocolError("frozen fold-local fit contract drift")
    if protocol.get("evaluation") != _evaluation_protocol():
        raise ProtocolError("frozen spatial evaluation contract drift")
    if protocol.get("source_sha256") != _source_hashes():
        raise ProtocolError("spatial executable source hash drift")
    if protocol.get("packages") != _packages():
        raise ProtocolError("spatial package-version drift")
    run_contract = protocol.get("spatial_run", {})
    if run_contract != {
        "modes": list(MODES),
        "folds_per_mode": N_FOLDS,
        "required_prediction_shards": len(MODES) * N_FOLDS,
        "single_process_estimated_wall_clock_hours": list(ESTIMATED_RUNTIME_HOURS),
    }:
        raise ProtocolError("spatial run contract drift")

    data = protocol.get("data", {})
    if data.get("eligible_wells") != EXPECTED_ELIGIBLE_WELLS:
        raise ProtocolError("frozen eligible-well population drifted")
    if data.get("unique_typewell_groups") != EXPECTED_TYPEWELL_GROUPS:
        raise ProtocolError("frozen exact-profile population drifted")
    if data.get("suffix_rows") != EXPECTED_SUFFIX_ROWS:
        raise ProtocolError("frozen suffix-row population drifted")
    inventory_name = str(data.get("inventory_file", ""))
    if Path(inventory_name).name != inventory_name:
        raise ProtocolError("unsafe data-inventory filename")
    inventory_path = protocol_path.with_name(inventory_name)
    if group_gate._read_sidecar(inventory_path) != group_gate.sha256_file(
        inventory_path
    ):
        raise ProtocolError("data-inventory sidecar/hash drift")
    if group_gate.sha256_file(inventory_path) != data.get("inventory_sha256"):
        raise ProtocolError("data-inventory protocol hash drift")
    inventory = _validate_inventory(pd.read_csv(inventory_path))
    if data.get("inventory_columns") != list(INVENTORY_COLUMNS):
        raise ProtocolError("data-inventory schema contract drift")

    manifest_contract = protocol.get("spatial_manifests")
    if not isinstance(manifest_contract, dict) or set(manifest_contract) != set(MODES):
        raise ProtocolError("protocol does not bind both spatial manifests")
    manifests = {}
    manifest_paths = {}
    for mode in MODES:
        contract = manifest_contract[mode]
        if not isinstance(contract, dict):
            raise ProtocolError(f"invalid {mode} manifest contract")
        expected_role = (
            "primary spatial falsification gate"
            if mode == PRIMARY_MODE
            else "secondary centroid-component sensitivity only"
        )
        if contract.get("role") != expected_role:
            raise ProtocolError(f"{mode} manifest role drift")
        path = _safe_manifest_path(contract, mode)
        if group_gate.sha256_file(path) != contract.get("byte_sha256"):
            raise ProtocolError(f"{mode} manifest byte hash drift")
        manifest = _load_manifest(path, mode, inventory)
        if manifest.get("manifest_sha256") != contract.get("logical_manifest_sha256"):
            raise ProtocolError(f"{mode} logical manifest identity drift")
        expected_binding = {
            "schema_version": manifest["schema_version"],
            "dataset_sha256": manifest["dataset_sha256"],
            "construction_columns": manifest["construction_columns"],
            "construction_provenance": manifest["construction_provenance"],
            "resampled_polyline_spacing_ft": manifest["resampled_polyline_spacing_ft"],
            "parameters": manifest["parameters"],
            "eligible_wells": len(manifest["wells"]),
            "equality_groups": len(
                {row["equality_group"] for row in manifest["wells"]}
            ),
            "suffix_rows": manifest["diagnostics"]["n_suffix_rows"],
            "excluded_ids": manifest["excluded_ids"],
        }
        for key, expected in expected_binding.items():
            if contract.get(key) != expected:
                raise ProtocolError(f"{mode} manifest protocol binding drift: {key}")
        manifests[mode] = manifest
        manifest_paths[mode] = path
    _validate_cross_manifest_population(manifests)

    data_dir = Path(str(data.get("data_dir", ""))).resolve()
    if verify_data:
        _audit_dataset(inventory, data_dir)
    return SpatialAuditBundle(
        protocol=protocol,
        inventory=inventory,
        manifests=manifests,
        protocol_sha256=group_gate.sha256_file(protocol_path),
        protocol_path=protocol_path,
        inventory_path=inventory_path,
        manifest_paths=manifest_paths,
    )


def _fold_roles(
    audit: SpatialAuditBundle, mode: str, fold: int
) -> tuple[list[str], list[str], list[str]]:
    row = _manifest_fold(audit.manifests[mode], fold)
    return (
        [str(value) for value in row["training_ids"]],
        [str(value) for value in row["validation_ids"]],
        [str(value) for value in row["embargo_ids"]],
    )


def _spatial_cluster_by_well(audit: SpatialAuditBundle, mode: str) -> dict[str, str]:
    if mode == PRIMARY_MODE:
        return {
            str(row["well_id"]): f"region_fold_{int(row['validation_fold'])}"
            for row in audit.manifests[mode]["wells"]
        }
    return {
        str(row["well_id"]): str(row["pad_component"])
        for row in audit.manifests[mode]["wells"]
    }


def _fit_candidate_fold(
    train_files: Sequence[str],
    validation_files: Sequence[str],
    group_by_well: dict[str, str],
) -> tuple[
    list[group_gate.PredictionRecord],
    dict,
    dict[str, dict],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Refit the identical v2 candidate using retained spatial training only."""

    static_matrix = group_gate._build_static_training_matrix(train_files)
    train_records, validation_records, base_meta = group_gate._fit_base_fold(
        train_files,
        validation_files,
        group_by_well,
        static_matrix,
    )
    if any(np.isfinite(record.truth).any() for record in validation_records):
        raise ProtocolError("spatial validation record unexpectedly exposed truth")
    train_tw, train_ordered_raw, _ = group_gate._evidence_for_records(train_records)
    validation_tw, validation_ordered_raw, diagnostics = (
        group_gate._evidence_for_records(validation_records)
    )
    if not (
        len(train_records) == len(train_tw) == len(train_ordered_raw)
        and len(validation_records) == len(validation_tw) == len(validation_ordered_raw)
    ):
        raise ProtocolError("spatial evidence output length mismatch")
    train_target = np.array([record.oracle_shift for record in train_records])
    train_tw_scalar, validation_tw_scalar, typewell_shrink = (
        group_gate._scalar_correction(train_tw, train_target, validation_tw)
    )
    ordered_shrink = group_gate._calibrate_vector_shrink(
        train_records, train_ordered_raw
    )
    train_ordered = [ordered_shrink * value for value in train_ordered_raw]
    validation_ordered = [ordered_shrink * value for value in validation_ordered_raw]
    train_tw_arrays = [
        np.full(len(record.idx), correction)
        for record, correction in zip(train_records, train_tw_scalar, strict=True)
    ]
    validation_tw_arrays = [
        np.full(len(record.idx), correction)
        for record, correction in zip(
            validation_records, validation_tw_scalar, strict=True
        )
    ]
    joint = group_gate._fit_joint_correction(
        train_records, train_tw_arrays, train_ordered
    )
    joint_corrections = [
        np.clip(joint[0] * first + joint[1] * second, -25.0, 25.0)
        for first, second in zip(validation_tw_arrays, validation_ordered, strict=True)
    ]
    learned = {
        **base_meta,
        "typewell_shrink": float(typewell_shrink),
        "ordered_shrink": float(ordered_shrink),
        "joint_coefficients": [float(joint[0]), float(joint[1])],
    }
    return (
        validation_records,
        learned,
        diagnostics,
        validation_tw_arrays,
        validation_ordered,
        joint_corrections,
    )


def _expected_validation_wells(
    audit: SpatialAuditBundle, mode: str, fold: int
) -> set[str]:
    return set(_fold_roles(audit, mode, fold)[1])


def _contains_forbidden_scoring_field(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if "truth" in normalized or normalized.endswith(("_sse", "_rmse")):
                return True
            if _contains_forbidden_scoring_field(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_scoring_field(item) for item in value)
    return False


def _validate_shard(
    shard: dict,
    audit: SpatialAuditBundle,
    mode: str,
    fold: int,
    shard_path: Path,
) -> dict[str, np.ndarray]:
    if shard.get("status") != ("MEASURE_ONLY_SPATIAL_PREDICTIONS_SEALED_TRUTH_UNREAD"):
        raise ProtocolError("spatial shard metric-silent status drift")
    if _contains_forbidden_scoring_field(shard):
        raise ProtocolError("spatial shard contains truth or scoring fields")
    if group_gate._read_sidecar(shard_path) != group_gate.sha256_file(shard_path):
        raise ProtocolError("spatial shard metadata hash drift")
    if shard.get("protocol_sha256") != audit.protocol_sha256:
        raise ProtocolError("spatial shard protocol hash mismatch")
    manifest_contract = audit.protocol["spatial_manifests"][mode]
    if (
        shard.get("manifest_byte_sha256") != manifest_contract["byte_sha256"]
        or shard.get("manifest_logical_sha256")
        != manifest_contract["logical_manifest_sha256"]
    ):
        raise ProtocolError("spatial shard manifest identity mismatch")
    expected_all_manifests = {
        manifest_mode: {
            "byte_sha256": audit.protocol["spatial_manifests"][manifest_mode][
                "byte_sha256"
            ],
            "logical_manifest_sha256": audit.protocol["spatial_manifests"][
                manifest_mode
            ]["logical_manifest_sha256"],
        }
        for manifest_mode in MODES
    }
    if shard.get("all_spatial_manifest_sha256") != expected_all_manifests:
        raise ProtocolError("spatial shard cross-mode manifest binding drift")
    if shard.get("method") != METHOD:
        raise ProtocolError("spatial shard method drift")
    if shard.get("mode") != mode or int(shard.get("fold", -1)) != fold:
        raise ProtocolError("spatial shard identity mismatch")
    if shard.get("mode_role") != audit.protocol["spatial_manifests"][mode]["role"]:
        raise ProtocolError("spatial shard mode role drift")

    training_ids, validation_ids, embargo_ids = _fold_roles(audit, mode, fold)
    membership = shard.get("frozen_membership_sha256", {})
    expected_membership = {
        "training_ids": _id_digest(training_ids),
        "validation_ids": _id_digest(validation_ids),
        "embargo_ids": _id_digest(embargo_ids),
    }
    if membership != expected_membership:
        raise ProtocolError("spatial shard membership digest drift")
    if int(shard.get("train_wells", -1)) != len(training_ids):
        raise ProtocolError("spatial shard training-well count drift")
    if int(shard.get("validation_well_count", -1)) != len(validation_ids):
        raise ProtocolError("spatial shard validation-well count drift")
    if int(shard.get("embargo_well_count", -1)) != len(embargo_ids):
        raise ProtocolError("spatial shard embargo-well count drift")
    if mode == PRIMARY_MODE and set(training_ids) & set(embargo_ids):
        raise ProtocolError("region embargo well was repurposed for training")

    equality_by_well = {
        str(item["well_id"]): str(item["equality_group"])
        for item in audit.manifests[mode]["wells"]
    }
    expected_train_groups = {equality_by_well[wid] for wid in training_ids}
    expected_validation_groups = {equality_by_well[wid] for wid in validation_ids}
    if int(shard.get("train_typewell_groups", -1)) != len(expected_train_groups):
        raise ProtocolError("spatial shard training-group count drift")
    if int(shard.get("validation_typewell_groups", -1)) != len(
        expected_validation_groups
    ):
        raise ProtocolError("spatial shard validation-group count drift")
    learned = shard.get("learned_from_manifest_retained_training_only")
    if not isinstance(learned, dict):
        raise ProtocolError("spatial shard lacks fold-local learned parameters")
    required_learned = {
        "base_shrink",
        "inner_oof_point_r2",
        "training_points",
        "training_wells",
        "training_typewell_groups",
        "typewell_shrink",
        "ordered_shrink",
        "joint_coefficients",
    }
    if set(learned) != required_learned:
        raise ProtocolError("spatial shard learned-parameter schema drift")
    scalar_names = (
        "base_shrink",
        "inner_oof_point_r2",
        "typewell_shrink",
        "ordered_shrink",
    )
    if any(not np.isfinite(float(learned[name])) for name in scalar_names):
        raise ProtocolError("spatial shard learned parameter is non-finite")
    if not (0.0 <= float(learned["typewell_shrink"]) <= 1.5) or not (
        0.0 <= float(learned["ordered_shrink"]) <= 1.5
    ):
        raise ProtocolError("spatial shard shrink is outside frozen bounds")
    joint = np.asarray(learned["joint_coefficients"], dtype=float)
    if (
        joint.shape != (2,)
        or not np.isfinite(joint).all()
        or np.any(joint < -1.0)
        or np.any(joint > 2.0)
    ):
        raise ProtocolError("spatial shard joint coefficient is invalid")
    if int(learned["training_points"]) <= 0:
        raise ProtocolError("spatial shard has no fold-local training points")
    if int(learned["training_wells"]) != len(training_ids):
        raise ProtocolError("spatial shard learned training-well count drift")
    if int(learned["training_typewell_groups"]) != len(expected_train_groups):
        raise ProtocolError("spatial shard learned training-group count drift")
    if (
        not np.isfinite(float(shard.get("runtime_seconds", np.nan)))
        or float(shard["runtime_seconds"]) < 0.0
    ):
        raise ProtocolError("spatial shard runtime is invalid")

    inventory_by_well = audit.inventory.set_index("well", drop=False)
    cluster_by_well = _spatial_cluster_by_well(audit, mode)
    rows = shard.get("test_wells")
    if not isinstance(rows, list):
        raise ProtocolError("spatial shard has no validation-well rows")
    wells = [str(row.get("well", "")) for row in rows]
    if len(wells) != len(set(wells)) or set(wells) != set(validation_ids):
        raise ProtocolError("spatial shard validation membership drift")
    diagnostics = shard.get("validation_diagnostics")
    if not isinstance(diagnostics, dict) or set(map(str, diagnostics)) != set(
        validation_ids
    ):
        raise ProtocolError("spatial shard validation diagnostics membership drift")
    if shard.get("prediction_channels") != list(PREDICTION_ARRAYS[2:]):
        raise ProtocolError("spatial shard prediction-channel contract drift")
    for metadata_index, row in enumerate(rows):
        wid = str(row.get("well", ""))
        if int(row.get("well_index", -1)) != metadata_index:
            raise ProtocolError("spatial shard well-index ordering drift")
        if str(row.get("typewell_profile_hash", "")) != str(
            inventory_by_well.loc[wid, "typewell_profile_hash"]
        ):
            raise ProtocolError("spatial shard exact-profile identity drift")
        if str(row.get("spatial_cluster_id", "")) != cluster_by_well[wid]:
            raise ProtocolError("spatial shard cluster identity drift")
        if str(row.get("inner_equality_group", "")) != equality_by_well[wid]:
            raise ProtocolError("spatial shard inner-group identity drift")
        if int(row.get("n_rows", 0)) != int(inventory_by_well.loc[wid, "suffix_rows"]):
            raise ProtocolError("spatial shard suffix-row count drift")

    prediction_name = str(shard.get("prediction_file", ""))
    if Path(prediction_name).name != prediction_name:
        raise ProtocolError("unsafe prediction filename in spatial shard")
    prediction_path = shard_path.with_name(prediction_name)
    if prediction_path.resolve() != _prediction_path(shard_path).resolve():
        raise ProtocolError("spatial prediction filename drift")
    if not prediction_path.is_file():
        raise ProtocolError(
            f"spatial prediction artifact is missing: {prediction_path}"
        )
    if group_gate.sha256_file(prediction_path) != shard.get("prediction_sha256"):
        raise ProtocolError("spatial prediction NPZ byte hash drift")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_ARRAYS):
            raise ProtocolError("spatial prediction NPZ schema drift")
        arrays = {name: archive[name].copy() for name in PREDICTION_ARRAYS}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or lengths.pop() != int(shard.get("prediction_rows", -1)):
        raise ProtocolError("spatial prediction arrays have inconsistent lengths")
    if int(shard.get("prediction_rows", -1)) != sum(int(row["n_rows"]) for row in rows):
        raise ProtocolError("spatial shard prediction-row metadata drift")
    if group_gate._logical_array_hash(arrays) != shard.get("prediction_logical_sha256"):
        raise ProtocolError("spatial prediction logical-array hash drift")
    if (
        arrays["well_index"].dtype.kind not in "iu"
        or arrays["row_index"].dtype.kind not in "iu"
    ):
        raise ProtocolError("spatial prediction identity arrays must be integers")
    if np.any(arrays["well_index"] < 0) or np.any(arrays["well_index"] >= len(rows)):
        raise ProtocolError("spatial prediction well index is out of range")
    for name in PREDICTION_ARRAYS[2:]:
        if not np.isfinite(arrays[name]).all():
            raise ProtocolError(f"spatial prediction array is non-finite: {name}")
    for well_index, row in enumerate(rows):
        selected = arrays["well_index"] == well_index
        if int(selected.sum()) != int(row["n_rows"]):
            raise ProtocolError("spatial prediction row count differs from metadata")
        indices = arrays["row_index"][selected]
        if len(indices) != len(np.unique(indices)) or np.any(indices < 0):
            raise ProtocolError("spatial prediction indices are duplicated or negative")
    return arrays


def _run_one_fold(
    audit: SpatialAuditBundle,
    mode: str,
    fold: int,
    output_dir: Path,
    resume: bool,
) -> Path:
    shard_path = output_dir / _fold_shard_name(mode, fold)
    prediction_path = _prediction_path(shard_path)
    sidecar_path = group_gate._protocol_sidecar(shard_path)
    if shard_path.exists():
        if not resume:
            raise ProtocolError(f"spatial shard already exists: {shard_path}")
        shard = _read_json(shard_path, "spatial shard")
        _validate_shard(shard, audit, mode, fold, shard_path)
        return shard_path
    if prediction_path.exists() or sidecar_path.exists():
        raise ProtocolError(
            f"incomplete/stale spatial fold artifacts exist: {shard_path}"
        )

    started = time.time()
    training_ids, validation_ids, embargo_ids = _fold_roles(audit, mode, fold)
    if set(training_ids) & set(validation_ids) or set(training_ids) & set(embargo_ids):
        raise ProtocolError("spatial training role overlaps validation or embargo")
    if mode == PRIMARY_MODE:
        frozen_fold = _manifest_fold(audit.manifests[mode], fold)
        if training_ids != [str(value) for value in frozen_fold["training_ids"]]:
            raise ProtocolError("region training IDs differ from frozen retained set")

    data_dir = Path(audit.protocol["data"]["data_dir"])
    inventory_by_well = audit.inventory.set_index("well", drop=False)
    cluster_by_well = _spatial_cluster_by_well(audit, mode)
    train_files = [
        str(
            group_gate._safe_train_file(
                data_dir, str(inventory_by_well.loc[wid, "horizontal_file"])
            )
        )
        for wid in training_ids
    ]
    validation_files = [
        str(
            group_gate._safe_train_file(
                data_dir, str(inventory_by_well.loc[wid, "horizontal_file"])
            )
        )
        for wid in validation_ids
    ]
    profile_by_well = dict(
        zip(
            audit.inventory["well"].astype(str),
            audit.inventory["typewell_profile_hash"].astype(str),
            strict=True,
        )
    )
    full_group_by_well = {
        str(row["well_id"]): str(row["equality_group"])
        for row in audit.manifests[mode]["wells"]
    }
    training_group_by_well = {wid: full_group_by_well[wid] for wid in training_ids}
    if set(training_group_by_well) & set(embargo_ids):
        raise ProtocolError("embargo identity entered fold-local fit grouping state")
    train_groups = {full_group_by_well[wid] for wid in training_ids}
    validation_groups = {full_group_by_well[wid] for wid in validation_ids}
    if train_groups & validation_groups:
        raise ProtocolError("sealed equality group crossed spatial boundary")

    (
        records,
        learned,
        diagnostics,
        typewell_corrections,
        ordered_corrections,
        joint_corrections,
    ) = _fit_candidate_fold(train_files, validation_files, training_group_by_well)
    if len(records) != len(validation_ids):
        raise ProtocolError("candidate did not predict every spatial validation well")

    well_indices = []
    row_indices = []
    base_predictions = []
    typewell_predictions = []
    ordered_predictions = []
    joint_predictions = []
    test_wells = []
    for well_index, (record, typewell, ordered, joint) in enumerate(
        zip(
            records,
            typewell_corrections,
            ordered_corrections,
            joint_corrections,
            strict=True,
        )
    ):
        n_rows = len(record.idx)
        if not (
            len(record.prediction)
            == len(typewell)
            == len(ordered)
            == len(joint)
            == n_rows
        ):
            raise ProtocolError(f"spatial prediction length mismatch for {record.well}")
        if len(np.unique(record.idx)) != n_rows or np.any(record.idx < 0):
            raise ProtocolError(f"invalid suffix row indices for {record.well}")
        well_indices.append(np.full(n_rows, well_index, dtype=np.int32))
        row_indices.append(np.asarray(record.idx, dtype=np.int32))
        base_predictions.append(np.asarray(record.prediction, dtype=np.float64))
        typewell_predictions.append(
            np.asarray(record.prediction + typewell, dtype=np.float64)
        )
        ordered_predictions.append(
            np.asarray(record.prediction + ordered, dtype=np.float64)
        )
        joint_predictions.append(
            np.asarray(record.prediction + joint, dtype=np.float64)
        )
        test_wells.append(
            {
                "well": record.well,
                "well_index": well_index,
                "typewell_profile_hash": profile_by_well[record.well],
                "inner_equality_group": full_group_by_well[record.well],
                "spatial_cluster_id": cluster_by_well[record.well],
                "n_rows": n_rows,
            }
        )
    if {row["well"] for row in test_wells} != set(validation_ids):
        raise ProtocolError("sealed predictions do not cover frozen validation IDs")
    arrays = {
        "well_index": np.concatenate(well_indices),
        "row_index": np.concatenate(row_indices),
        "base_prediction": np.concatenate(base_predictions),
        "typewell_prediction": np.concatenate(typewell_predictions),
        "ordered_prediction": np.concatenate(ordered_predictions),
        "joint_prediction": np.concatenate(joint_predictions),
    }
    if any(not np.isfinite(arrays[name]).all() for name in PREDICTION_ARRAYS[2:]):
        raise ProtocolError("non-finite value entered sealed spatial predictions")
    group_gate._atomic_write_npz(prediction_path, arrays)
    shard = {
        "status": "MEASURE_ONLY_SPATIAL_PREDICTIONS_SEALED_TRUTH_UNREAD",
        "protocol_sha256": audit.protocol_sha256,
        "manifest_byte_sha256": audit.protocol["spatial_manifests"][mode][
            "byte_sha256"
        ],
        "manifest_logical_sha256": audit.protocol["spatial_manifests"][mode][
            "logical_manifest_sha256"
        ],
        "all_spatial_manifest_sha256": {
            manifest_mode: {
                "byte_sha256": audit.protocol["spatial_manifests"][manifest_mode][
                    "byte_sha256"
                ],
                "logical_manifest_sha256": audit.protocol["spatial_manifests"][
                    manifest_mode
                ]["logical_manifest_sha256"],
            }
            for manifest_mode in MODES
        },
        "method": METHOD,
        "mode": mode,
        "mode_role": audit.protocol["spatial_manifests"][mode]["role"],
        "fold": fold,
        "train_wells": len(training_ids),
        "validation_well_count": len(validation_ids),
        "embargo_well_count": len(embargo_ids),
        "frozen_membership_sha256": {
            "training_ids": _id_digest(training_ids),
            "validation_ids": _id_digest(validation_ids),
            "embargo_ids": _id_digest(embargo_ids),
        },
        "train_typewell_groups": len(train_groups),
        "validation_typewell_groups": len(validation_groups),
        "learned_from_manifest_retained_training_only": learned,
        "validation_diagnostics": diagnostics,
        "prediction_file": prediction_path.name,
        "prediction_sha256": group_gate.sha256_file(prediction_path),
        "prediction_logical_sha256": group_gate._logical_array_hash(arrays),
        "prediction_rows": int(len(arrays["row_index"])),
        "prediction_channels": list(PREDICTION_ARRAYS[2:]),
        "test_wells": test_wells,
        "runtime_seconds": float(time.time() - started),
    }
    group_gate._atomic_write_json(shard_path, shard)
    group_gate._write_hash_sidecar(shard_path)
    _validate_shard(shard, audit, mode, fold, shard_path)
    return shard_path


def _all_folds() -> list[tuple[str, int]]:
    return [(mode, fold) for mode in MODES for fold in range(N_FOLDS)]


def run_folds(
    protocol_path: Path,
    output_dir: Path | None,
    folds: Sequence[tuple[str, int]],
    resume: bool,
) -> list[Path]:
    """Produce metric-silent spatial shards after a full frozen audit."""

    audit = audit_protocol(protocol_path, verify_data=True)
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    normalized = []
    for mode, fold in folds:
        if mode not in MODES or fold not in range(N_FOLDS):
            raise ProtocolError(f"invalid spatial fold identity ({mode}, {fold})")
        if (mode, fold) not in normalized:
            normalized.append((mode, fold))
    completed = []
    pending = []
    for mode, fold in normalized:
        path = output_dir / _fold_shard_name(mode, fold)
        if path.exists() and resume:
            shard = _read_json(path, "spatial shard")
            _validate_shard(shard, audit, mode, fold, path)
            completed.append(path)
        else:
            pending.append((mode, fold))
    if not pending:
        print(
            "all requested spatial shards pass audit; metrics remain withheld",
            flush=True,
        )
        return completed
    for mode, fold in pending:
        print(
            f"running frozen {mode} fold {fold + 1}/{N_FOLDS}; "
            "interim metrics withheld",
            flush=True,
        )
        completed.append(_run_one_fold(audit, mode, fold, output_dir, resume))
        audit_protocol(protocol_path, verify_data=True)
    print(
        f"completed {len(pending)} spatial shard(s); metrics remain withheld",
        flush=True,
    )
    return completed


def _score_sealed_predictions(
    audit: SpatialAuditBundle,
    shard: dict,
    arrays: dict[str, np.ndarray],
    mode: str,
    fold: int,
) -> list[dict]:
    """First and only phase that opens spatial-validation suffix truth."""

    _, validation_ids, _ = _fold_roles(audit, mode, fold)
    inventory_by_well = audit.inventory.set_index("well", drop=False)
    data_dir = Path(audit.protocol["data"]["data_dir"])
    rows = []
    for well_index, metadata in enumerate(shard["test_wells"]):
        wid = str(metadata["well"])
        if wid not in validation_ids or int(metadata["well_index"]) != well_index:
            raise ProtocolError("spatial scoring metadata identity drift")
        horizontal = str(
            group_gate._safe_train_file(
                data_dir, str(inventory_by_well.loc[wid, "horizontal_file"])
            )
        )
        well = group_gate.load_well(horizontal)
        if well is None or well["truth"] is None:
            raise ProtocolError(f"aggregate could not load validation truth for {wid}")
        selected = arrays["well_index"] == well_index
        indices = arrays["row_index"][selected].astype(np.int64, copy=False)
        expected_indices = np.flatnonzero(well["tail"])
        if not np.array_equal(indices, expected_indices):
            raise ProtocolError(f"sealed prediction rows do not equal suffix for {wid}")
        truth = np.asarray(well["truth"][indices], dtype=float)
        if not np.isfinite(truth).all():
            raise ProtocolError(f"validation truth is non-finite for {wid}")
        prediction = {
            name: np.asarray(arrays[name][selected], dtype=float)
            for name in PREDICTION_ARRAYS[2:]
        }
        if any(len(value) != len(truth) for value in prediction.values()):
            raise ProtocolError(f"prediction/truth length mismatch for {wid}")
        rows.append(
            {
                "mode": mode,
                "fold": fold,
                "well": wid,
                "typewell_profile_hash": str(metadata["typewell_profile_hash"]),
                "spatial_cluster_id": str(metadata["spatial_cluster_id"]),
                "n_rows": int(len(truth)),
                "base_sse": float(np.sum((prediction["base_prediction"] - truth) ** 2)),
                "typewell_sse": float(
                    np.sum((prediction["typewell_prediction"] - truth) ** 2)
                ),
                "ordered_sse": float(
                    np.sum((prediction["ordered_prediction"] - truth) ** 2)
                ),
                "joint_sse": float(
                    np.sum((prediction["joint_prediction"] - truth) ** 2)
                ),
            }
        )
    return rows


def _summary(rows: pd.DataFrame, candidate_sse_column: str = "joint_sse") -> dict:
    required = {"well", "n_rows", "base_sse", candidate_sse_column}
    missing = required - set(rows.columns)
    if missing:
        raise ProtocolError(f"spatial summary is missing columns: {sorted(missing)}")
    if rows["well"].duplicated().any():
        raise ProtocolError("spatial mode summary contains duplicate validation wells")
    count = rows["n_rows"].to_numpy(dtype=float)
    base_sse = rows["base_sse"].to_numpy(dtype=float)
    candidate_sse = rows[candidate_sse_column].to_numpy(dtype=float)
    if (
        not np.isfinite(count).all()
        or not np.isfinite(base_sse).all()
        or not np.isfinite(candidate_sse).all()
        or np.any(count <= 0)
        or np.any(base_sse < 0)
        or np.any(candidate_sse < 0)
    ):
        raise ProtocolError("invalid contribution entered spatial summary")
    base_well = np.sqrt(base_sse / count)
    candidate_well = np.sqrt(candidate_sse / count)
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
        "n_unique_wells": int(len(rows)),
        "n_rows": int(count.sum()),
    }


def _fold_pooled_gains(
    rows: pd.DataFrame,
    candidate_sse_column: str = "joint_sse",
) -> dict[str, float]:
    folds = sorted(rows["fold"].astype(int).unique().tolist())
    if folds != list(range(N_FOLDS)):
        raise ProtocolError(f"spatial scoring expected folds 0..{N_FOLDS - 1}")
    gains = {}
    for fold in folds:
        frame = rows.loc[rows["fold"].astype(int) == fold]
        count = float(frame["n_rows"].sum())
        gains[str(fold)] = float(
            np.sqrt(float(frame["base_sse"].sum()) / count)
            - np.sqrt(float(frame[candidate_sse_column].sum()) / count)
        )
    return gains


def _spatial_cluster_bootstrap(rows: pd.DataFrame, mode: str) -> dict:
    """Couple all channels under the predeclared spatial resampling unit."""

    required = {
        "spatial_cluster_id",
        "n_rows",
        "base_sse",
        "typewell_sse",
        "ordered_sse",
        "joint_sse",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ProtocolError(f"spatial bootstrap is missing columns: {sorted(missing)}")
    grouped = (
        rows.groupby("spatial_cluster_id", sort=True)
        .agg(
            n_rows=("n_rows", "sum"),
            base_sse=("base_sse", "sum"),
            typewell_sse=("typewell_sse", "sum"),
            ordered_sse=("ordered_sse", "sum"),
            joint_sse=("joint_sse", "sum"),
        )
        .reset_index()
    )
    expected_clusters = N_FOLDS if mode == PRIMARY_MODE else EXPECTED_PAD_COMPONENTS
    if len(grouped) != expected_clusters:
        raise ProtocolError(
            f"{mode} bootstrap expected {expected_clusters} sealed clusters, "
            f"got {len(grouped)}"
        )
    contribution = {
        name: grouped[name].to_numpy(dtype=float)
        for name in (
            "n_rows",
            "base_sse",
            "typewell_sse",
            "ordered_sse",
            "joint_sse",
        )
    }
    if (
        not all(np.isfinite(value).all() for value in contribution.values())
        or np.any(contribution["n_rows"] <= 0)
        or any(
            np.any(contribution[name] < 0)
            for name in ("base_sse", "typewell_sse", "ordered_sse", "joint_sse")
        )
    ):
        raise ProtocolError("invalid contribution entered spatial bootstrap")

    if mode == PRIMARY_MODE:
        # Exactly enumerate all ordered samples of five sealed regions drawn
        # five times with replacement: 5^5 = 3,125 coupled resamples.
        samples = np.indices((N_FOLDS,) * N_FOLDS).reshape(N_FOLDS, -1).T
        method = "exhaustive ordered resampling with replacement"
        seed: int | None = None
    elif mode == SECONDARY_MODE:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = rng.integers(
            0,
            expected_clusters,
            size=(BOOTSTRAP_DRAWS, expected_clusters),
        )
        method = "seeded cluster bootstrap with replacement"
        seed = BOOTSTRAP_SEED
    else:
        raise ProtocolError(f"unsupported bootstrap mode: {mode}")

    denominator = contribution["n_rows"][samples].sum(axis=1)
    rmse_draws = {
        channel: np.sqrt(
            contribution[f"{channel}_sse"][samples].sum(axis=1) / denominator
        )
        for channel in ("base", "typewell", "ordered", "joint")
    }
    gain_draws = {
        channel: rmse_draws["base"] - rmse_draws[channel]
        for channel in ("typewell", "ordered", "joint")
    }
    best_component_draws = np.minimum(rmse_draws["typewell"], rmse_draws["ordered"])
    joint_vs_best_draws = best_component_draws - rmse_draws["joint"]

    total_count = float(contribution["n_rows"].sum())
    observed_rmse = {
        channel: float(np.sqrt(contribution[f"{channel}_sse"].sum() / total_count))
        for channel in ("base", "typewell", "ordered", "joint")
    }
    observed_gain = {
        channel: observed_rmse["base"] - observed_rmse[channel]
        for channel in ("typewell", "ordered", "joint")
    }

    def distribution(values: np.ndarray, observed: float) -> dict:
        low, high = np.quantile(values, [0.025, 0.975])
        return {
            "observed_ft": float(observed),
            "ci95_low_ft": float(low),
            "ci95_high_ft": float(high),
            "probability_positive": float(np.mean(values > 0.0)),
        }

    return {
        "unit": "sealed validation region fold"
        if mode == PRIMARY_MODE
        else "sealed pad component",
        "method": method,
        "draws": int(len(samples)),
        "seed": seed,
        "n_clusters": int(len(grouped)),
        "same_cluster_multiplicities_applied_to_all_channels": True,
        "pooled_rmse_by_channel_ft": observed_rmse,
        "gain_vs_base_by_channel": {
            channel: distribution(gain_draws[channel], observed_gain[channel])
            for channel in ("typewell", "ordered", "joint")
        },
        "joint_vs_resample_best_component": distribution(
            joint_vs_best_draws,
            min(observed_rmse["typewell"], observed_rmse["ordered"])
            - observed_rmse["joint"],
        ),
    }


def _top_positive_sse_removal(
    rows: pd.DataFrame,
    candidate_sse_column: str = "joint_sse",
) -> dict:
    if rows["well"].duplicated().any():
        raise ProtocolError("spatial influence audit contains duplicate wells")
    benefits = rows.assign(_benefit=rows["base_sse"] - rows[candidate_sse_column])
    positive = benefits.loc[benefits["_benefit"] > 0].sort_values(
        ["_benefit", "well"], ascending=[False, True], kind="mergesort"
    )
    remove = positive.head(TOP_POSITIVE_SSE_REMOVAL_WELLS)["well"].astype(str).tolist()
    remaining = rows.loc[~rows["well"].astype(str).isin(remove)]
    if remaining.empty:
        raise ProtocolError("spatial influence removal exhausted the population")
    summary = _summary(remaining, candidate_sse_column)
    gain = summary["pooled_row_rmse_gain_ft"]
    return {
        "ranking": (
            f"descending positive base_sse minus {candidate_sse_column}; "
            "well ID tie-break"
        ),
        "requested_remove": TOP_POSITIVE_SSE_REMOVAL_WELLS,
        "removed_count": len(remove),
        "removed_wells": remove,
        "remaining_wells": int(len(remaining)),
        "remaining_pooled_rmse_gain_ft": gain,
        "passed": bool(len(remove) == TOP_POSITIVE_SSE_REMOVAL_WELLS and gain > 0.0),
    }


def _coefficient_stability(learned: Sequence[dict]) -> dict:
    if len(learned) != N_FOLDS:
        raise ProtocolError(
            f"spatial coefficient audit expected {N_FOLDS} folds, got {len(learned)}"
        )
    ordered = sorted(learned, key=lambda row: int(row["fold"]))
    if [int(row["fold"]) for row in ordered] != list(range(N_FOLDS)):
        raise ProtocolError("spatial coefficient fold identities drifted")
    definitions = {
        "typewell_shrink": (
            np.array([row["typewell_shrink"] for row in ordered], dtype=float),
            (0.0, 1.5),
        ),
        "ordered_shrink": (
            np.array([row["ordered_shrink"] for row in ordered], dtype=float),
            (0.0, 1.5),
        ),
        "joint_typewell_coefficient": (
            np.array([row["joint_coefficients"][0] for row in ordered], dtype=float),
            (-1.0, 2.0),
        ),
        "joint_ordered_coefficient": (
            np.array([row["joint_coefficients"][1] for row in ordered], dtype=float),
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
        lower_hits = int(
            np.count_nonzero(
                np.isclose(
                    values,
                    lower,
                    rtol=0.0,
                    atol=COEFFICIENT_BOUND_TOLERANCE,
                )
            )
        )
        upper_hits = int(
            np.count_nonzero(
                np.isclose(
                    values,
                    upper,
                    rtol=0.0,
                    atol=COEFFICIENT_BOUND_TOLERANCE,
                )
            )
        )
        sign_flip = positive and negative
        repeated_bound_hit = lower_hits >= 2 or upper_hits >= 2
        coefficients[name] = {
            "bounds": [lower, upper],
            "values_by_fold": values.tolist(),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "sign_flip": bool(sign_flip),
            "lower_bound_hits": lower_hits,
            "upper_bound_hits": upper_hits,
            "repeated_bound_hit": bool(repeated_bound_hit),
            "passed": bool(not sign_flip and not repeated_bound_hit),
        }
    return {
        "fold_order": list(range(N_FOLDS)),
        "absolute_tolerance": COEFFICIENT_BOUND_TOLERANCE,
        "repeated_bound_hit_definition": "two or more folds at one bound",
        "coefficients": coefficients,
        "passed": bool(all(value["passed"] for value in coefficients.values())),
    }


def _artifact_descriptor(path: Path, *, logical_sha256: str | None = None) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise ProtocolError(f"artifact inventory path is missing: {path}")
    descriptor = {
        "name": path.name,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "byte_sha256": group_gate.sha256_file(path),
    }
    if logical_sha256 is not None:
        if not _is_sha256(logical_sha256):
            raise ProtocolError("artifact logical SHA-256 is invalid")
        descriptor["logical_sha256"] = logical_sha256
    sidecar = group_gate._protocol_sidecar(path)
    if sidecar.is_file():
        if group_gate._read_sidecar(path) != descriptor["byte_sha256"]:
            raise ProtocolError(f"artifact sidecar drift: {path}")
        descriptor["sha256_sidecar"] = {
            "name": sidecar.name,
            "path": str(sidecar.resolve()),
            "size_bytes": int(sidecar.stat().st_size),
            "byte_sha256": group_gate.sha256_file(sidecar),
        }
    return descriptor


def _inventory_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_pretruth_artifact_inventory(
    audit: SpatialAuditBundle,
    validated: Sequence[tuple[str, int, Path, dict, dict[str, np.ndarray]]],
) -> dict:
    expected = set(_all_folds())
    identities = [(mode, fold) for mode, fold, _, _, _ in validated]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ProtocolError("spatial artifact inventory does not cover all ten folds")
    fold_items = []
    for mode, fold, shard_path, shard, arrays in validated:
        prediction_path = _prediction_path(shard_path)
        if _read_json(shard_path, "spatial shard") != shard:
            raise ProtocolError("spatial shard changed before artifact inventory")
        shard_descriptor = _artifact_descriptor(shard_path)
        prediction_descriptor = _artifact_descriptor(
            prediction_path,
            logical_sha256=group_gate._logical_array_hash(arrays),
        )
        if shard_descriptor["byte_sha256"] != group_gate.sha256_file(shard_path):
            raise ProtocolError("spatial shard changed during artifact inventory")
        if prediction_descriptor["byte_sha256"] != shard.get("prediction_sha256"):
            raise ProtocolError("spatial prediction changed during artifact inventory")
        if prediction_descriptor["logical_sha256"] != shard.get(
            "prediction_logical_sha256"
        ):
            raise ProtocolError("spatial prediction logical hash changed")
        fold_items.append(
            {
                "mode": mode,
                "fold": fold,
                "shard_json": shard_descriptor,
                "prediction_npz": prediction_descriptor,
            }
        )
    fold_items.sort(key=lambda row: (MODES.index(row["mode"]), row["fold"]))
    payload = {
        "protocol": _artifact_descriptor(audit.protocol_path),
        "data_inventory": _artifact_descriptor(audit.inventory_path),
        "spatial_manifests": {
            mode: _artifact_descriptor(
                audit.manifest_paths[mode],
                logical_sha256=audit.manifests[mode]["manifest_sha256"],
            )
            for mode in MODES
        },
        "prediction_fold_count": len(fold_items),
        "prediction_folds": fold_items,
    }
    return {**payload, "pretruth_inventory_sha256": _inventory_digest(payload)}


def _validate_pretruth_inventory(inventory: Mapping[str, Any]) -> None:
    payload = dict(inventory)
    digest = payload.pop("pretruth_inventory_sha256", None)
    if digest != _inventory_digest(payload):
        raise ProtocolError("pretruth artifact inventory logical digest drift")
    folds = payload.get("prediction_folds")
    if (
        not isinstance(folds, list)
        or int(payload.get("prediction_fold_count", -1)) != 10
    ):
        raise ProtocolError("pretruth artifact inventory is incomplete")
    identities = {(str(row.get("mode")), int(row.get("fold", -1))) for row in folds}
    if len(folds) != 10 or identities != set(_all_folds()):
        raise ProtocolError("pretruth artifact inventory fold identities drifted")

    def validate_descriptor(
        descriptor: object,
        label: str,
        *,
        require_sidecar: bool,
        require_logical: bool = False,
    ) -> str:
        if not isinstance(descriptor, dict):
            raise ProtocolError(f"pretruth inventory lacks {label}")
        name = str(descriptor.get("name", ""))
        path = str(descriptor.get("path", ""))
        if Path(path).name != name or not _is_sha256(descriptor.get("byte_sha256")):
            raise ProtocolError(f"pretruth inventory has invalid {label} identity")
        if int(descriptor.get("size_bytes", -1)) < 0:
            raise ProtocolError(f"pretruth inventory has invalid {label} size")
        if require_logical and not _is_sha256(descriptor.get("logical_sha256")):
            raise ProtocolError(f"pretruth inventory lacks {label} logical digest")
        sidecar = descriptor.get("sha256_sidecar")
        if require_sidecar:
            if not isinstance(sidecar, dict):
                raise ProtocolError(f"pretruth inventory lacks {label} sidecar")
            if (
                Path(str(sidecar.get("path", ""))).name != str(sidecar.get("name", ""))
                or not _is_sha256(sidecar.get("byte_sha256"))
                or int(sidecar.get("size_bytes", -1)) < 0
            ):
                raise ProtocolError(f"pretruth inventory has invalid {label} sidecar")
        return path

    artifact_paths = []
    for label in ("protocol", "data_inventory"):
        descriptor = payload.get(label)
        artifact_paths.append(
            validate_descriptor(descriptor, label, require_sidecar=True)
        )
    manifests = payload.get("spatial_manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(MODES):
        raise ProtocolError("pretruth inventory lacks both spatial manifests")
    for mode in MODES:
        artifact_paths.append(
            validate_descriptor(
                manifests[mode],
                f"{mode} manifest",
                require_sidecar=False,
                require_logical=True,
            )
        )
    for row in folds:
        artifact_paths.append(
            validate_descriptor(
                row.get("shard_json"),
                "spatial shard JSON",
                require_sidecar=True,
            )
        )
        artifact_paths.append(
            validate_descriptor(
                row.get("prediction_npz"),
                "spatial prediction NPZ",
                require_sidecar=False,
                require_logical=True,
            )
        )
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ProtocolError("pretruth artifact inventory aliases artifact paths")


def _persist_aggregate_artifacts(
    output_path: Path,
    scored: pd.DataFrame,
    result: dict,
) -> dict:
    output_path = output_path.resolve()
    scored_path = _scored_sse_path(output_path)
    output_sidecar = group_gate._protocol_sidecar(output_path)
    scored_sidecar = group_gate._protocol_sidecar(scored_path)
    for path in (scored_path, scored_sidecar, output_path, output_sidecar):
        if path.exists():
            raise ProtocolError(f"refusing to overwrite aggregate artifact: {path}")
    pretruth = result.get("pretruth_artifact_inventory")
    if not isinstance(pretruth, dict):
        raise ProtocolError("result lacks pretruth artifact inventory")
    _validate_pretruth_inventory(pretruth)
    missing = set(SCORED_SSE_COLUMNS) - set(scored.columns)
    if missing:
        raise ProtocolError(f"scored SSE rows are missing columns: {sorted(missing)}")
    ordered = scored.loc[:, SCORED_SSE_COLUMNS].sort_values(
        ["mode", "fold", "well"], kind="mergesort"
    )
    if ordered.duplicated(["mode", "well"]).any():
        raise ProtocolError("scored SSE contains duplicate mode/well rows")
    scored_bytes = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    scored_hash = hashlib.sha256(scored_bytes).hexdigest()
    prospective_sidecar_bytes = f"{scored_hash}  {scored_path.name}\n".encode("ascii")
    scored_descriptor = {
        "name": scored_path.name,
        "path": str(scored_path),
        "size_bytes": len(scored_bytes),
        "byte_sha256": scored_hash,
        "rows": int(len(ordered)),
        "columns": list(SCORED_SSE_COLUMNS),
        "sha256_sidecar": {
            "name": scored_sidecar.name,
            "path": str(scored_sidecar),
            "size_bytes": len(prospective_sidecar_bytes),
            "byte_sha256": hashlib.sha256(prospective_sidecar_bytes).hexdigest(),
        },
    }
    complete_inventory = {
        **pretruth,
        "scored_sse": scored_descriptor,
        "result_json": {
            "name": output_path.name,
            "path": str(output_path),
            "sha256_sidecar_name": output_sidecar.name,
            "byte_sha256_location": (
                "external write-once SHA-256 sidecar to avoid recursive self-hash"
            ),
        },
    }
    complete_inventory["complete_inventory_sha256"] = _inventory_digest(
        complete_inventory
    )
    sealed_result = dict(result)
    sealed_result["complete_artifact_inventory"] = complete_inventory
    sealed_result["scored_sse_artifact"] = scored_descriptor
    sealed_result["result_sha256_sidecar"] = output_sidecar.name
    result_bytes = (json.dumps(sealed_result, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    group_gate._atomic_write_bytes(scored_path, scored_bytes)
    group_gate._write_hash_sidecar(scored_path)
    group_gate._atomic_write_bytes(output_path, result_bytes)
    group_gate._write_hash_sidecar(output_path)
    return sealed_result


def _mode_result(
    rows: pd.DataFrame,
    learned: Sequence[dict],
    mode: str,
) -> dict:
    summary = _summary(rows, "joint_sse")
    fold_summaries = {
        str(fold): _summary(rows.loc[rows["fold"] == fold], "joint_sse")
        for fold in range(N_FOLDS)
    }
    fold_gains = _fold_pooled_gains(rows, "joint_sse")
    components = {}
    for label, column in (
        ("typewell", "typewell_sse"),
        ("ordered", "ordered_sse"),
    ):
        component = _summary(rows, column)
        component["fold_pooled_rmse_gains_ft"] = _fold_pooled_gains(rows, column)
        component["top_10_positive_sse_removal"] = _top_positive_sse_removal(
            rows, column
        )
        components[label] = component
    bootstrap = _spatial_cluster_bootstrap(rows, mode)
    influence = _top_positive_sse_removal(rows)
    coefficient_stability = _coefficient_stability(learned)
    fixed_scorecard = {
        "pooled_rmse_gain_at_least_ft": {
            "threshold_ft": MATERIALITY_FT,
            "observed_ft": summary["pooled_row_rmse_gain_ft"],
            "passed": bool(summary["pooled_row_rmse_gain_ft"] >= MATERIALITY_FT),
        },
        "all_five_fold_gains_positive": bool(
            all(value > 0.0 for value in fold_gains.values())
        ),
        "spatial_cluster_bootstrap_ci95_low_positive": bool(
            bootstrap["gain_vs_base_by_channel"]["joint"]["ci95_low_ft"] > 0.0
        ),
        "paired_median_well_gain_at_least_ft": {
            "threshold_ft": MATERIALITY_FT,
            "observed_ft": summary["median_well_rmse_gain_ft"],
            "passed": bool(summary["median_well_rmse_gain_ft"] >= MATERIALITY_FT),
        },
        "joint_vs_resample_best_component_ci95_low_positive": bool(
            bootstrap["joint_vs_resample_best_component"]["ci95_low_ft"] > 0.0
        ),
        "top_10_positive_sse_removal_gain_positive": bool(influence["passed"]),
        "coefficient_sign_and_bound_stability": bool(coefficient_stability["passed"]),
    }
    fixed_scorecard["passed"] = bool(
        fixed_scorecard["pooled_rmse_gain_at_least_ft"]["passed"]
        and fixed_scorecard["all_five_fold_gains_positive"]
        and fixed_scorecard["spatial_cluster_bootstrap_ci95_low_positive"]
        and fixed_scorecard["paired_median_well_gain_at_least_ft"]["passed"]
        and fixed_scorecard["joint_vs_resample_best_component_ci95_low_positive"]
        and fixed_scorecard["top_10_positive_sse_removal_gain_positive"]
        and fixed_scorecard["coefficient_sign_and_bound_stability"]
    )
    return {
        "mode": mode,
        "role": "primary spatial falsification gate"
        if mode == PRIMARY_MODE
        else "secondary centroid-component sensitivity only",
        "aggregate": summary,
        "fold_summaries": fold_summaries,
        "fold_pooled_rmse_gains_ft": fold_gains,
        "spatial_cluster_bootstrap_all_channels_coupled": bootstrap,
        "top_10_positive_sse_removal": influence,
        "coefficient_sign_and_bound_stability": coefficient_stability,
        "fixed_scorecard_no_mode_selection": fixed_scorecard,
        "predeclared_component_diagnostics_no_selection": components,
        "learned_parameters_by_fold": sorted(learned, key=lambda row: int(row["fold"])),
    }


def aggregate_folds(
    protocol_path: Path,
    shard_dir: Path | None,
    output_path: Path,
) -> dict:
    """Validate every spatial shard before revealing any validation metric."""

    output_path = output_path.resolve()
    for path in (
        _scored_sse_path(output_path),
        group_gate._protocol_sidecar(_scored_sse_path(output_path)),
        output_path,
        group_gate._protocol_sidecar(output_path),
    ):
        if path.exists():
            raise ProtocolError(f"refusing to overwrite aggregate artifact: {path}")
    audit = audit_protocol(protocol_path, verify_data=True)
    shard_dir = (
        shard_dir.resolve()
        if shard_dir is not None
        else audit.protocol_path.with_name(audit.protocol_path.stem + "_folds")
    )
    expected_names = {_fold_shard_name(mode, fold) for mode, fold in _all_folds()}
    present_names = {path.name for path in shard_dir.glob("*_fold_*.json")}
    if present_names != expected_names:
        missing = sorted(expected_names - present_names)
        extra = sorted(present_names - expected_names)
        raise ProtocolError(
            "spatial shard set is incomplete or unexpected; "
            f"missing={missing}, extra={extra}"
        )
    expected_predictions = {
        Path(name).with_suffix(".npz").name for name in expected_names
    }
    present_predictions = {path.name for path in shard_dir.glob("*_fold_*.npz")}
    if present_predictions != expected_predictions:
        missing = sorted(expected_predictions - present_predictions)
        extra = sorted(present_predictions - expected_predictions)
        raise ProtocolError(
            "spatial prediction set is incomplete or unexpected; "
            f"missing={missing}, extra={extra}"
        )
    expected_sidecars = {f"{name}.sha256" for name in expected_names}
    present_sidecars = {path.name for path in shard_dir.glob("*_fold_*.json.sha256")}
    if present_sidecars != expected_sidecars:
        missing = sorted(expected_sidecars - present_sidecars)
        extra = sorted(present_sidecars - expected_sidecars)
        raise ProtocolError(
            "spatial shard-sidecar set is incomplete or unexpected; "
            f"missing={missing}, extra={extra}"
        )

    validated = []
    learned_by_mode: dict[str, list[dict]] = {mode: [] for mode in MODES}
    runtime_by_mode = {mode: 0.0 for mode in MODES}
    for mode, fold in _all_folds():
        path = shard_dir / _fold_shard_name(mode, fold)
        shard = _read_json(path, "spatial shard")
        arrays = _validate_shard(shard, audit, mode, fold, path)
        runtime_by_mode[mode] += float(shard["runtime_seconds"])
        learned_by_mode[mode].append(
            {
                "fold": fold,
                **shard["learned_from_manifest_retained_training_only"],
            }
        )
        validated.append((mode, fold, path, shard, arrays))

    pretruth_inventory = _build_pretruth_artifact_inventory(audit, validated)

    # Validation suffix truth remains unopened until the complete ten-shard
    # prediction inventory above passes.  The following loop is the first truth
    # boundary in the aggregate phase.
    scored_rows = []
    for mode, fold, _, shard, arrays in validated:
        scored_rows.extend(_score_sealed_predictions(audit, shard, arrays, mode, fold))
    scored = pd.DataFrame(scored_rows)
    expected_scored_rows = len(MODES) * EXPECTED_ELIGIBLE_WELLS
    if len(scored) != expected_scored_rows:
        raise ProtocolError(
            f"spatial aggregate expected exactly {expected_scored_rows} per-well "
            f"SSE rows, got {len(scored)}"
        )
    for mode in MODES:
        expected_wells = set(audit.inventory["well"].astype(str))
        observed_wells = set(scored.loc[scored["mode"] == mode, "well"].astype(str))
        if observed_wells != expected_wells:
            raise ProtocolError(
                f"{mode} scoring does not cover every well exactly once"
            )
        if scored.loc[scored["mode"] == mode, "well"].duplicated().any():
            raise ProtocolError(f"{mode} scoring contains duplicate wells")

    mode_results = {
        mode: _mode_result(
            scored.loc[scored["mode"] == mode].copy(), learned_by_mode[mode], mode
        )
        for mode in MODES
    }
    secondary = mode_results[SECONDARY_MODE]
    secondary_summary = secondary["aggregate"]
    secondary_bootstrap = secondary["spatial_cluster_bootstrap_all_channels_coupled"]
    if (
        secondary_summary["pooled_row_rmse_gain_ft"] < 0.0
        or secondary_summary["median_well_rmse_gain_ft"] < 0.0
    ):
        secondary_classification = "SECONDARY_CONTRADICTION"
    elif (
        secondary_summary["pooled_row_rmse_gain_ft"] > 0.0
        and secondary_summary["median_well_rmse_gain_ft"] > 0.0
        and secondary_bootstrap["gain_vs_base_by_channel"]["joint"]["ci95_low_ft"] > 0.0
    ):
        secondary_classification = "SECONDARY_SUPPORTIVE_SENSITIVITY"
    else:
        secondary_classification = "SECONDARY_INCONCLUSIVE"
    primary = mode_results[PRIMARY_MODE]
    primary_summary = primary["aggregate"]
    primary_fold_gains = primary["fold_pooled_rmse_gains_ft"]
    primary_bootstrap = primary["spatial_cluster_bootstrap_all_channels_coupled"]
    primary_influence = primary["top_10_positive_sse_removal"]
    primary_coefficients = primary["coefficient_sign_and_bound_stability"]
    primary_gate = {
        "pooled_rmse_gain_at_least_ft": {
            "threshold_ft": MATERIALITY_FT,
            "observed_ft": primary_summary["pooled_row_rmse_gain_ft"],
            "passed": bool(
                primary_summary["pooled_row_rmse_gain_ft"] >= MATERIALITY_FT
            ),
        },
        "all_five_region_fold_gains_positive": bool(
            all(value > 0.0 for value in primary_fold_gains.values())
        ),
        "sealed_region_fold_bootstrap_ci95_low_positive": bool(
            primary_bootstrap["gain_vs_base_by_channel"]["joint"]["ci95_low_ft"] > 0.0
        ),
        "joint_vs_resample_best_component_ci95_low_positive": bool(
            primary_bootstrap["joint_vs_resample_best_component"]["ci95_low_ft"] > 0.0
        ),
        "paired_median_well_gain_at_least_ft": {
            "threshold_ft": MATERIALITY_FT,
            "observed_ft": primary_summary["median_well_rmse_gain_ft"],
            "passed": bool(
                primary_summary["median_well_rmse_gain_ft"] >= MATERIALITY_FT
            ),
        },
        "top_10_positive_sse_removal_gain_positive": bool(primary_influence["passed"]),
        "coefficient_sign_and_bound_stability": bool(primary_coefficients["passed"]),
    }
    primary_gate["passed"] = bool(
        primary_gate["pooled_rmse_gain_at_least_ft"]["passed"]
        and primary_gate["all_five_region_fold_gains_positive"]
        and primary_gate["sealed_region_fold_bootstrap_ci95_low_positive"]
        and primary_gate["joint_vs_resample_best_component_ci95_low_positive"]
        and primary_gate["paired_median_well_gain_at_least_ft"]["passed"]
        and primary_gate["top_10_positive_sse_removal_gain_positive"]
        and primary_gate["coefficient_sign_and_bound_stability"]
    )
    if primary_gate["passed"] != primary["fixed_scorecard_no_mode_selection"]["passed"]:
        raise ProtocolError("primary and fixed spatial scorecards disagree")
    result = {
        "status": "MEASURE_ONLY_SPATIAL_SUPPORT"
        if primary_gate["passed"]
        else "MEASURE_ONLY_SPATIAL_NOT_SUPPORTED",
        "protocol_sha256": audit.protocol_sha256,
        "method": METHOD,
        "predeclared_mode_policy": _evaluation_protocol()["mode_policy"],
        "primary_spatial_support_gate": primary_gate,
        "mode_results_no_selection": mode_results,
        "secondary_pad_interpretation_no_gate_effect": {
            "classification": secondary_classification,
            "rule": (
                "negative pooled or paired-median point gain is a contradiction; "
                "positive but uncertain gain is inconclusive; this field never "
                "rescues or vetoes region_out"
            ),
        },
        "sum_fold_runtime_seconds_by_mode": runtime_by_mode,
        "pretruth_artifact_inventory": pretruth_inventory,
        "interpretation": (
            "Only region_out determines spatial falsification survival. pad_out is "
            "reported solely as a centroid-component sensitivity diagnostic. Neither "
            "outcome can promote production OPEN."
        ),
    }

    commit_audit = audit_protocol(protocol_path, verify_data=True)
    if commit_audit.protocol_sha256 != audit.protocol_sha256:
        raise ProtocolError("spatial protocol drifted before aggregate commit")
    commit_validated = []
    for mode, fold, path, _, _ in validated:
        shard = _read_json(path, "spatial shard")
        arrays = _validate_shard(shard, commit_audit, mode, fold, path)
        commit_validated.append((mode, fold, path, shard, arrays))
    if (
        _build_pretruth_artifact_inventory(commit_audit, commit_validated)
        != pretruth_inventory
    ):
        raise ProtocolError("spatial artifact inventory drifted before commit")
    sealed = _persist_aggregate_artifacts(output_path, scored, result)
    print(json.dumps(sealed, indent=2), flush=True)
    return sealed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze both spatial modes")
    freeze.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing train/ and test/",
    )
    freeze.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "research/results/spatial_gate_manifests",
    )
    freeze.add_argument("--protocol", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="fail closed on frozen drift")
    audit.add_argument("--protocol", type=Path, required=True)

    run = subparsers.add_parser("run", help="write metric-silent spatial shards")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-folds", action="store_true")
    selection.add_argument("--mode", choices=MODES)
    run.add_argument("--fold", type=int)
    run.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate", help="validate all ten shards, then reveal truth once"
    )
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--shard-dir", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.all_folds and args.fold is not None:
            parser.error("--fold cannot be used with --all-folds")
        if args.mode is not None and args.fold is None:
            parser.error("--fold is required when --mode is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        protocol, inventory, protocol_sidecar, inventory_sidecar = freeze_protocol(
            args.data_dir, args.manifest_dir, args.protocol
        )
        print(f"froze spatial protocol {protocol}")
        print(f"froze data inventory {inventory}")
        print(f"wrote protocol hash {protocol_sidecar}")
        print(f"wrote inventory hash {inventory_sidecar}")
    elif args.command == "audit":
        bundle = audit_protocol(args.protocol, verify_data=True)
        manifest_hashes = ", ".join(
            f"{mode}: {bundle.manifests[mode]['manifest_sha256']}" for mode in MODES
        )
        print(
            f"spatial audit passed: protocol={bundle.protocol_sha256}, "
            f"manifests={{{manifest_hashes}}}"
        )
    elif args.command == "run":
        folds = _all_folds() if args.all_folds else [(args.mode, args.fold)]
        run_folds(args.protocol, args.output_dir, folds, args.resume)
    elif args.command == "aggregate":
        aggregate_folds(args.protocol, args.shard_dir, args.output)
    else:  # pragma: no cover
        raise ProtocolError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
