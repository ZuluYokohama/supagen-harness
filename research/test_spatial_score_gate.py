from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import spatial_score_gate as gate
from research import spatial_split


MANIFEST_DIR = gate.ROOT / "research/results/spatial_gate_manifests"


def _load_real_manifest(mode: str) -> dict:
    return json.loads((MANIFEST_DIR / f"{mode}.json").read_text(encoding="utf-8"))


def _inventory_for_manifest(manifest: dict) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(manifest["wells"]):
        profile = f"{index:064x}"[-64:]
        rows.append(
            {
                "well": row["well_id"],
                "typewell_profile_hash": profile,
                "rows": row["n_rows"],
                "prefix_rows": row["n_rows"] - row["n_suffix_rows"],
                "suffix_rows": row["n_suffix_rows"],
                "gr_valid_fraction": 1.0,
                "horizontal_sha256": f"{index + 1:064x}"[-64:],
                "typewell_sha256": row["typewell_sha256"],
                "horizontal_file": row["source_file"],
                "typewell_file": row["typewell_file"],
            }
        )
    return pd.DataFrame(rows)


def _metric_rows(mode: str) -> pd.DataFrame:
    clusters = (
        gate.N_FOLDS if mode == gate.PRIMARY_MODE else gate.EXPECTED_PAD_COMPONENTS
    )
    records = []
    for index in range(clusters):
        base_rmse = 10.0 + 0.02 * (index % 7)
        typewell_rmse = base_rmse - 0.25
        ordered_rmse = base_rmse - 0.20
        joint_rmse = base_rmse - 0.45
        records.append(
            {
                "mode": mode,
                "fold": index if mode == gate.PRIMARY_MODE else index % gate.N_FOLDS,
                "well": f"w{index:04d}",
                "typewell_profile_hash": f"{index:064x}"[-64:],
                "spatial_cluster_id": (
                    f"region_fold_{index}"
                    if mode == gate.PRIMARY_MODE
                    else f"pad_{index:04d}"
                ),
                "n_rows": 100 + index,
                "base_sse": base_rmse**2 * (100 + index),
                "typewell_sse": typewell_rmse**2 * (100 + index),
                "ordered_sse": ordered_rmse**2 * (100 + index),
                "joint_sse": joint_rmse**2 * (100 + index),
            }
        )
    return pd.DataFrame(records)


def _learned() -> list[dict]:
    return [
        {
            "fold": fold,
            "typewell_shrink": 0.5 + 0.02 * fold,
            "ordered_shrink": 0.7 + 0.01 * fold,
            "joint_coefficients": [0.4 + 0.01 * fold, 0.8 - 0.01 * fold],
        }
        for fold in range(gate.N_FOLDS)
    ]


def _fake_pretruth_inventory(tmp_path: Path) -> tuple[dict, gate.SpatialAuditBundle]:
    protocol_path = tmp_path / "protocol.json"
    inventory_path = tmp_path / "protocol_data_inventory.csv"
    gate.group_gate._atomic_write_bytes(protocol_path, b"{}\n")
    gate.group_gate._write_hash_sidecar(protocol_path)
    gate.group_gate._atomic_write_bytes(inventory_path, b"well\nw0\n")
    gate.group_gate._write_hash_sidecar(inventory_path)

    manifest_paths = {}
    manifests = {}
    for mode in gate.MODES:
        path = tmp_path / f"{mode}.json"
        gate.group_gate._atomic_write_bytes(path, b"{}\n")
        manifest_paths[mode] = path
        manifests[mode] = {
            "manifest_sha256": ("a" if mode == gate.PRIMARY_MODE else "b") * 64
        }
    bundle = gate.SpatialAuditBundle(
        protocol={},
        inventory=pd.DataFrame(),
        manifests=manifests,
        protocol_sha256=gate.group_gate.sha256_file(protocol_path),
        protocol_path=protocol_path,
        inventory_path=inventory_path,
        manifest_paths=manifest_paths,
    )

    validated = []
    for mode, fold in gate._all_folds():
        shard_path = tmp_path / gate._fold_shard_name(mode, fold)
        prediction_path = gate._prediction_path(shard_path)
        arrays = {
            "row_index": np.array([fold], dtype=np.int32),
            "joint_prediction": np.array([100.0 + fold], dtype=np.float64),
        }
        gate.group_gate._atomic_write_npz(prediction_path, arrays)
        shard = {
            "prediction_file": prediction_path.name,
            "prediction_sha256": gate.group_gate.sha256_file(prediction_path),
            "prediction_logical_sha256": gate.group_gate._logical_array_hash(arrays),
        }
        gate.group_gate._atomic_write_json(shard_path, shard)
        gate.group_gate._write_hash_sidecar(shard_path)
        validated.append((mode, fold, shard_path, shard, arrays))
    return gate._build_pretruth_artifact_inventory(bundle, validated), bundle


def test_real_manifests_satisfy_seal_roles_and_shared_identity() -> None:
    manifests = {mode: _load_real_manifest(mode) for mode in gate.MODES}
    for mode, manifest in manifests.items():
        gate._validate_spatial_manifest(manifest, mode)
        assert gate._load_manifest(MANIFEST_DIR / f"{mode}.json", mode) == manifest
    gate._validate_cross_manifest_population(manifests)

    assert (
        manifests[gate.PRIMARY_MODE]["dataset_sha256"]
        == gate.EXPECTED_SPATIAL_DATASET_SHA256
    )
    assert manifests[gate.PRIMARY_MODE]["parameters"]["trajectory_isolation"] is True
    assert manifests[gate.SECONDARY_MODE]["parameters"]["trajectory_isolation"] is False
    assert (
        len({row["pad_component"] for row in manifests[gate.SECONDARY_MODE]["wells"]})
        == 183
    )
    assert manifests[gate.PRIMARY_MODE]["diagnostics"]["n_suffix_rows"] == 3_769_838
    assert (
        len({row["equality_group"] for row in manifests[gate.PRIMARY_MODE]["wells"]})
        == 749
    )
    assert manifests[gate.PRIMARY_MODE]["resampled_polyline_spacing_ft"] == 100.0
    assert manifests[gate.PRIMARY_MODE]["parameters"]["centroid_embargo_ft"] == 5000.0
    assert (
        manifests[gate.PRIMARY_MODE]["parameters"]["resampled_polyline_embargo_ft"]
        == 1500.0
    )
    assert (
        manifests[gate.SECONDARY_MODE]["parameters"]["centroid_component_radius_ft"]
        == 1500.0
    )


def test_region_uses_retained_training_and_never_repurposes_embargo() -> None:
    manifest = _load_real_manifest(gate.PRIMARY_MODE)
    universe = {row["well_id"] for row in manifest["wells"]}
    for fold in manifest["folds"]:
        training = set(fold["training_ids"])
        validation = set(fold["validation_ids"])
        embargo = set(fold["embargo_ids"])
        assert training.isdisjoint(embargo)
        assert training != universe - validation
        assert training == universe - validation - embargo

    source = inspect.getsource(gate._run_one_fold)
    assert "training_ids, validation_ids, embargo_ids = _fold_roles" in source
    # The fold must take its roles from _fold_roles and never re-derive them
    # from a universe set. Check executable text only, so a comment or docstring
    # that happens to mention the word cannot fail this.
    code_only = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "universe" not in code_only
    assert "train_files" in source
    assert "set(training_group_by_well) & set(embargo_ids)" in source


def test_parsed_exact_profile_crossing_any_spatial_role_fails_closed() -> None:
    manifest = _load_real_manifest(gate.PRIMARY_MODE)
    inventory = _inventory_for_manifest(manifest)
    fold = manifest["folds"][0]
    validation_well = fold["validation_ids"][0]
    training_well = fold["training_ids"][0]
    shared_hash = "f" * 64
    inventory.loc[
        inventory["well"].isin([validation_well, training_well]),
        "typewell_profile_hash",
    ] = shared_hash
    with pytest.raises(gate.ProtocolError, match="one-to-one with parsed profiles"):
        gate._validate_spatial_manifest(manifest, gate.PRIMARY_MODE, inventory)


def test_cross_mode_geometry_or_hash_drift_fails_even_when_resealed() -> None:
    manifests = {mode: _load_real_manifest(mode) for mode in gate.MODES}
    changed = copy.deepcopy(manifests[gate.SECONDARY_MODE])
    changed["wells"][0]["trajectory_sha256"] = "0" * 64
    manifests[gate.SECONDARY_MODE] = spatial_split.seal_manifest(changed)
    with pytest.raises(gate.ProtocolError, match="cross-mode manifest identity drift"):
        gate._validate_cross_manifest_population(manifests)


def test_predeclared_mode_and_fit_contract_forbid_selection_and_state_sharing() -> None:
    evaluation = gate._evaluation_protocol()
    policy = evaluation["mode_policy"]
    assert policy["primary"] == "region_out"
    assert policy["secondary"] == "pad_out"
    assert policy["winner_selection"] == "forbidden"
    assert policy["secondary_can_rescue_primary"] is False
    assert evaluation["bootstrap"]["region_out"]["draws"] == 5**5
    assert evaluation["bootstrap"]["pad_out"]["expected_clusters"] == 183
    assert (
        evaluation["primary_support_criteria"][
            "joint_vs_resample_best_component_ci95_low_positive"
        ]
        is True
    )
    assert "no fitted model" in gate._fit_contract()["state_isolation"]
    assert gate._fit_contract()["method"] == gate.group_gate.METHOD == gate.METHOD


def test_region_bootstrap_is_exhaustive_coupled_and_order_invariant() -> None:
    rows = _metric_rows(gate.PRIMARY_MODE)
    first = gate._spatial_cluster_bootstrap(rows, gate.PRIMARY_MODE)
    shuffled = gate._spatial_cluster_bootstrap(
        rows.sample(frac=1.0, random_state=17), gate.PRIMARY_MODE
    )
    assert first == shuffled
    assert first["draws"] == 3125
    assert first["seed"] is None
    assert first["n_clusters"] == 5
    assert first["same_cluster_multiplicities_applied_to_all_channels"] is True
    assert first["gain_vs_base_by_channel"]["joint"]["ci95_low_ft"] > 0.0
    assert first["joint_vs_resample_best_component"]["ci95_low_ft"] > 0.0


def test_pad_bootstrap_is_seeded_over_all_183_components() -> None:
    rows = _metric_rows(gate.SECONDARY_MODE)
    first = gate._spatial_cluster_bootstrap(rows, gate.SECONDARY_MODE)
    second = gate._spatial_cluster_bootstrap(rows, gate.SECONDARY_MODE)
    assert first == second
    assert first["draws"] == 4000
    assert first["seed"] == 20260810
    assert first["n_clusters"] == 183


def test_spatial_cluster_bootstrap_rejects_wrong_primary_unit_count() -> None:
    rows = _metric_rows(gate.PRIMARY_MODE).iloc[:-1]
    with pytest.raises(gate.ProtocolError, match="expected 5 sealed clusters"):
        gate._spatial_cluster_bootstrap(rows, gate.PRIMARY_MODE)


def test_primary_metrics_are_paired_robust_and_coefficient_gated() -> None:
    rows = _metric_rows(gate.PRIMARY_MODE)
    summary = gate._summary(rows)
    assert summary["pooled_row_rmse_gain_ft"] > gate.MATERIALITY_FT
    assert summary["median_well_rmse_gain_ft"] > gate.MATERIALITY_FT
    assert all(value > 0 for value in gate._fold_pooled_gains(rows).values())
    # Repeat rows so at least ten positive wells exist for the influence gate.
    expanded = pd.concat(
        [
            rows,
            rows.assign(well=lambda frame: frame["well"] + "b"),
            rows.assign(well=lambda frame: frame["well"] + "c"),
        ],
        ignore_index=True,
    )
    assert gate._top_positive_sse_removal(expanded)["passed"] is True
    assert gate._coefficient_stability(_learned())["passed"] is True

    bounded = _learned()
    bounded[0] = {**bounded[0], "ordered_shrink": 1.5}
    bounded[1] = {**bounded[1], "ordered_shrink": 1.5}
    assert gate._coefficient_stability(bounded)["passed"] is False


def test_both_modes_receive_the_same_fixed_scorecard_without_selection() -> None:
    region = _metric_rows(gate.PRIMARY_MODE)
    region = pd.concat(
        [
            region,
            region.assign(well=lambda frame: frame["well"] + "b"),
            region.assign(well=lambda frame: frame["well"] + "c"),
        ],
        ignore_index=True,
    )
    pad = _metric_rows(gate.SECONDARY_MODE)
    region_result = gate._mode_result(region, _learned(), gate.PRIMARY_MODE)
    pad_result = gate._mode_result(pad, _learned(), gate.SECONDARY_MODE)
    assert set(region_result["fixed_scorecard_no_mode_selection"]) == set(
        pad_result["fixed_scorecard_no_mode_selection"]
    )
    assert region_result["fixed_scorecard_no_mode_selection"]["passed"] is True
    assert pad_result["fixed_scorecard_no_mode_selection"]["passed"] is True
    for result in (region_result, pad_result):
        assert (
            "top_10_positive_sse_removal"
            in result["predeclared_component_diagnostics_no_selection"]["typewell"]
        )
        assert (
            "top_10_positive_sse_removal"
            in result["predeclared_component_diagnostics_no_selection"]["ordered"]
        )


def test_all_ten_shards_are_inventoried_before_truth(tmp_path: Path) -> None:
    inventory, _ = _fake_pretruth_inventory(tmp_path)
    gate._validate_pretruth_inventory(inventory)
    assert inventory["prediction_fold_count"] == 10
    assert len(inventory["prediction_folds"]) == 10
    assert set(inventory["spatial_manifests"]) == set(gate.MODES)
    assert "sha256_sidecar" in inventory["protocol"]
    assert "sha256_sidecar" in inventory["data_inventory"]
    assert all(
        "sha256_sidecar" in row["shard_json"] for row in inventory["prediction_folds"]
    )

    incomplete = copy.deepcopy(inventory)
    incomplete["prediction_folds"].pop()
    incomplete_payload = dict(incomplete)
    incomplete_payload.pop("pretruth_inventory_sha256")
    incomplete["pretruth_inventory_sha256"] = gate._inventory_digest(incomplete_payload)
    with pytest.raises(gate.ProtocolError, match=r"incomplete|identities"):
        gate._validate_pretruth_inventory(incomplete)


def test_aggregate_artifacts_are_complete_sealed_and_write_once(tmp_path: Path) -> None:
    pretruth, _ = _fake_pretruth_inventory(tmp_path / "inputs")
    scored = pd.concat(
        [_metric_rows(mode).iloc[:1] for mode in gate.MODES], ignore_index=True
    )
    output = tmp_path / "result.json"
    sealed = gate._persist_aggregate_artifacts(
        output,
        scored,
        {
            "status": "test",
            "pretruth_artifact_inventory": pretruth,
        },
    )
    scored_path = gate._scored_sse_path(output)
    assert output.is_file() and scored_path.is_file()
    assert gate.group_gate._protocol_sidecar(output).is_file()
    assert gate.group_gate._protocol_sidecar(scored_path).is_file()
    complete = sealed["complete_artifact_inventory"]
    assert complete["prediction_fold_count"] == 10
    assert complete["scored_sse"]["byte_sha256"] == gate.group_gate.sha256_file(
        scored_path
    )
    assert complete["result_json"]["name"] == output.name
    assert gate._is_sha256(complete["complete_inventory_sha256"])
    with pytest.raises(gate.ProtocolError, match="refusing to overwrite"):
        gate._persist_aggregate_artifacts(
            output,
            scored,
            {"status": "again", "pretruth_artifact_inventory": pretruth},
        )


def test_run_has_no_validation_truth_and_aggregate_crosses_boundary_once() -> None:
    run_source = inspect.getsource(gate._run_one_fold)
    fit_source = inspect.getsource(gate._fit_candidate_fold)
    score_source = inspect.getsource(gate._score_sealed_predictions)
    aggregate_source = inspect.getsource(gate.aggregate_folds)

    assert ".truth" not in run_source and '["truth"]' not in run_source
    assert "base_sse" not in run_source and "joint_sse" not in run_source
    assert 'well["truth"]' in score_source
    assert "base_sse" in score_source and "joint_sse" in score_source
    assert "group_gate._build_static_training_matrix(train_files)" in fit_source
    assert "group_gate._fit_base_fold" in fit_source
    assert "group_gate._fit_joint_correction" in fit_source
    assert "all_spatial_manifest_sha256" in run_source
    # Assert presence first: str.index would otherwise raise a bare ValueError
    # that says nothing about which stage went missing.
    ordered_stages = (
        "validated.append",
        "pretruth_inventory = _build_pretruth_artifact_inventory",
        "_score_sealed_predictions",
    )
    positions = []
    for stage in ordered_stages:
        position = aggregate_source.find(stage)
        assert position >= 0, f"aggregate stage missing from source: {stage}"
        positions.append(position)
    assert positions == sorted(positions), (
        f"aggregate stages out of order: {ordered_stages}"
    )
    assert "expected exactly {expected_scored_rows}" in aggregate_source


def test_cli_requires_both_or_one_complete_fold_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = gate.parse_args(["run", "--protocol", "p.json", "--all-folds", "--resume"])
    assert args.all_folds and args.resume
    args = gate.parse_args(
        ["run", "--protocol", "p.json", "--mode", "region_out", "--fold", "3"]
    )
    assert args.mode == "region_out" and args.fold == 3
    with pytest.raises(SystemExit, match="2"):
        gate.parse_args(["run", "--protocol", "p.json", "--all-folds", "--fold", "2"])
    assert "--fold cannot be used with --all-folds" in capsys.readouterr().err


def test_logical_hash_and_manifest_seal_tampering_fail_closed() -> None:
    manifest = _load_real_manifest(gate.PRIMARY_MODE)
    assert spatial_split.verify_manifest_sha256(manifest)
    manifest["parameters"]["centroid_embargo_ft"] += 1.0
    with pytest.raises(gate.ProtocolError, match="logical SHA-256 drift"):
        gate._validate_spatial_manifest(manifest, gate.PRIMARY_MODE)

    arrays = {
        "row_index": np.array([1, 2], dtype=np.int32),
        "joint_prediction": np.array([10.0, 11.0]),
    }
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["joint_prediction"][0] += 1e-10
    assert gate.group_gate._logical_array_hash(
        arrays
    ) != gate.group_gate._logical_array_hash(changed)
