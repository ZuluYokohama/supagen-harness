from __future__ import annotations

import copy
import ast
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research import structural_field_gate_v2 as gate


def _frame(*, suffix_tvt_shift: float = 0.0, x_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(7, dtype=float) * 100.0,
            "X": x_shift + np.arange(7, dtype=float) * 100.0,
            "Y": np.zeros(7),
            "Z": np.linspace(1_000.0, 994.0, 7),
            "TVT_input": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan, np.nan],
            "TVT": [
                10.0,
                11.0,
                12.0,
                13.0 + suffix_tvt_shift,
                14.0 + suffix_tvt_shift,
                15.0 + suffix_tvt_shift,
                16.0 + suffix_tvt_shift,
            ],
            "Geology": ["a", "a", "b", "secret", "secret", "secret", "secret"],
        }
    )


def _suffix(*, base_shift: float = 0.0) -> gate.IncumbentSuffix:
    return gate.IncumbentSuffix(
        row_index=np.arange(3, 7, dtype=np.int64),
        base=np.array([12.8, 13.8, 14.8, 15.8]) + base_shift,
        joint=np.array([13.0, 14.0, 15.0, 16.0]),
    )


def _well(
    well_id: str,
    *,
    truth: bool = True,
    suffix_tvt_shift: float = 0.0,
    x_shift: float = 0.0,
) -> gate.WellPath:
    frame = _frame(suffix_tvt_shift=suffix_tvt_shift, x_shift=x_shift)
    if truth:
        selected = frame.loc[:, list(gate.TRAINING_COLUMNS)]
        return gate._compose_well(well_id, selected, _suffix(), "training")
    selected = frame.loc[:, list(gate.INFERENCE_COLUMNS)]
    return gate._compose_well(well_id, selected, _suffix(), "validation")


def _training_population(*, truth_shift: float = 0.0) -> tuple[dict, dict]:
    wells = {}
    groups = {}
    for index in range(8):
        well_id = f"w{index}"
        wells[well_id] = _well(
            well_id,
            suffix_tvt_shift=(truth_shift if index == 3 else 0.0),
            x_shift=10_000.0 * index,
        )
        groups[well_id] = f"g{index}"
    return wells, groups


def _proposal(
    delta: np.ndarray,
    support: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> SimpleNamespace:
    count = len(delta)
    return SimpleNamespace(
        field_delta_without_prefix_bias_tvt=np.asarray(delta, dtype=float),
        prefix_bias_delta_tvt=(
            np.zeros(count, dtype=float)
            if bias is None
            else np.asarray(bias, dtype=float)
        ),
        support_mask=np.asarray(support, dtype=bool),
        confidence=(
            np.asarray(support, dtype=float)
            if confidence is None
            else np.asarray(confidence, dtype=float)
        ),
        diagnostics=SimpleNamespace(
            status="anchored_field_100ft_knots_with_policy_fallback",
            evaluation_rows=count,
            prefix_rows=3,
            suffix_rows=count - 3,
            prefix_bias=0.0,
            prefix_bias_intervals=2,
            nearest_resampled_training_midpoint_distance_mean_ft=1.0,
            nearest_resampled_training_midpoint_distance_max_ft=2.0,
            effective_well_support_mean=2.0,
            query_direction_observability_mean=0.8,
            cut_edge_crossings=0,
            fallback_fraction=0.0,
            mean_confidence=float(
                np.mean(confidence if confidence is not None else support)
            ),
        ),
    )


def _model(*, sigma: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        config=gate.field_config(),
        diagnostics=SimpleNamespace(
            wells=6,
            resampled_intervals=30,
            inducing_nodes=9,
            graph_edges=12,
            graph_faces=4,
            discontinuity_candidates=0,
            graph_components_after_cuts=1,
            actual_inducing_cell_ft=gate.FIXED_INDUCING_CELL_FT,
            derivative_residual_scale=sigma,
            solver_stop_codes=(1, 1),
        ),
    )


def test_fixed_physical_identity_and_no_calibration_surface() -> None:
    assert gate.FIXED_H_FT == 15_000.0
    assert gate.FIXED_INDUCING_CELL_FT == 7_500.0
    assert gate.FIXED_LAPLACIAN == 3.0
    assert gate.THETA_FIELD == gate.THETA_BIAS == 1.0
    config = gate.field_config()
    assert config.support_length_ft == 15_000.0
    assert config.inducing_cell_ft == 7_500.0
    assert config.graph_max_edge_ft == 22_500.0
    assert config.laplacian_strength == 3.0
    assert config.circulation_strength == 0.1
    assert config.ridge_strength == 1.0e-6
    assert config.max_distinct_support_wells == 16
    assert config.max_support_neighbors == 4_096
    fixed = gate.fixed_configuration()
    assert fixed == {
        "physical_h_ft": 15_000.0,
        "effective_field_config": asdict(config),
        "theta_field": 1.0,
        "theta_bias": 1.0,
    }
    assert asdict(config) == {
        "resample_step_md": 100.0,
        "lateral_max_abs_dz_dmd": 0.15,
        "min_horizontal_speed": 1.0e-4,
        "inducing_cell_ft": 7_500.0,
        "max_nodes": 2_000,
        "max_resampled_intervals_per_well": 20_000,
        "max_training_observations": 250_000,
        "max_prediction_rows": 100_000,
        "max_support_neighbors": 4_096,
        "max_distinct_support_wells": 16,
        "interpolation_neighbors": 6,
        "support_length_ft": 15_000.0,
        "graph_neighbors": 6,
        "graph_max_edge_ft": 22_500.0,
        "laplacian_strength": 3.0,
        "circulation_strength": 0.1,
        "ridge_strength": 1.0e-6,
        "huber_delta": 1.5,
        "irls_iterations": 4,
        "solver_max_iterations": 2_000,
        "discontinuity_mad_threshold": 4.0,
        "discontinuity_absolute_floor": 1.0e-4,
        "cut_fallback_radius_ft": 500.0,
        "prefix_bias_window_md": 1_000.0,
        "prefix_bias_shrink": 1.0,
        "max_abs_prefix_bias": 0.10,
        "blend_alpha": 1.0,
        "min_effective_wells": 1.5,
        "min_directional_observability": 0.05,
    }

    contract = gate._evaluation_contract()
    serialized = json.dumps(contract, sort_keys=True).lower()
    assert "calibrat" not in serialized
    assert "grid" not in serialized
    assert contract["fixed_coefficients"] == {
        "theta_field": 1.0,
        "theta_bias": 1.0,
    }


def test_v2_source_contains_no_lightgbm_or_theta_solver() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "LGBMRegressor" not in source
    assert "solve_theta" not in source
    assert "_crossfit_incumbent_training" not in source
    assert "_fit_field_grid" not in source


def test_every_legacy_attribute_reference_is_in_exact_audited_allowlist() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "legacy"
    }
    assert referenced == gate.LEGACY_HELPER_ALLOWLIST
    forbidden_fragments = {
        "calibr",
        "evidence",
        "fit_joint",
        "fit_field_grid",
        "crossfit",
        "lgb",
    }
    assert not {
        name
        for name in referenced
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    }


def test_inner_groupkfold_roles_ignore_training_values() -> None:
    original, groups = _training_population()
    mutated, _ = _training_population(truth_shift=1_000_000.0)
    first = gate._inner_fold_roles(original, groups)
    second = gate._inner_fold_roles(mutated, groups)
    assert first.fold_by_well == second.fold_by_well
    assert first.metadata == second.metadata
    assert set(first.fold_by_well.values()) == set(range(4))
    for group in set(groups.values()):
        assert (
            len(
                {
                    first.fold_by_well[well_id]
                    for well_id in groups
                    if groups[well_id] == group
                }
            )
            == 1
        )


def test_inner_groupkfold_rejects_outer_or_group_alias() -> None:
    wells, groups = _training_population()
    del groups["w7"]
    with pytest.raises(gate.GateError, match="group membership"):
        gate._inner_fold_roles(wells, groups)
    groups["w7"] = groups["w0"]
    del wells["w7"]
    with pytest.raises(gate.GateError, match="group membership"):
        gate._inner_fold_roles(wells, groups)


def test_fixed_fit_executes_four_leave_one_roles_and_one_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wells, groups = _training_population()
    partition = gate._inner_fold_roles(wells, groups)
    fitted: list[tuple[str, ...]] = []

    def fit(training: list, config: object) -> SimpleNamespace:
        assert config == gate.field_config()
        fitted.append(tuple(sorted(item.well_id for item in training)))
        model = _model()
        model.diagnostics.wells = len(training)
        return model

    monkeypatch.setattr(gate.field_core, "fit_structural_field", fit)
    result = gate._fit_fixed_models(wells, partition)
    assert len(result.leave_one_models) == 4
    assert len(fitted) == 5
    all_ids = set(wells)
    for fold, role in enumerate(fitted[:4]):
        excluded = {
            well_id
            for well_id, assigned in partition.fold_by_well.items()
            if assigned == fold
        }
        assert set(role) == all_ids - excluded
    assert set(fitted[-1]) == all_ids
    assert result.metadata["field_fit_count"] == 5
    assert result.metadata["validation_proposals_per_well"] == 5


def test_leave_one_j_fit_is_invariant_to_values_inside_excluded_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wells, groups = _training_population()
    partition = gate._inner_fold_roles(wells, groups)
    excluded_fold = partition.fold_by_well["w3"]
    captured_runs: list[list[tuple[tuple[str, tuple[float, ...]], ...]]] = []

    def execute(population: dict[str, gate.WellPath]) -> None:
        captured: list[tuple[tuple[str, tuple[float, ...]], ...]] = []

        def fit(training: list, config: object) -> SimpleNamespace:
            captured.append(
                tuple(
                    sorted(
                        (
                            item.well_id,
                            tuple(np.asarray(item.tvt, dtype=float).tolist()),
                        )
                        for item in training
                    )
                )
            )
            return _model()

        monkeypatch.setattr(gate.field_core, "fit_structural_field", fit)
        gate._fit_fixed_models(population, partition)
        captured_runs.append(captured)

    execute(wells)
    changed = copy.deepcopy(wells)
    for well_id, assigned in partition.fold_by_well.items():
        if assigned == excluded_fold:
            assert changed[well_id].truth is not None
            changed[well_id].truth = changed[well_id].truth + 1_000_000.0
    execute(changed)
    first, second = captured_runs
    assert first[excluded_fold] == second[excluded_fold]
    for index in set(range(5)) - {excluded_fold}:
        assert first[index] != second[index]


def test_validation_disk_view_is_suffix_value_and_geology_invariant(
    tmp_path: Path,
) -> None:
    path = tmp_path / "well.csv"
    original = _frame()
    original.to_csv(path, index=False)
    first = gate._read_well_csv(path, "validation")
    changed = original.copy()
    changed.loc[3:, "TVT"] += 1_000_000.0
    changed.loc[3:, "Geology"] = "mutated-secret"
    changed.to_csv(path, index=False)
    second = gate._read_well_csv(path, "validation")
    pd.testing.assert_frame_equal(first, second)
    assert tuple(second.columns) == gate.INFERENCE_COLUMNS


def test_work_proxy_is_inference_only_and_fixed_five_fit_mass(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    path = train / "well.csv"
    original = _frame()
    original.to_csv(path, index=False)
    inventory = pd.DataFrame([{"well": "w", "horizontal_file": path.name}])
    first = gate._work_proxy_well_stats(tmp_path, inventory)
    changed = original.copy()
    changed.loc[3:, "TVT"] += 1_000_000.0
    changed.loc[3:, "Geology"] = "mutated-secret"
    changed.to_csv(path, index=False)
    second = gate._work_proxy_well_stats(tmp_path, inventory)
    assert first == second
    assert first["w"]["fixed_inducing_cells"] >= 1


def test_region_roles_preserve_manifest_training_and_exclude_embargo() -> None:
    manifest = {
        "folds": [
            {
                "fold": 0,
                "training_ids": ["a", "b", "c", "d"],
                "validation_ids": ["v"],
                "embargo_ids": ["e"],
            }
        ],
        "wells": [
            {"well_id": value, "equality_group": value}
            for value in ["a", "b", "c", "d", "v", "e"]
        ],
    }
    audit = SimpleNamespace(region_manifest=manifest)
    training, validation, embargo, _ = gate._outer_roles(audit, "region", 0, 0)
    assert training == ["a", "b", "c", "d"]
    assert validation == ["v"]
    assert embargo == ["e"]
    assert not (set(embargo) & (set(training) | set(validation)))


def test_jackknife_confidence_reuses_persistent_suffix_definition() -> None:
    md = np.arange(7, dtype=float) * 100.0
    support = np.array([False, False, False, True, True, True, True])
    proposals = []
    for addition in (0.0, 1.0, -1.0, 3.0):
        delta = md / 100.0
        delta = delta.copy()
        delta[2] = 1_000_000.0
        delta[-1] += addition
        proposals.append(_proposal(delta, support))
    confidence = gate.jackknife_confidence(md, proposals, 1.0, 3)
    assert np.all(confidence[:3] == 0.0)
    assert confidence[3] == pytest.approx(1.0)
    assert np.all(confidence[3:] > 0.0)

    bad_support = np.array([False, False, False, True, False, True, True])
    with pytest.raises(gate.GateError, match="reactivated"):
        gate.jackknife_confidence(
            md, [_proposal(md, bad_support) for _ in range(4)], 1.0, 3
        )


def test_candidate_is_identity_field_over_exact_joint_and_base_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well = _well("v", truth=False)
    final_model = _model(sigma=2.0)
    leave_one = tuple(_model() for _ in range(4))
    seen_policies: list[np.ndarray] = []

    def predict(model: object, path: gate.WellPath) -> SimpleNamespace:
        seen_policies.append(path.base_full.copy())
        support = np.array([False, False, False, True, True, True, True])
        if model is final_model:
            return _proposal(
                np.arange(7, dtype=float) * 0.1,
                support,
                confidence=np.full(7, 0.5),
                bias=np.full(7, 0.2),
            )
        return _proposal(np.arange(7, dtype=float) * 0.1, support)

    monkeypatch.setattr(gate, "_predict_core", predict)
    arrays, _ = gate._predict_candidate(final_model, leave_one, well)
    assert len(seen_policies) == 5
    assert all(np.array_equal(policy, well.base_full) for policy in seen_policies)
    expected = arrays["joint_prediction"] + arrays["field_confidence"] * (
        arrays["field_delta_without_prefix_bias"] + arrays["prefix_bias_delta"]
    )
    assert np.array_equal(arrays["candidate_prediction"], expected)
    assert np.array_equal(arrays["base_prediction"], _suffix().base)
    assert np.array_equal(arrays["joint_prediction"], _suffix().joint)


def test_inference_geometry_mutation_can_change_candidate_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_model = _model()
    leave_one = tuple(_model() for _ in range(4))

    def predict(model: object, path: gate.WellPath) -> SimpleNamespace:
        support = np.array([False, False, False, True, True, True, True])
        scale = float(path.x[-1] - path.x[0]) / 600.0
        return _proposal(scale * np.arange(7, dtype=float), support)

    monkeypatch.setattr(gate, "_predict_core", predict)
    first = _well("v", truth=False)
    second = _well("v", truth=False)
    second.x = second.x * 2.0
    first_arrays, _ = gate._predict_candidate(final_model, leave_one, first)
    second_arrays, _ = gate._predict_candidate(final_model, leave_one, second)
    assert not np.array_equal(
        first_arrays["candidate_prediction"], second_arrays["candidate_prediction"]
    )


def test_run_path_fits_exact_roles_predicts_only_validation_and_never_embargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_ids = [f"train_{index}" for index in range(8)]
    validation_ids = ["validation_0", "validation_1"]
    embargo_ids = ["embargo_never"]
    groups = {
        well_id: f"group_{index}"
        for index, well_id in enumerate([*training_ids, *validation_ids, *embargo_ids])
    }
    results = tmp_path / "results"
    manifests = tmp_path / "manifests"
    output = tmp_path / "v2_shards"
    results.mkdir()
    manifests.mkdir()
    audit = SimpleNamespace(
        results_dir=results,
        manifest_dir=manifests,
        protocol_path=tmp_path / "v2_protocol.json",
        protocol_sha256="a" * 64,
        protocol={
            "incumbent_pretruth_inventory": {"inventory_sha256": "b" * 64},
            "parent_v1_lineage": {"lineage_sha256": "c" * 64},
        },
    )
    requested_roles: list[list[str]] = []
    loaded_roles: list[tuple[str, list[str]]] = []
    fit_roles: list[tuple[str, ...]] = []
    prediction_wells: list[str] = []

    monkeypatch.setattr(
        gate,
        "_outer_roles",
        lambda *_: (training_ids, validation_ids, embargo_ids, groups),
    )

    def load_suffixes(
        _: object, __: str, ___: int, requested: list[str]
    ) -> dict[str, gate.IncumbentSuffix]:
        requested_roles.append(list(requested))
        return {well_id: _suffix() for well_id in requested}

    monkeypatch.setattr(gate, "_load_incumbent_suffixes", load_suffixes)

    def load_wells(
        _: object,
        well_ids: list[str],
        __: dict[str, gate.IncumbentSuffix],
        role: str,
    ) -> dict[str, gate.WellPath]:
        loaded_roles.append((role, list(well_ids)))
        assert not (set(well_ids) & set(embargo_ids))
        return {
            well_id: _well(
                well_id,
                truth=role == "training",
                x_shift=10_000.0 * index,
            )
            for index, well_id in enumerate(well_ids)
        }

    monkeypatch.setattr(gate, "_load_role_wells", load_wells)

    def fit(training: list, _: object) -> SimpleNamespace:
        fit_roles.append(tuple(sorted(item.well_id for item in training)))
        model = _model()
        model.diagnostics.wells = len(training)
        return model

    monkeypatch.setattr(gate.field_core, "fit_structural_field", fit)

    def predict(_: object, well: gate.WellPath) -> SimpleNamespace:
        prediction_wells.append(well.well_id)
        support = np.array([False, False, False, True, True, True, True])
        return _proposal(np.arange(7, dtype=float) * 0.1, support)

    monkeypatch.setattr(gate, "_predict_core", predict)
    monkeypatch.setattr(gate, "_validate_field_shard", lambda *args, **kwargs: {})

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("forbidden legacy learning path was reached")

    monkeypatch.setattr(gate.legacy, "_crossfit_incumbent_training", forbidden)
    monkeypatch.setattr(gate.legacy, "_fit_field_grid", forbidden)
    monkeypatch.setattr(gate.legacy.incumbent_exact, "_evidence_for_records", forbidden)
    monkeypatch.setattr(gate.legacy.incumbent_exact, "_fit_joint_correction", forbidden)
    monkeypatch.setattr(gate.legacy.incumbent_exact.lgb, "LGBMRegressor", forbidden)

    shard_path = gate._run_one_fold(
        audit,
        "region",
        0,
        0,
        output,
        False,
        tmp_path / "benchmark.json",
        "d" * 64,
    )
    assert requested_roles == [[*training_ids, *validation_ids]]
    assert loaded_roles == [
        ("training", training_ids),
        ("validation", validation_ids),
    ]
    assert len(fit_roles) == 5
    assert set(fit_roles[-1]) == set(training_ids)
    assert prediction_wells == [well_id for well_id in validation_ids for _ in range(5)]
    assert not (set(prediction_wells) & (set(training_ids) | set(embargo_ids)))

    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    gate._validate_field_shard_schema(shard)
    extra = copy.deepcopy(shard)
    extra["unexpected"] = 1
    with pytest.raises(gate.GateError, match="schema"):
        gate._validate_field_shard_schema(extra)
    sensitive = copy.deepcopy(shard)
    sensitive["candidate_rmse"] = 1.0
    with pytest.raises(gate.GateError, match="metric-silent"):
        gate._validate_field_shard_schema(sensitive)


def _sealed_arrays() -> dict[str, np.ndarray]:
    confidence = np.full(4, 0.5)
    delta = np.arange(4, dtype=float) * 0.1
    bias = np.full(4, 0.2)
    joint = _suffix().joint.copy()
    return {
        "well_index": np.zeros(4, dtype=np.int32),
        "row_index": _suffix().row_index.astype(np.int32),
        "base_prediction": _suffix().base.copy(),
        "joint_prediction": joint,
        "candidate_prediction": joint + confidence * (delta + bias),
        "field_confidence": confidence,
        "field_delta_without_prefix_bias": delta,
        "prefix_bias_delta": bias,
    }


def test_prediction_schema_and_comparator_are_exact() -> None:
    arrays = _sealed_arrays()
    gate._validate_prediction_arrays(arrays, ["v"], {"v": _suffix()})

    extra = dict(arrays)
    extra["hidden_metric"] = np.zeros(4)
    with pytest.raises(gate.GateError, match="schema"):
        gate._validate_prediction_arrays(extra, ["v"], {"v": _suffix()})

    changed = {name: value.copy() for name, value in arrays.items()}
    changed["base_prediction"][0] += 1.0e-12
    with pytest.raises(gate.GateError, match="role-exact incumbent"):
        gate._validate_prediction_arrays(changed, ["v"], {"v": _suffix()})

    formula = {name: value.copy() for name, value in arrays.items()}
    formula["candidate_prediction"][0] += 1.0e-9
    with pytest.raises(gate.GateError, match="formula"):
        gate._validate_prediction_arrays(formula, ["v"], {"v": _suffix()})


def _proxy() -> dict:
    rows = []
    for mode, repeat, fold in gate._all_fold_identities():
        leave_observations = [75, 75, 75, 75]
        final_observations = 100
        derivative_mass = sum(leave_observations) + final_observations
        leave_nodes = [20, 20, 20, 20]
        final_nodes = 25
        node_mass = sum(leave_nodes) + final_nodes
        queries = 50 + fold + (100 if mode == "region" else repeat)
        five_queries = 5 * queries
        units = 8 * derivative_mass + 800 * node_mass + 4_096 * five_queries
        identity = f"{mode}:{repeat}:{fold}"
        rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "training_wells": 20,
                "validation_wells": 5,
                "embargo_wells": 1 if mode == "region" else 0,
                "inner_role_partition_sha256": gate._canonical_digest(
                    [identity, "partition"]
                ),
                "leave_one_training_wells": [15, 15, 15, 15],
                "leave_one_role_sha256": [
                    gate._canonical_digest([identity, "leave", inner])
                    for inner in range(4)
                ],
                "final_role_sha256": gate._canonical_digest([identity, "final"]),
                "leave_one_derivative_observations": leave_observations,
                "final_derivative_observations": final_observations,
                "derivative_fit_mass": derivative_mass,
                "leave_one_fixed_node_count": leave_nodes,
                "final_fixed_node_count": final_nodes,
                "fixed_node_fit_mass": node_mass,
                "validation_support_queries": queries,
                "validation_suffix_rows": 20,
                "five_validation_proposal_queries": five_queries,
                "proxy_units": units,
            }
        )
    maximizing = max(rows, key=gate._work_proxy_order_key)
    payload = {
        "definition": gate._work_proxy_definition(),
        "folds": rows,
        "maximizing_identity": {
            "mode": maximizing["mode"],
            "repeat": maximizing["repeat"],
            "fold": maximizing["fold"],
        },
    }
    return {**payload, "proxy_sha256": gate._canonical_digest(payload)}


def _benchmark(proxy: dict) -> dict:
    row = max(proxy["folds"], key=gate._work_proxy_order_key)
    total = 100.0
    timing = {
        "initial_live_audit": 2.0,
        "field": 80.0,
        "final_live_audit": 3.0,
        "total": total,
        "extrapolated_two_worker_fifteen_fold": total * 8,
    }
    shape = {
        "lineage_audits": 2,
        "leave_one_field_fits": 4,
        "final_field_fits": 1,
        "field_fits_total": 5,
        "validation_proposals_per_well": 5,
        "prediction_rows": row["validation_suffix_rows"],
        "solver_caps_changed": False,
        "coarsening_allowed": False,
        "support_query_truncation_count": 0,
        "model_observations": [
            *[
                {
                    "role": f"leave_one_{inner}",
                    "fitting_well_count": row["leave_one_training_wells"][inner],
                    "fitting_ids_sha256": row["leave_one_role_sha256"][inner],
                    "requested_inducing_cell_ft": gate.FIXED_INDUCING_CELL_FT,
                    "actual_inducing_cell_ft": gate.FIXED_INDUCING_CELL_FT,
                    "solver_stop_codes": [1, 1],
                }
                for inner in range(4)
            ],
            {
                "role": "final",
                "fitting_well_count": row["training_wells"],
                "fitting_ids_sha256": row["final_role_sha256"],
                "requested_inducing_cell_ft": gate.FIXED_INDUCING_CELL_FT,
                "actual_inducing_cell_ft": gate.FIXED_INDUCING_CELL_FT,
                "solver_stop_codes": [1, 1],
            },
        ],
    }
    acceptance = gate._benchmark_acceptance(
        timing, {"field_peak_rss": 1.0, "total_peak_rss": 2.0}, shape
    )
    return {
        "status": gate.BENCHMARK_STATUS,
        "status_ceiling": gate.STATUS_CEILING,
        "method": gate.METHOD,
        "protocol_sha256": "a" * 64,
        "benchmark_work_proxy_sha256": proxy["proxy_sha256"],
        "worst_work_identity": {
            "mode": row["mode"],
            "repeat": row["repeat"],
            "fold": row["fold"],
            "training_wells": row["training_wells"],
            "validation_wells": row["validation_wells"],
            "embargo_wells": row["embargo_wells"],
            "proxy_units": row["proxy_units"],
        },
        "work_shape": shape,
        "fixed_configuration": gate.fixed_configuration(),
        "timing_seconds": timing,
        "memory_gib": {"field_peak_rss": 1.0, "total_peak_rss": 2.0},
        "acceptance": acceptance,
        "all_acceptance_pass": True,
        "validation_metrics": "withheld; validation TVT was not parsed",
    }


def test_benchmark_requires_two_audits_exact_shape_and_recomputed_acceptance() -> None:
    proxy = _proxy()
    gate._validate_work_proxy_payload(proxy)
    audit = SimpleNamespace(
        protocol={"benchmark_work_proxy": proxy}, protocol_sha256="a" * 64
    )
    payload = _benchmark(proxy)
    gate._validate_benchmark_payload(payload, audit)

    for mutation in (
        lambda value: value["work_shape"].__setitem__("field_fits_total", 4),
        lambda value: value["work_shape"].__setitem__("lineage_audits", 1),
        lambda value: value["work_shape"].__setitem__("prediction_rows", 19),
        lambda value: value["work_shape"].__setitem__(
            "support_query_truncation_count", 1
        ),
        lambda value: value["work_shape"]["model_observations"][0].__setitem__(
            "actual_inducing_cell_ft", 7_501.0
        ),
        lambda value: value["work_shape"]["model_observations"][0].__setitem__(
            "solver_stop_codes", [1, 3]
        ),
        lambda value: value["work_shape"]["model_observations"][0].__setitem__(
            "fitting_ids_sha256", "f" * 64
        ),
        lambda value: value["fixed_configuration"].__setitem__("theta_field", 0.5),
        lambda value: value["fixed_configuration"][
            "effective_field_config"
        ].__setitem__("max_nodes", 1_999),
        lambda value: value["timing_seconds"].__setitem__("total", 99.0),
        lambda value: value["timing_seconds"].__setitem__("final_live_audit", 30.0),
        lambda value: value["acceptance"].__setitem__("total_wall", False),
        lambda value: value["worst_work_identity"].__setitem__("fold", 0),
    ):
        forged = copy.deepcopy(payload)
        mutation(forged)
        with pytest.raises(gate.GateError):
            gate._validate_benchmark_payload(forged, audit)


def test_work_proxy_recomputes_role_mass_query_mass_and_identity() -> None:
    proxy = _proxy()
    gate._validate_work_proxy_payload(proxy)
    for mutate in (
        lambda row: row.__setitem__(
            "derivative_fit_mass", row["derivative_fit_mass"] + 1
        ),
        lambda row: row.__setitem__(
            "fixed_node_fit_mass", row["fixed_node_fit_mass"] + 1
        ),
        lambda row: row.__setitem__(
            "five_validation_proposal_queries",
            row["five_validation_proposal_queries"] + 1,
        ),
        lambda row: row.__setitem__("proxy_units", row["proxy_units"] + 1),
        lambda row: row["leave_one_training_wells"].__setitem__(
            0, row["training_wells"]
        ),
    ):
        forged = copy.deepcopy(proxy)
        mutate(forged["folds"][0])
        payload = {key: value for key, value in forged.items() if key != "proxy_sha256"}
        forged["proxy_sha256"] = gate._canonical_digest(payload)
        with pytest.raises(gate.GateError):
            gate._validate_work_proxy_payload(forged)


def test_benchmark_thresholds_are_v2_limits() -> None:
    thresholds = gate._evaluation_contract()["runtime_acceptance"]
    assert thresholds == {
        "field_wall_seconds_at_most": 600.0,
        "field_peak_rss_gib_at_most": 6.0,
        "total_wall_seconds_at_most": 900.0,
        "total_peak_rss_gib_at_most": 10.0,
        "extrapolated_two_worker_fifteen_fold_seconds_at_most": 7_200.0,
        "extrapolation": "measured worst-fold total including two live audits multiplied by ceil(15/2)",
        "caps_solver_or_coarsening": "forbidden; inducing coarsening is STOP",
    }


def test_parent_lineage_binds_v1_protocol_and_stop_record() -> None:
    lineage = gate._parent_lineage(gate.ROOT / "research/results")
    assert lineage["v1_stop"]["byte_sha256"] == gate.V1_STOP_SHA256
    assert lineage["v1_stop_reason_code"] == gate.V1_STOP_REASON_CODE
    assert lineage["v1_protocol"]["name"] == gate.V1_PROTOCOL_NAME
    assert (
        lineage["fresh_method_nonmutation_statement"] == gate.V1_NONMUTATION_STATEMENT
    )
    assert lineage["lineage_sha256"] == gate._canonical_digest(
        {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    )
    assert lineage["v1_source_and_test_sha256"] == {
        name: gate.V1_SOURCE_SHA256[name] for name in sorted(gate.V1_SOURCE_SHA256)
    }
    assert len(lineage["v1_source_and_test_sha256"]) == 6


def test_parent_artifact_accepts_uppercase_sidecar_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = gate.sha256_file(artifact)
    sidecar = gate._sha_sidecar(artifact)
    sidecar.write_text(f"{digest.upper()}  {artifact.name}\n", encoding="ascii")
    binding = gate._artifact_binding(artifact)
    assert binding["byte_sha256"] == digest
    changed = ("0" if digest[0] != "0" else "1") + digest[1:]
    sidecar.write_text(f"{changed.upper()}  {artifact.name}\n", encoding="ascii")
    with pytest.raises(gate.GateError, match="sidecar drift"):
        gate._artifact_binding(artifact)


def test_output_targets_cannot_alias_parent_or_incumbent_artifacts(
    tmp_path: Path,
) -> None:
    reserved = tmp_path / "v1.json"
    reserved.write_text("{}", encoding="utf-8")
    with pytest.raises(gate.GateError, match="aliases"):
        gate._assert_write_target(reserved, {reserved})
    target = tmp_path / "v2.json"
    gate._assert_write_target(target, {reserved})
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(gate.GateError, match="overwrite"):
        gate._assert_write_target(target, {reserved})


def test_output_tree_separation_rejects_nested_and_ancestor_paths(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    manifests = tmp_path / "manifests"
    results.mkdir()
    manifests.mkdir()
    audit = SimpleNamespace(results_dir=results, manifest_dir=manifests)
    exact = results / gate.EXACT_SHARD_DIR_NAME
    spatial = results / gate.SPATIAL_SHARD_DIR_NAME
    for forbidden in (
        exact,
        exact / "nested_v2",
        spatial / "nested_v2",
        results / "anchored_structural_field_protocol_folds" / "nested_v2",
        tmp_path,
    ):
        with pytest.raises(gate.GateError, match="aliases"):
            gate._assert_output_directory_separate(forbidden, audit)
    gate._assert_output_directory_separate(results / "dedicated_v2_folds", audit)
    for frozen_tree in (
        results / gate.EXACT_SHARD_DIR_NAME,
        results / gate.SPATIAL_SHARD_DIR_NAME,
        results / "anchored_structural_field_protocol_folds",
    ):
        with pytest.raises(gate.GateError, match="reserved tree"):
            gate._assert_artifact_outside_frozen_trees(
                frozen_tree / "protocol_or_benchmark.json", results
            )
    gate._assert_artifact_outside_frozen_trees(results / "v2_protocol.json", results)
    with pytest.raises(gate.GateError, match="reserved tree"):
        gate._assert_outside_tree(
            results / "dedicated_v2_folds" / "benchmark.json",
            results / "dedicated_v2_folds",
            "benchmark",
        )


def test_exact_fifteen_shard_directory_rejects_missing_extra_and_hardlinks(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    json_names = {
        gate._field_shard_name(mode, repeat, fold)
        for mode, repeat, fold in gate._all_fold_identities()
    }
    npz_names = {Path(name).with_suffix(".npz").name for name in json_names}
    sidecar_names = {name + ".sha256" for name in json_names}
    expected = json_names | npz_names | sidecar_names
    for name in expected:
        (shard_dir / name).write_bytes(name.encode("ascii"))
    gate._validate_exact_shard_directory(shard_dir)

    missing = shard_dir / sorted(expected)[0]
    missing.unlink()
    with pytest.raises(gate.GateError, match="partial"):
        gate._validate_exact_shard_directory(shard_dir)
    missing.write_bytes(b"restored")

    extra = shard_dir / "pad.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(gate.GateError, match="extra"):
        gate._validate_exact_shard_directory(shard_dir)
    extra.unlink()

    first, second = [shard_dir / name for name in sorted(npz_names)[:2]]
    second.unlink()
    try:
        os.link(first, second)
    except OSError:
        pytest.skip("hard links unavailable in this test environment")
    with pytest.raises(gate.GateError, match="hard-linked"):
        gate._validate_exact_shard_directory(shard_dir)


def test_source_binding_requires_v2_gate_tests_and_future_scorer() -> None:
    required = set(gate.RUNTIME_SOURCE_FILES)
    assert "research/structural_field_gate_v2.py" in required
    assert "research/test_structural_field_gate_v2.py" in required
    assert "research/structural_field_score_v2.py" in required
    assert "research/test_structural_field_score_v2.py" in required
    assert "research/structural_field_gate.py" in required
    assert "research/structural_field.py" in required


def test_all_fold_inventory_is_fresh_fifteen_shards() -> None:
    identities = gate._all_fold_identities()
    assert len(identities) == 15
    assert len(set(identities)) == 15
    assert identities[:10] == [
        ("exact", repeat, fold) for repeat in range(2) for fold in range(5)
    ]
    assert identities[10:] == [("region", 0, fold) for fold in range(5)]


def test_metric_silent_recursive_allowlist_rejects_sensitive_keys() -> None:
    gate._reject_sensitive_field_names({"derivative_residual_scale": 1.0})
    for key in ("truth", "suffix_target", "candidate_rmse", "oracle_value"):
        with pytest.raises(gate.GateError, match="metric-silent"):
            gate._reject_sensitive_field_names({key: 1.0})
