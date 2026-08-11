from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from research import structural_field_gate as gate


def _frame(*, suffix_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(6, dtype=float) * 100.0,
            "X": np.arange(6, dtype=float) * 100.0,
            "Y": np.zeros(6),
            "Z": np.linspace(1_000.0, 995.0, 6),
            "TVT_input": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan],
            "TVT": [10.0, 11.0, 12.0, 13.0 + suffix_shift, 14.0, 15.0],
            "Geology": ["a", "a", "b", "secret", "secret", "secret"],
        }
    )


def _suffix() -> gate.IncumbentSuffix:
    return gate.IncumbentSuffix(
        row_index=np.arange(3, 6, dtype=np.int64),
        base=np.array([12.8, 13.8, 14.8]),
        joint=np.array([13.0, 14.0, 15.0]),
    )


def _well(*, truth: bool = True) -> gate.WellPath:
    frame = _frame().loc[:, list(gate.TRAINING_COLUMNS)]
    return (
        gate._compose_well("w", frame, _suffix(), "training")
        if truth
        else gate._compose_well(
            "w", frame.loc[:, list(gate.INFERENCE_COLUMNS)], _suffix(), "validation"
        )
    )


def test_frozen_grid_and_field_config_are_exact() -> None:
    assert gate.GRID == (
        (5_000.0, 3.0),
        (5_000.0, 0.3),
        (15_000.0, 3.0),
        (15_000.0, 0.3),
        (30_000.0, 3.0),
        (30_000.0, 0.3),
    )
    config = gate.field_config(15_000.0, 3.0)
    assert config.support_length_ft == 15_000.0
    assert config.inducing_cell_ft == 7_500.0
    assert config.graph_max_edge_ft == 22_500.0
    assert config.graph_neighbors == config.interpolation_neighbors == 6
    assert config.min_effective_wells == 1.5
    assert config.min_directional_observability == 0.05
    assert config.max_distinct_support_wells == 16
    assert config.max_support_neighbors == 4_096
    with pytest.raises(gate.GateError):
        gate.field_config(10_000.0, 0.3)


def test_theta_solver_recovers_interior_and_enforces_triangle() -> None:
    first = np.array([1.0, -2.0, 3.0, 0.5, 4.0])
    second = np.array([-1.0, 0.5, 2.0, 3.0, -0.25])
    residual = 0.7 * first + 0.3 * second
    theta_field, theta_bias, objective = gate.solve_theta(residual, first, second)
    assert theta_field == pytest.approx(0.7)
    assert theta_bias == pytest.approx(0.3)
    assert objective == pytest.approx(0.0, abs=1.0e-12)

    theta_field, theta_bias, _ = gate.solve_theta(
        0.1 * first + 2.0 * second, first, second
    )
    assert 0.0 <= theta_bias <= theta_field <= 1.0


def test_prefix_policy_convention_and_contiguous_suffix() -> None:
    well = _well()
    assert np.array_equal(well.base_full[:3], np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(well.joint_full[:3], np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(well.base_full[3:], _suffix().base)
    bad = gate.IncumbentSuffix(
        row_index=np.array([3, 5, 4]), base=_suffix().base, joint=_suffix().joint
    )
    with pytest.raises(gate.GateError, match="not contiguous"):
        gate._compose_well("w", _frame(), bad, "training")


def test_validation_disk_view_is_suffix_and_surface_mutation_invariant(
    tmp_path: Path,
) -> None:
    path = tmp_path / "well.csv"
    original = _frame()
    original.to_csv(path, index=False)
    first = gate._read_well_csv(path, "validation")
    mutated = original.copy()
    mutated.loc[3:, "TVT"] += 1_000_000.0
    mutated.loc[3:, "Geology"] = "mutated-secret"
    mutated.to_csv(path, index=False)
    second = gate._read_well_csv(path, "validation")
    pd.testing.assert_frame_equal(first, second)
    assert "TVT" not in second and "Geology" not in second


def test_benchmark_work_proxy_is_suffix_truth_and_surface_invariant(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train"
    train.mkdir()
    path = train / "well.csv"
    original = _frame()
    original.to_csv(path, index=False)
    inventory = pd.DataFrame([{"well": "w", "horizontal_file": path.name}])
    first = gate._work_proxy_well_stats(tmp_path, inventory)
    mutated = original.copy()
    mutated.loc[3:, "TVT"] += 1_000_000.0
    mutated.loc[3:, "Geology"] = "mutated-secret"
    mutated.to_csv(path, index=False)
    second = gate._work_proxy_well_stats(tmp_path, inventory)
    assert first == second


def test_role_isolation_and_true_field_fit_target(tmp_path: Path) -> None:
    path = tmp_path / "well.csv"
    _frame().to_csv(path, index=False)
    assert "TVT" in gate._read_well_csv(path, "training")
    assert "TVT" not in gate._read_well_csv(path, "validation")

    well = _well()
    training = gate._as_training_wells([well])[0]
    assert np.array_equal(np.asarray(training.tvt), well.truth)
    mutated = _well()
    assert mutated.truth is not None
    mutated.truth = mutated.truth.copy()
    mutated.truth[4] += 500.0
    changed = gate._as_training_wells([mutated])[0]
    assert not np.array_equal(np.asarray(training.tvt), np.asarray(changed.tvt))
    with pytest.raises(gate.GateError, match="lacks labeled TVT"):
        gate._as_training_wells([_well(truth=False)])


def _proposal(
    delta: np.ndarray, support: np.ndarray, confidence: np.ndarray | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        field_delta_without_prefix_bias_tvt=np.asarray(delta, dtype=float),
        prefix_bias_delta_tvt=np.zeros(len(delta), dtype=float),
        support_mask=np.asarray(support, dtype=bool),
        confidence=(
            np.asarray(confidence, dtype=float)
            if confidence is not None
            else np.asarray(support, dtype=float)
        ),
    )


def test_jackknife_suffix_endpoint_and_no_reactivation() -> None:
    md = np.arange(7, dtype=float) * 100.0
    support = np.array([False, False, True, True, True, True, True])
    proposals = []
    final_additions = (0.0, 1.0, -1.0, 3.0)
    for index, final_addition in enumerate(final_additions):
        delta = md / 100.0
        # Prefix discontinuities differ radically, but the suffix-domain
        # forward derivative at the anchor is identical and must ignore them.
        delta = delta.copy()
        delta[1] = 10_000.0 * (index + 1)
        # Exercise the one-sided TD derivative with a nonlinear last segment.
        delta[-1] += final_addition
        proposals.append(_proposal(delta, support))
    confidence = gate.jackknife_confidence(md, proposals, sigma=1.0, suffix_start=2)
    assert np.all(confidence[:2] == 0.0)
    assert confidence[2] == pytest.approx(1.0)
    assert np.all(confidence[2:] > 0.0)
    td_derivatives = np.array([1.0 + value / 100.0 for value in final_additions])
    td_median = np.median(td_derivatives)
    td_tau = 1.4826 * np.median(np.abs(td_derivatives - td_median))
    assert confidence[-1] == pytest.approx(1.0 / (1.0 + td_tau**2))

    only_two = [
        proposals[0],
        proposals[1],
        _proposal(np.zeros(7), np.zeros(7, dtype=bool)),
        _proposal(np.zeros(7), np.zeros(7, dtype=bool)),
    ]
    assert np.all(gate.jackknife_confidence(md, only_two, 1.0, 2) == 0.0)


def test_jackknife_rejects_nonpersistent_support() -> None:
    md = np.arange(6, dtype=float)
    bad = np.array([False, False, True, False, True, True])
    proposals = [_proposal(md, bad) for _ in range(4)]
    with pytest.raises(gate.GateError, match="reactivated"):
        gate.jackknife_confidence(md, proposals, 1.0, 2)


def test_grid_uses_ten_strict_field_roles_per_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wells = {}
    fold_by_well = {}
    for fold in range(4):
        well = _well()
        well.well_id = f"w{fold}"
        wells[well.well_id] = well
        fold_by_well[well.well_id] = fold
    by_fold = {fold: wells for fold in range(4)}
    fitted_roles: list[tuple[str, ...]] = []
    prediction_routes: list[tuple[tuple[int, ...], str]] = []

    def fit(training: list, config: object) -> SimpleNamespace:
        role = tuple(sorted(item.well_id for item in training))
        fitted_roles.append(role)
        excluded = tuple(sorted(set(range(4)) - {fold_by_well[item] for item in role}))
        return SimpleNamespace(
            role=role,
            excluded_folds=excluded,
            config=config,
            diagnostics=SimpleNamespace(
                actual_inducing_cell_ft=config.inducing_cell_ft,
                derivative_residual_scale=1.0,
            ),
        )

    def predict(model: SimpleNamespace, well: gate.WellPath) -> SimpleNamespace:
        prediction_routes.append((model.excluded_folds, well.well_id))
        slope = 1.0 + 0.01 * len(model.role)
        support = np.array([False, False, False, True, True, True])
        return _proposal(slope * well.md / 100.0, support)

    monkeypatch.setattr(gate.field_core, "fit_structural_field", fit)
    monkeypatch.setattr(gate, "_predict_core", predict)
    selected, metadata = gate._fit_field_grid(by_fold, fold_by_well)
    assert metadata["inner_field_fits"] == 60
    assert metadata["strict_models_per_grid_cell"] == 10
    assert len(selected.inner_models) == 4
    for start in range(0, len(fitted_roles), 10):
        cell = fitted_roles[start : start + 10]
        assert sorted(map(len, cell)) == [2] * 6 + [3] * 4
    assert len(prediction_routes) == len(gate.GRID) * 4 * 4
    assert all(
        fold_by_well[well_id] in excluded for excluded, well_id in prediction_routes
    )


def test_inducing_coarsening_is_hold() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(inducing_cell_ft=2_500.0),
        diagnostics=SimpleNamespace(actual_inducing_cell_ft=5_000.0),
    )
    with pytest.raises(gate.GateHold, match="coarsening"):
        gate._assert_no_coarsening(model)


def test_support_query_truncation_is_counted_and_stops() -> None:
    support_xy = np.vstack(
        (np.zeros((4_097, 2), dtype=float), np.full((15, 2), 20_000.0))
    )
    well_index = np.concatenate(
        (np.zeros(4_097, dtype=np.int64), np.arange(1, 16, dtype=np.int64))
    )
    model = SimpleNamespace(
        config=gate.field_config(5_000.0, 3.0),
        support_xy=support_xy,
        support_tree=cKDTree(support_xy),
        support_well_index=well_index,
    )
    well = _well(truth=False)
    assert gate._support_query_truncation_count(model, well) > 0
    with pytest.raises(gate.GateHold, match="support-query truncation"):
        gate._predict_core(model, well)


def test_crossfit_heldout_target_mutation_does_not_change_its_base_or_joint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"w{i}" for i in range(4)]
    folds = {well_id: index for index, well_id in enumerate(ids)}
    mutation = {"value": 0.0}

    def raw_oof(*_: object) -> tuple[dict, dict, dict, np.ndarray, np.ndarray, dict]:
        raw = {
            well_id: gate._RawBaseRecord(
                well_id=well_id,
                path=f"{well_id}.csv",
                row_index=np.array([2, 3]),
                raw_delta=np.array([1.0 + index, 2.0 + index]),
                target_delta=np.array(
                    [3.0 + index + (mutation["value"] if index == 0 else 0.0), 4.0]
                ),
                anchor_tvt=10.0,
            )
            for index, well_id in enumerate(ids)
        }
        y = np.array([1.0 + (mutation["value"] if i == 0 else 0.0) for i in range(4)])
        return (
            {fold: raw for fold in range(4)},
            folds,
            {fold: np.ones(4) for fold in range(4)},
            y,
            np.asarray(ids),
            {"base_model_count": 10},
        )

    monkeypatch.setattr(gate, "_raw_base_oof", raw_oof)
    monkeypatch.setattr(gate.incumbent_exact, "calibrate_shrink", lambda a, b: 0.5)
    monkeypatch.setattr(
        gate.incumbent_exact,
        "_evidence_for_records",
        lambda records: (
            np.arange(1, len(records) + 1, dtype=float),
            [np.ones(len(record.idx)) for record in records],
            {},
        ),
    )
    monkeypatch.setattr(
        gate.incumbent_exact,
        "_scalar_correction",
        lambda dev, target, hold: (dev * 0.1, hold * 0.1, 0.1),
    )
    monkeypatch.setattr(
        gate.incumbent_exact, "_calibrate_vector_shrink", lambda records, values: 0.2
    )
    monkeypatch.setattr(
        gate.incumbent_exact, "_fit_joint_correction", lambda records, a, b: np.ones(2)
    )
    groups = {well_id: f"g{index}" for index, well_id in enumerate(ids)}
    first, _, _ = gate._crossfit_incumbent_training(None, ids, groups)  # type: ignore[arg-type]
    mutation["value"] = 10_000.0
    second, _, _ = gate._crossfit_incumbent_training(None, ids, groups)  # type: ignore[arg-type]
    assert np.array_equal(first[0]["w0"].base, second[0]["w0"].base)
    assert np.array_equal(first[0]["w0"].joint, second[0]["w0"].joint)


def test_ten_model_roles_exclude_heldout_label_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = [f"w{i}" for i in range(8)]
    folds = {well_id: index // 2 for index, well_id in enumerate(ids)}
    mutation = {"value": 0.0}
    fit_roles: list[set[str]] = []
    prediction_routes: list[tuple[tuple[int, ...], str]] = []

    def static_matrix(_: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        y = np.arange(8, dtype=float)
        y[:2] += mutation["value"]
        return (
            pd.DataFrame({"f": np.arange(8, dtype=float)}, index=ids),
            y,
            np.asarray(ids),
        )

    class FakeModel:
        def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FakeModel":
            self.value = float(np.sum(y))
            self.fit_ids = set(map(str, x.index))
            self.excluded_folds = tuple(
                sorted(set(range(4)) - {folds[well_id] for well_id in self.fit_ids})
            )
            fit_roles.append(self.fit_ids)
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            predicted_ids = set(map(str, x.index))
            if len(predicted_ids) == 1 and len(x) > 1:
                prediction_routes.append(
                    (self.excluded_folds, next(iter(predicted_ids)))
                )
            return np.full(len(x), self.value, dtype=float)

    monkeypatch.setattr(gate, "_well_file", lambda audit, wid: tmp_path / f"{wid}.csv")
    monkeypatch.setattr(gate, "_inner_fold_ids", lambda static, groups: folds)
    monkeypatch.setattr(
        gate.incumbent_exact, "_build_static_training_matrix", static_matrix
    )
    monkeypatch.setattr(gate.incumbent_exact, "well_id", lambda path: Path(path).stem)
    monkeypatch.setattr(
        gate.incumbent_exact.lgb, "LGBMRegressor", lambda **kwargs: FakeModel()
    )
    monkeypatch.setattr(gate.incumbent_exact, "frozen_research_params", lambda: {})
    monkeypatch.setattr(
        gate.incumbent_exact,
        "load_well",
        lambda path: {
            "id": Path(path).stem,
            "tvt_prefix": np.array([10.0, 11.0, np.nan, np.nan]),
            "known": np.array([True, True, False, False]),
        },
    )
    monkeypatch.setattr(
        gate.incumbent_exact,
        "point_frame",
        lambda well, stride: (
            pd.DataFrame({"f": [1.0, 2.0]}, index=[well["id"], well["id"]]),
            np.array([2, 3]),
            np.array(
                [
                    2.0 + (mutation["value"] if well["id"] in {"w0", "w1"} else 0.0),
                    3.0,
                ]
            ),
        ),
    )
    groups = {well_id: f"g{well_id}" for well_id in ids}
    first, _, _, _, _, roles = gate._raw_base_oof(None, ids, groups)  # type: ignore[arg-type]
    assert roles["base_model_count"] == 10
    assert len(roles["leave_one_roles"]) == 4
    assert len(roles["leave_two_roles"]) == 6
    assert len(fit_roles) == 10
    assert all(set(first[fold]) == set(ids) for fold in range(4))
    # Check all ten fitted roles, not just the first. Every role must be a union
    # of whole folds -- a role holding part of a fold would leak the rest.
    for role in fit_roles:
        excluded = set(range(4)) - {folds[well_id] for well_id in role}
        assert role == {
            well_id for well_id in ids if folds[well_id] not in excluded
        }
    # Fold 0 is exactly {w0, w1}, so every role excluding it must hold out both.
    fold_zero_roles = [
        role for role in fit_roles if 0 not in {folds[well_id] for well_id in role}
    ]
    assert len(fold_zero_roles) == 4
    assert all(not ({"w0", "w1"} & role) for role in fold_zero_roles)
    assert len(prediction_routes) == 32
    assert all(folds[well_id] in excluded for excluded, well_id in prediction_routes)

    fit_roles.clear()
    prediction_routes.clear()
    mutation["value"] = 50_000.0
    second, _, _, _, _, _ = gate._raw_base_oof(None, ids, groups)  # type: ignore[arg-type]
    for well_id in ids:
        assert np.array_equal(first[0][well_id].raw_delta, second[0][well_id].raw_delta)


def test_artifact_descriptor_detects_byte_drift(tmp_path: Path) -> None:
    results = tmp_path / "results"
    manifests = tmp_path / "manifests"
    results.mkdir()
    manifests.mkdir()
    path = results / "artifact.json"
    path.write_text("{}\n", encoding="utf-8")
    descriptor = gate._descriptor(path, "results")
    assert gate._validate_descriptor(descriptor, results, manifests) == path
    path.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(gate.GateError, match="drift"):
        gate._validate_descriptor(descriptor, results, manifests)


def _seal_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate._sha_sidecar(path).write_text(
        f"{gate.sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def _work_proxy() -> dict:
    rows = []
    for mode, repeat, fold in gate._all_fold_identities():
        rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "training_wells": 1,
                "validation_wells": 1,
                "embargo_wells": 0,
                "derivative_observations": 1,
                "inducing_node_upper_by_h": {
                    "5000": 1,
                    "15000": 1,
                    "30000": 1,
                },
                "support_query_count": 1,
                "base_static_row_work": 1,
                "base_full_path_row_work": 1,
                "proxy_units": (100 if (mode, repeat, fold) == ("exact", 0, 0) else 1),
            }
        )
    payload = {
        "definition": {},
        "folds": rows,
        "maximizing_identity": {"mode": "exact", "repeat": 0, "fold": 0},
    }
    return {**payload, "proxy_sha256": gate._canonical_digest(payload)}


def _benchmark(path: Path, audit: gate.GateAudit, *, accepted: bool = True) -> None:
    if accepted:
        field_seconds, total_seconds = 10.0, 20.0
        field_rss, total_rss = 1.0, 2.0
    else:
        field_seconds, total_seconds = 2_000.0, 4_000.0
        field_rss, total_rss = 9.0, 13.0
    extrapolated = total_seconds * 8
    thresholds = gate._evaluation_contract()["runtime_acceptance"]
    acceptance = {
        "field_wall": field_seconds <= thresholds["field_wall_seconds_at_most"],
        "field_peak_rss": field_rss <= thresholds["field_peak_rss_gib_at_most"],
        "total_wall": total_seconds <= thresholds["total_wall_seconds_at_most"],
        "total_peak_rss": total_rss <= thresholds["total_peak_rss_gib_at_most"],
        "two_worker_fifteen_fold": extrapolated
        <= thresholds["extrapolated_two_worker_fifteen_fold_seconds_at_most"],
        "no_caps_solver_or_coarsening": True,
        "no_support_query_truncation": True,
    }
    proxy = audit.protocol["benchmark_work_proxy"]
    value = {
        "status": "MEASURE_ONLY_TRUTH_QUARANTINED_BENCHMARK",
        "status_ceiling": gate.STATUS_CEILING,
        "method": gate.METHOD,
        "protocol_sha256": audit.protocol_sha256,
        "benchmark_work_proxy_sha256": proxy["proxy_sha256"],
        "worst_work_identity": {
            "mode": "exact",
            "repeat": 0,
            "fold": 0,
            "training_wells": 1,
            "validation_wells": 1,
            "embargo_wells": 0,
            "proxy_units": 100,
        },
        "work_shape": {
            "strict_base_models": 10,
            "leave_one_base_models": 4,
            "leave_two_base_models": 6,
            "grid_cells": 6,
            "strict_inner_field_models_per_cell": 10,
            "retained_outer_jackknife_models": 4,
            "inner_field_fits": 60,
            "selected_outer_refits": 1,
            "prediction_rows": 3,
            "support_query_truncation_count": 0,
            "solver_caps_changed": False,
            "coarsening_allowed": False,
        },
        "selected_cell": {
            "h_ft": 5_000.0,
            "laplacian": 3.0,
            "theta_field": 0.6,
            "theta_bias": 0.2,
        },
        "timing_seconds": {
            "incumbent_crossfit": min(5.0, total_seconds),
            "field": field_seconds,
            "total": total_seconds,
            "extrapolated_two_worker_fifteen_fold": extrapolated,
        },
        "memory_gib": {
            "field_peak_rss": field_rss,
            "total_peak_rss": total_rss,
        },
        "acceptance": acceptance,
        "all_acceptance_pass": all(acceptance.values()),
        "validation_metrics": "withheld; validation TVT was not parsed",
    }
    _seal_json(path, value)


def _synthetic_audit(tmp_path: Path) -> gate.GateAudit:
    exact_manifest = pd.DataFrame(
        [
            {"repeat": 0, "outer_fold": 0, "well": "v", "typewell_profile_hash": "gv"},
            {"repeat": 0, "outer_fold": 1, "well": "t", "typewell_profile_hash": "gt"},
        ]
    )
    inventory = pd.DataFrame(
        [
            {"well": "v", "rows": 5, "prefix_rows": 2, "suffix_rows": 3},
            {"well": "t", "rows": 5, "prefix_rows": 2, "suffix_rows": 3},
        ]
    )
    return gate.GateAudit(
        protocol={
            "incumbent_pretruth_inventory": {"inventory_sha256": "i" * 64},
            "benchmark_work_proxy": _work_proxy(),
        },
        protocol_path=tmp_path / "protocol.json",
        protocol_sha256="p" * 64,
        data_dir=tmp_path,
        results_dir=tmp_path,
        manifest_dir=tmp_path,
        exact_manifest=exact_manifest,
        spatial_inventory=inventory,
        region_manifest={},
    )


def _model_metadata(*, training_wells: int) -> dict:
    return {
        "training_wells": training_wells,
        "resampled_intervals": 10,
        "inducing_nodes": 4,
        "graph_edges": 3,
        "graph_faces": 1,
        "discontinuity_candidates": 0,
        "graph_components_after_cuts": 1,
        "requested_inducing_cell_ft": 2_500.0,
        "actual_inducing_cell_ft": 2_500.0,
        "derivative_residual_scale": 1.0,
        "solver_stop_codes": [0, 0],
    }


def _diagnostic() -> dict:
    return {
        "status": "anchored_field_100ft_knots_with_policy_fallback",
        "evaluation_rows": 5,
        "prefix_rows": 2,
        "suffix_rows": 3,
        "prefix_bias": 0.0,
        "prefix_bias_intervals": 1,
        "mean_training_midpoint_distance_ft": 100.0,
        "max_training_midpoint_distance_ft": 200.0,
        "effective_well_support_mean": 2.0,
        "query_direction_observability_mean": 0.5,
        "cut_edge_crossings": 0,
        "fallback_fraction": 0.0,
        "mean_core_confidence": 0.5,
        "mean_jackknife_confidence": 0.5,
        "mean_final_confidence": 0.25,
        "supported_fraction": 2.0 / 3.0,
    }


def _field_shard(tmp_path: Path) -> tuple[dict, Path, Path, gate.GateAudit]:
    audit = _synthetic_audit(tmp_path)
    benchmark = tmp_path / "benchmark.json"
    _benchmark(benchmark, audit)
    arrays = {
        "well_index": np.zeros(3, dtype=np.int32),
        "row_index": np.arange(2, 5, dtype=np.int32),
        "base_prediction": np.array([1.0, 2.0, 3.0]),
        "joint_prediction": np.array([1.1, 2.1, 3.1]),
        "field_confidence": np.array([0.5, 0.5, 0.0]),
        "field_delta_without_prefix_bias": np.array([1.0, 2.0, 3.0]),
        "prefix_bias_delta": np.array([0.2, 0.3, 0.4]),
    }
    arrays["candidate_prediction"] = arrays["joint_prediction"] + arrays[
        "field_confidence"
    ] * (
        0.6 * arrays["field_delta_without_prefix_bias"]
        + 0.2 * arrays["prefix_bias_delta"]
    )
    prediction = tmp_path / "exact_repeat_0_fold_0.npz"
    np.savez_compressed(prediction, **arrays)
    shard_path = tmp_path / "exact_repeat_0_fold_0.json"
    shard = {
        "status": "MEASURE_ONLY_FIELD_PREDICTIONS_SEALED_TRUTH_UNREAD",
        "status_ceiling": gate.STATUS_CEILING,
        "method": gate.METHOD,
        "protocol_sha256": audit.protocol_sha256,
        "benchmark_file": benchmark.name,
        "benchmark_sha256": gate.sha256_file(benchmark),
        "incumbent_inventory_sha256": "i" * 64,
        "mode": "exact",
        "repeat": 0,
        "fold": 0,
        "outer_role_sha256": {
            "training_ids": gate._id_digest(["t"]),
            "validation_ids": gate._id_digest(["v"]),
            "embargo_ids": gate._id_digest([]),
        },
        "training_well_count": 1,
        "validation_well_count": 1,
        "embargo_well_count": 0,
        "learned_from_outer_training_only": {
            "incumbent_crossfit": {
                "strict_base_crossfit": {
                    "base_model_count": 10,
                    "leave_one_roles": [
                        {
                            "heldout_fold": inner_fold,
                            "fitting_ids_sha256": "a" * 64,
                            "predicted_ids_sha256": "b" * 64,
                        }
                        for inner_fold in range(4)
                    ],
                    "leave_two_roles": [
                        {
                            "excluded_folds": list(pair),
                            "fitting_ids_sha256": "c" * 64,
                            "first_predicted_ids_sha256": "d" * 64,
                            "second_predicted_ids_sha256": "e" * 64,
                        }
                        for pair in itertools.combinations(range(4), 2)
                    ],
                },
                "crossfit_fold_parameters": [
                    {
                        "inner_fold": inner_fold,
                        "calibration_wells": 0,
                        "heldout_wells": 1,
                        "base_shrink": 0.5,
                        "typewell_shrink": 0.5,
                        "ordered_shrink": 0.5,
                        "joint_coefficients": [0.5, 0.5],
                    }
                    for inner_fold in range(4)
                ],
                "path_dependent_evidence_passes": 4,
            },
            "field_grid": {
                "evaluated_grid": list(gate.frozen_grid()),
                "inner_field_fits": 60,
                "strict_models_per_grid_cell": 10,
                "role_contract": {
                    "leave_one_excluded_folds": [[value] for value in range(4)],
                    "leave_two_excluded_folds": [
                        list(pair) for pair in itertools.combinations(range(4), 2)
                    ],
                    "pair_models_shared_only_by_identical_unordered_exclusion_role": True,
                },
                "tie_break": "shorter h, then stronger Laplacian",
            },
            "support_query_truncation_count": 0,
            "selected_field_cell": {
                "h_ft": 5_000.0,
                "laplacian": 3.0,
                "theta_field": 0.6,
                "theta_bias": 0.2,
            },
            "final_field_model": _model_metadata(training_wells=1),
            "jackknife_inner_models": [
                _model_metadata(training_wells=1) for _ in range(4)
            ],
        },
        "validation_diagnostics": {"v": _diagnostic()},
        "prediction_file": prediction.name,
        "prediction_sha256": gate.sha256_file(prediction),
        "prediction_logical_sha256": gate._logical_array_hash(arrays),
        "prediction_rows": 3,
        "prediction_channels": list(gate.PREDICTION_ARRAYS[2:]),
        "validation_wells": [
            {"well": "v", "well_index": 0, "equality_group": "gv", "n_rows": 3}
        ],
        "runtime_seconds": 1.0,
    }
    _seal_json(shard_path, shard)
    return shard, shard_path, benchmark, audit


def _patch_role_exact_incumbent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate,
        "_load_incumbent_suffixes",
        lambda audit, mode, repeat: {
            "v": gate.IncumbentSuffix(
                row_index=np.arange(2, 5, dtype=np.int64),
                base=np.array([1.0, 2.0, 3.0]),
                joint=np.array([1.1, 2.1, 3.1]),
            )
        },
    )


def test_field_shard_exact_schema_formula_and_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_role_exact_incumbent(monkeypatch)
    shard, shard_path, benchmark, audit = _field_shard(tmp_path)
    arrays = gate._validate_field_shard(
        shard, audit, "exact", 0, 0, shard_path, benchmark
    )
    assert set(arrays) == set(gate.PREDICTION_ARRAYS)
    shard_path.write_text(
        shard_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(gate.GateError, match="hash drift"):
        gate._validate_field_shard(shard, audit, "exact", 0, 0, shard_path, benchmark)


def test_field_shard_rejects_npz_key_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_role_exact_incumbent(monkeypatch)
    shard, shard_path, benchmark, audit = _field_shard(tmp_path)
    prediction = tmp_path / shard["prediction_file"]
    with np.load(prediction, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "prefix_bias_delta"
        }
    np.savez_compressed(prediction, **arrays)
    shard["prediction_sha256"] = gate.sha256_file(prediction)
    shard["prediction_logical_sha256"] = gate._logical_array_hash(arrays)
    _seal_json(shard_path, shard)
    with pytest.raises(gate.GateError, match="schema drift"):
        gate._validate_field_shard(shard, audit, "exact", 0, 0, shard_path, benchmark)


def test_field_shard_rejects_consistent_comparator_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_role_exact_incumbent(monkeypatch)
    shard, shard_path, benchmark, audit = _field_shard(tmp_path)
    prediction = tmp_path / shard["prediction_file"]
    with np.load(prediction, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["joint_prediction"] += 5.0
    arrays["candidate_prediction"] += 5.0
    np.savez_compressed(prediction, **arrays)
    shard["prediction_sha256"] = gate.sha256_file(prediction)
    shard["prediction_logical_sha256"] = gate._logical_array_hash(arrays)
    _seal_json(shard_path, shard)
    with pytest.raises(gate.GateError, match="role-exact incumbent"):
        gate._validate_field_shard(shard, audit, "exact", 0, 0, shard_path, benchmark)


def test_field_shard_strict_allowlists_reject_label_and_unknown_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_role_exact_incumbent(monkeypatch)
    original, shard_path, benchmark, audit = _field_shard(tmp_path)
    mutations = (
        (
            "diagnostic target",
            lambda value: value["validation_diagnostics"]["v"].update(target_tvt=[1.0]),
        ),
        (
            "diagnostic labels",
            lambda value: value["validation_diagnostics"]["v"].update(labels=[1.0]),
        ),
        (
            "diagnostic oracle",
            lambda value: value["validation_diagnostics"]["v"].update(oracle_error=0.0),
        ),
        (
            "unknown learned",
            lambda value: value["learned_from_outer_training_only"].update(
                rescue_factor=1.0
            ),
        ),
    )
    for _, mutate in mutations:
        shard = json.loads(json.dumps(original))
        mutate(shard)
        _seal_json(shard_path, shard)
        with pytest.raises(gate.GateError):
            gate._validate_field_shard(
                shard, audit, "exact", 0, 0, shard_path, benchmark
            )


def test_resume_reaudits_existing_shard_and_rejects_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_role_exact_incumbent(monkeypatch)
    _, shard_path, benchmark, audit = _field_shard(tmp_path)
    monkeypatch.setattr(gate, "audit_protocol", lambda *args, **kwargs: audit)
    completed = gate.run_folds(
        audit.protocol_path,
        benchmark,
        tmp_path,
        [("exact", 0, 0)],
        resume=True,
    )
    assert completed == [shard_path]
    shard_path.write_text(
        shard_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(gate.GateError, match="metadata hash drift"):
        gate.run_folds(
            audit.protocol_path,
            benchmark,
            tmp_path,
            [("exact", 0, 0)],
            resume=True,
        )


def test_failed_benchmark_stops_run_lineage(tmp_path: Path) -> None:
    audit = _synthetic_audit(tmp_path)
    benchmark = tmp_path / "benchmark.json"
    _benchmark(benchmark, audit, accepted=False)
    with pytest.raises(gate.GateHold, match="acceptance failed"):
        gate._validate_benchmark(benchmark, audit)


def test_benchmark_recomputes_acceptance_and_binds_worst_proxy(tmp_path: Path) -> None:
    audit = _synthetic_audit(tmp_path)
    benchmark = tmp_path / "benchmark.json"
    _benchmark(benchmark, audit)
    value = json.loads(benchmark.read_text(encoding="utf-8"))
    value["acceptance"]["field_wall"] = False
    value["all_acceptance_pass"] = False
    _seal_json(benchmark, value)
    with pytest.raises(gate.GateError, match="recomputed"):
        gate._validate_benchmark(benchmark, audit)

    _benchmark_value = json.loads(benchmark.read_text(encoding="utf-8"))
    _benchmark_value["acceptance"]["field_wall"] = True
    _benchmark_value["all_acceptance_pass"] = True
    _benchmark_value["worst_work_identity"]["fold"] = 1
    _seal_json(benchmark, _benchmark_value)
    with pytest.raises(gate.GateError, match="worst-work identity"):
        gate._validate_benchmark(benchmark, audit)


def test_aggregate_inventory_output_must_be_separate_from_shards_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.json"
    benchmark = tmp_path / "benchmark.json"
    shard_dir = tmp_path / "folds"
    shard_dir.mkdir()
    monkeypatch.setattr(
        gate,
        "build_pretruth_field_inventory",
        lambda protocol_path, benchmark_path, shard_dir: {"inventory_sha256": "a" * 64},
    )
    with pytest.raises(gate.GateError, match="outside the shard directory"):
        gate.aggregate_barrier(
            protocol, benchmark, shard_dir, shard_dir / "pretruth.json"
        )
    with pytest.raises(gate.GateError, match="aliases a frozen gate artifact"):
        gate.aggregate_barrier(protocol, benchmark, shard_dir, protocol)


def test_bootstraps_are_deterministic_and_region_is_exhaustive() -> None:
    exact = pd.DataFrame(
        [
            {
                "repeat": repeat,
                "typewell_profile_hash": group,
                "n_rows": 10,
                "joint_sse": 100.0,
                "candidate_sse": 64.0,
            }
            for repeat in (0, 1)
            for group in ("a", "b", "c")
        ]
    )
    first = gate.exact_group_bootstrap(exact)
    second = gate.exact_group_bootstrap(exact)
    assert first == second
    assert first["ci95_low_ft"] > 0.0

    region = pd.DataFrame(
        [
            {
                "fold": fold,
                "n_rows": 10,
                "joint_sse": 100.0,
                "candidate_sse": 81.0,
            }
            for fold in range(5)
        ]
    )
    result = gate.exhaustive_region_bootstrap(region)
    assert result["draws"] == 3_125
    assert result["ci95_low_ft"] > 0.0


def test_runtime_source_lineage_covers_executed_helpers() -> None:
    hashes = gate._source_hashes()
    assert set(hashes) == set(gate.RUNTIME_SOURCE_FILES)
    assert hashes["research/structural_field.py"] == gate.CORE_SHA256
    assert hashes["research/structural_field_score.py"] == gate.SCORE_SHA256
    assert hashes["research/test_structural_field_score.py"] == gate.SCORE_TEST_SHA256
    for name in (
        "geosteern/data.py",
        "geosteern/features.py",
        "geosteern/model.py",
        "research/interval_gate.py",
        "research/ordered_transport.py",
        "research/spatial_split.py",
        "research/structural_field_score.py",
        "research/test_structural_field_score.py",
    ):
        assert name in hashes
