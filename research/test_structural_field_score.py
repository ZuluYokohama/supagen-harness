from __future__ import annotations

import json
import os
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from research import structural_field_gate as gate
from research import structural_field_score as score


@dataclass
class SyntheticScoreRun:
    protocol: Path
    benchmark: Path
    shard_dir: Path
    barrier: Path
    output: Path
    audit: gate.GateAudit
    truths: dict[str, np.ndarray]
    audit_calls: list[bool]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _seal(path: Path) -> None:
    sidecar = gate._sha_sidecar(path)
    sidecar.write_text(
        f"{score.sha256_file(path)}  {path.name}\n",
        encoding="ascii",
        newline="\n",
    )


def _work_proxy(wells: int) -> dict[str, Any]:
    rows = []
    for mode, repeat, fold in gate._all_fold_identities():
        rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "training_wells": wells - wells // 5,
                "validation_wells": wells // 5,
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
                "proxy_units": 100 if (mode, repeat, fold) == ("exact", 0, 0) else 1,
            }
        )
    payload = {
        "definition": {},
        "folds": rows,
        "maximizing_identity": {"mode": "exact", "repeat": 0, "fold": 0},
    }
    return {**payload, "proxy_sha256": gate._canonical_digest(payload)}


def _benchmark_payload(
    protocol_sha256: str,
    work_proxy: dict[str, Any],
    wells: int,
) -> dict[str, Any]:
    acceptance = {
        "field_wall": True,
        "field_peak_rss": True,
        "total_wall": True,
        "total_peak_rss": True,
        "two_worker_fifteen_fold": True,
        "no_caps_solver_or_coarsening": True,
        "no_support_query_truncation": True,
    }
    return {
        "status": "MEASURE_ONLY_TRUTH_QUARANTINED_BENCHMARK",
        "status_ceiling": gate.STATUS_CEILING,
        "method": gate.METHOD,
        "protocol_sha256": protocol_sha256,
        "benchmark_work_proxy_sha256": work_proxy["proxy_sha256"],
        "worst_work_identity": {
            "mode": "exact",
            "repeat": 0,
            "fold": 0,
            "training_wells": wells - wells // 5,
            "validation_wells": wells // 5,
            "embargo_wells": 0,
            "proxy_units": 100,
        },
        "work_shape": {
            "grid_cells": 6,
            "strict_base_models": 10,
            "leave_one_base_models": 4,
            "leave_two_base_models": 6,
            "strict_inner_field_models_per_cell": 10,
            "retained_outer_jackknife_models": 4,
            "inner_field_fits": 60,
            "selected_outer_refits": 1,
            "support_query_truncation_count": 0,
            "solver_caps_changed": False,
            "coarsening_allowed": False,
            "prediction_rows": 4 * (wells // 5),
        },
        "selected_cell": {
            "h_ft": 5_000.0,
            "laplacian": 3.0,
            "theta_field": 1.0,
            "theta_bias": 0.0,
        },
        "timing_seconds": {
            "incumbent_crossfit": 1.0,
            "field": 2.0,
            "total": 3.0,
            "extrapolated_two_worker_fifteen_fold": 24.0,
        },
        "memory_gib": {"field_peak_rss": 1.0, "total_peak_rss": 2.0},
        "acceptance": acceptance,
        "all_acceptance_pass": True,
        "validation_metrics": "withheld; validation TVT was not parsed",
    }


def _model_metadata(training_wells: int) -> dict[str, Any]:
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


def _diagnostic() -> dict[str, Any]:
    return {
        "status": "anchored_field_100ft_knots_with_policy_fallback",
        "evaluation_rows": 6,
        "prefix_rows": 2,
        "suffix_rows": 4,
        "prefix_bias": 0.0,
        "prefix_bias_intervals": 1,
        "mean_training_midpoint_distance_ft": 100.0,
        "max_training_midpoint_distance_ft": 200.0,
        "effective_well_support_mean": 2.0,
        "query_direction_observability_mean": 0.5,
        "cut_edge_crossings": 0,
        "fallback_fraction": 0.0,
        "mean_core_confidence": 0.8,
        "mean_jackknife_confidence": 1.0,
        "mean_final_confidence": 0.8,
        "supported_fraction": 1.0,
    }


def _build_synthetic_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wells: int = 20,
    candidate_error: Callable[[str, int, int, int], float] | None = None,
) -> SyntheticScoreRun:
    if wells < 15:
        raise ValueError("synthetic score run needs enough wells for top-10 removal")
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    results_dir = tmp_path / "incumbent"
    manifest_dir = tmp_path / "manifests"
    shard_dir = tmp_path / "candidate_shards"
    for directory in (train_dir, results_dir, manifest_dir, shard_dir):
        directory.mkdir(parents=True, exist_ok=True)

    well_ids = [f"well_{index:03d}" for index in range(wells)]
    truths: dict[str, np.ndarray] = {}
    inventory_rows: list[dict[str, Any]] = []
    for index, well in enumerate(well_ids):
        truth = 100.0 + index * 0.25 + np.arange(6, dtype=float) * 0.1
        horizontal_name = f"{well}.csv"
        horizontal = train_dir / horizontal_name
        pd.DataFrame({"TVT": truth}).to_csv(horizontal, index=False)
        truths[horizontal_name] = truth
        inventory_rows.append(
            {
                "well": well,
                "horizontal_file": horizontal_name,
                "horizontal_sha256": score.sha256_file(horizontal),
                "rows": 6,
                "prefix_rows": 2,
                "suffix_rows": 4,
            }
        )
    spatial_inventory = pd.DataFrame(inventory_rows)

    exact_rows = []
    for repeat in range(2):
        for index, well in enumerate(well_ids):
            exact_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": (index + repeat) % 5,
                    "well": well,
                    "typewell_profile_hash": f"group_{index:03d}",
                }
            )
    exact_manifest = pd.DataFrame(exact_rows)
    region_manifest = {
        "folds": [
            {
                "fold": fold,
                "training_ids": [
                    well for index, well in enumerate(well_ids) if index % 5 != fold
                ],
                "validation_ids": [
                    well for index, well in enumerate(well_ids) if index % 5 == fold
                ],
                "embargo_ids": [],
            }
            for fold in range(5)
        ],
        "wells": [
            {"well_id": well, "equality_group": f"group_{index:03d}"}
            for index, well in enumerate(well_ids)
        ],
    }

    source_hashes = {
        relative: score.sha256_file(gate.ROOT / relative)
        for relative in score.SCORER_SOURCE_FILES
    }
    work_proxy = _work_proxy(wells)
    protocol_payload = {
        "source_sha256": source_hashes,
        "incumbent_pretruth_inventory": {"inventory_sha256": "a" * 64},
        "benchmark_work_proxy": work_proxy,
    }
    protocol = tmp_path / "field_protocol.json"
    _write_json(protocol, protocol_payload)
    _seal(protocol)
    protocol_sha256 = score.sha256_file(protocol)
    audit = gate.GateAudit(
        protocol=protocol_payload,
        protocol_path=protocol,
        protocol_sha256=protocol_sha256,
        data_dir=data_dir,
        results_dir=results_dir,
        manifest_dir=manifest_dir,
        exact_manifest=exact_manifest,
        spatial_inventory=spatial_inventory,
        region_manifest=region_manifest,
    )

    benchmark = tmp_path / "field_benchmark.json"
    _write_json(
        benchmark,
        _benchmark_payload(protocol_sha256, work_proxy, wells),
    )
    _seal(benchmark)

    def error_for(mode: str, repeat: int, fold: int, index: int) -> float:
        if candidate_error is None:
            return 0.5
        return float(candidate_error(mode, repeat, fold, index))

    for mode, repeat, fold in gate._all_fold_identities():
        training_ids, validation_ids, embargo_ids, group_by_well = gate._outer_roles(
            audit, mode, repeat, fold
        )
        blocks: dict[str, list[np.ndarray]] = {
            name: [] for name in gate.PREDICTION_ARRAYS
        }
        metadata = []
        for well_index, well in enumerate(validation_ids):
            index = well_ids.index(well)
            truth = truths[f"{well}.csv"][2:]
            joint = truth + 2.0
            candidate = truth + error_for(mode, repeat, fold, index)
            confidence = np.full(4, 0.8, dtype=np.float64)
            delta = (candidate - joint) / confidence
            blocks["well_index"].append(np.full(4, well_index, dtype=np.int32))
            blocks["row_index"].append(np.arange(2, 6, dtype=np.int32))
            blocks["base_prediction"].append(joint + 0.1)
            blocks["joint_prediction"].append(joint)
            blocks["candidate_prediction"].append(candidate)
            blocks["field_confidence"].append(confidence)
            blocks["field_delta_without_prefix_bias"].append(delta)
            blocks["prefix_bias_delta"].append(np.zeros(4, dtype=np.float64))
            metadata.append(
                {
                    "well": well,
                    "well_index": well_index,
                    "equality_group": group_by_well[well],
                    "n_rows": 4,
                }
            )
        arrays = {name: np.concatenate(value) for name, value in blocks.items()}
        shard_path = shard_dir / gate._field_shard_name(mode, repeat, fold)
        prediction_path = shard_path.with_suffix(".npz")
        np.savez_compressed(prediction_path, **arrays)
        shard = {
            "status": "MEASURE_ONLY_FIELD_PREDICTIONS_SEALED_TRUTH_UNREAD",
            "status_ceiling": gate.STATUS_CEILING,
            "method": gate.METHOD,
            "protocol_sha256": protocol_sha256,
            "benchmark_file": benchmark.name,
            "benchmark_sha256": score.sha256_file(benchmark),
            "incumbent_inventory_sha256": "a" * 64,
            "mode": mode,
            "repeat": repeat,
            "fold": fold,
            "outer_role_sha256": {
                "training_ids": gate._id_digest(training_ids),
                "validation_ids": gate._id_digest(validation_ids),
                "embargo_ids": gate._id_digest(embargo_ids),
            },
            "training_well_count": len(training_ids),
            "validation_well_count": len(validation_ids),
            "embargo_well_count": len(embargo_ids),
            "learned_from_outer_training_only": {
                "incumbent_crossfit": {
                    "strict_base_crossfit": {
                        "base_model_count": 10,
                        "leave_one_roles": [
                            {
                                "heldout_fold": inner_fold,
                                "fitting_ids_sha256": "b" * 64,
                                "predicted_ids_sha256": "c" * 64,
                            }
                            for inner_fold in range(4)
                        ],
                        "leave_two_roles": [
                            {
                                "excluded_folds": list(pair),
                                "fitting_ids_sha256": "d" * 64,
                                "first_predicted_ids_sha256": "e" * 64,
                                "second_predicted_ids_sha256": "f" * 64,
                            }
                            for pair in itertools.combinations(range(4), 2)
                        ],
                    },
                    "crossfit_fold_parameters": [
                        {
                            "inner_fold": inner_fold,
                            "calibration_wells": len(training_ids) - 1,
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
                    "theta_field": 1.0,
                    "theta_bias": 0.0,
                },
                "final_field_model": _model_metadata(len(training_ids)),
                "jackknife_inner_models": [
                    _model_metadata(len(training_ids)) for _ in range(4)
                ],
            },
            "validation_diagnostics": {well: _diagnostic() for well in validation_ids},
            "prediction_file": prediction_path.name,
            "prediction_sha256": score.sha256_file(prediction_path),
            "prediction_logical_sha256": gate._logical_array_hash(arrays),
            "prediction_rows": len(arrays["row_index"]),
            "prediction_channels": list(gate.PREDICTION_ARRAYS[2:]),
            "validation_wells": metadata,
            "runtime_seconds": 0.01,
        }
        _write_json(shard_path, shard)
        _seal(shard_path)

    audit_calls: list[bool] = []

    def fake_audit_protocol(
        path: Path,
        *,
        verify_data: bool = True,
        **_: object,
    ) -> gate.GateAudit:
        audit_calls.append(bool(verify_data))
        if Path(path).resolve() != protocol.resolve() or not verify_data:
            raise gate.GateError("synthetic audit invocation drift")
        if _read_sidecar_for_test(protocol) != score.sha256_file(protocol):
            raise gate.GateError("synthetic protocol drift")
        for row in spatial_inventory.itertuples(index=False):
            horizontal = train_dir / str(row.horizontal_file)
            if score.sha256_file(horizontal) != str(row.horizontal_sha256):
                raise gate.GateError("synthetic data drift")
        return audit

    monkeypatch.setattr(gate, "audit_protocol", fake_audit_protocol)

    def fake_incumbent_suffixes(
        _: gate.GateAudit, mode: str, repeat: int
    ) -> dict[str, gate.IncumbentSuffix]:
        if mode not in {"exact", "region"} or repeat not in {0, 1}:
            raise gate.GateError("synthetic incumbent role drift")
        return {
            well: gate.IncumbentSuffix(
                row_index=np.arange(2, 6, dtype=np.int64),
                base=truths[f"{well}.csv"][2:] + 2.1,
                joint=truths[f"{well}.csv"][2:] + 2.0,
            )
            for well in well_ids
        }

    monkeypatch.setattr(gate, "_load_incumbent_suffixes", fake_incumbent_suffixes)
    inventory = gate.build_pretruth_field_inventory(protocol, benchmark, shard_dir)
    barrier = tmp_path / "field_pretruth_inventory.json"
    _write_json(
        barrier,
        {
            "status": "MEASURE_ONLY_ALL_FIELD_SHARDS_AUDITED_TRUTH_STILL_UNREAD",
            "status_ceiling": gate.STATUS_CEILING,
            "pretruth_field_inventory": inventory,
            "next_phase": "HOLD: synthetic independent score review",
        },
    )
    _seal(barrier)
    return SyntheticScoreRun(
        protocol=protocol,
        benchmark=benchmark,
        shard_dir=shard_dir,
        barrier=barrier,
        output=tmp_path / "field_score.json",
        audit=audit,
        truths=truths,
        audit_calls=audit_calls,
    )


def _read_sidecar_for_test(path: Path) -> str:
    fields = gate._sha_sidecar(path).read_text(encoding="ascii").strip().split()
    return fields[0]


def _install_truth_spy(
    run: SyntheticScoreRun,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate_after_first: Callable[[], None] | None = None,
    nonfinite: bool = False,
) -> list[str]:
    calls: list[str] = []

    def loader(path: Path) -> np.ndarray:
        calls.append(path.name)
        if len(calls) == 1 and mutate_after_first is not None:
            mutate_after_first()
        truth = run.truths[path.name].copy()
        if nonfinite:
            truth[-1] = np.nan
        return truth

    monkeypatch.setattr(score, "_default_truth_loader", loader)
    return calls


def _reseal_prediction(
    run: SyntheticScoreRun,
    identity: tuple[str, int, int],
    mutation: Callable[[dict[str, np.ndarray]], None],
) -> None:
    mode, repeat, fold = identity
    shard_path = run.shard_dir / gate._field_shard_name(mode, repeat, fold)
    prediction_path = shard_path.with_suffix(".npz")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutation(arrays)
    np.savez_compressed(prediction_path, **arrays)
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["prediction_sha256"] = score.sha256_file(prediction_path)
    shard["prediction_logical_sha256"] = gate._logical_array_hash(arrays)
    _write_json(shard_path, shard)
    _seal(shard_path)


def _metric_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, raw in enumerate(records):
        joint_rmse = float(np.sqrt(raw["joint_sse"] / raw["n_rows"]))
        candidate_rmse = float(np.sqrt(raw["candidate_sse"] / raw["n_rows"]))
        rows.append(
            {
                "mode": raw.get("mode", "exact"),
                "repeat": raw.get("repeat", 0),
                "fold": raw.get("fold", index % 5),
                "well": raw.get("well", f"well_{index}"),
                "equality_group": raw.get("equality_group", f"group_{index}"),
                "n_rows": raw["n_rows"],
                "joint_sse": raw["joint_sse"],
                "candidate_sse": raw["candidate_sse"],
                "joint_rmse": joint_rmse,
                "candidate_rmse": candidate_rmse,
                "rmse_gain_ft": joint_rmse - candidate_rmse,
                "mean_field_confidence": 0.5,
                "support_fraction": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=score.SCORED_COLUMNS)


def test_success_seals_once_and_independent_audit_reconstructs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_synthetic_run(tmp_path, monkeypatch)
    calls = _install_truth_spy(run, monkeypatch)

    result = score.score_structural_field(
        run.protocol,
        run.benchmark,
        run.shard_dir,
        run.barrier,
        run.output,
    )

    assert len(calls) == 20  # cached once per bound on-disk well, not per mode
    assert result["status"] == "MEASURE_ONLY"
    assert result["production_open"] is False
    assert result["mode_interaction"]["both_passed"] is True
    assert (
        result["candidate_shard_inventory_sha256"]
        == result["candidate_shard_inventory"]["inventory_sha256"]
    )
    assert result["exact_gate"]["p90_distribution_guard"]["population_records"] == 40
    assert (
        "not q0.90 of paired differences"
        in result["exact_gate"]["p90_distribution_guard"]["definition"]
    )
    assert (
        result["exact_gate"]["descriptive_support_confidence"]["gate_effect"] == "none"
    )
    by_region = result["region_gate"]["descriptive_support_confidence_by_region_fold"]
    assert set(by_region) == {"0", "1", "2", "3", "4"}
    assert all(value["gate_effect"] == "none" for value in by_region.values())
    per_well = score._per_well_path(run.output)
    sealed_rows = pd.read_csv(per_well)
    assert len(sealed_rows) == 60
    assert _read_sidecar_for_test(per_well) == score.sha256_file(per_well)
    assert _read_sidecar_for_test(run.output) == score.sha256_file(run.output)
    truth_calls_before_artifact_audit = len(calls)
    audited = score.audit_score_artifacts(
        run.output,
        run.protocol,
        run.benchmark,
        run.shard_dir,
        run.barrier,
    )
    assert audited["status"] == "MEASURE_ONLY_SCORE_ARTIFACTS_AUDITED"
    assert audited["measured_status"] == "MEASURE_ONLY"
    assert len(calls) == truth_calls_before_artifact_audit

    fractional_rows = sealed_rows.copy()
    fractional_rows["n_rows"] = fractional_rows["n_rows"].astype(float)
    fractional_rows.loc[0, "n_rows"] = 4.5
    with pytest.raises(score.ScoreError, match="positive integers"):
        score._validate_scored_rows(fractional_rows, run.audit)
    invalid_support = sealed_rows.copy()
    invalid_support.loc[0, "support_fraction"] = 1.1
    with pytest.raises(score.ScoreError, match="support fraction"):
        score._validate_scored_rows(invalid_support, run.audit)

    calls.clear()
    with pytest.raises(score.ScoreError, match="overwrite"):
        score.score_structural_field(
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
            run.output,
        )
    assert calls == []

    frame = pd.read_csv(per_well)
    frame.loc[0, "mean_field_confidence"] = 0.1
    frame.to_csv(per_well, index=False, lineterminator="\n")
    _seal(per_well)
    payload = json.loads(run.output.read_text(encoding="utf-8"))
    payload["per_well_artifact"]["byte_sha256"] = score.sha256_file(per_well)
    _write_json(run.output, payload)
    _seal(run.output)
    with pytest.raises(score.ScoreError, match="reconstruct"):
        score.audit_score_artifacts(
            run.output,
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
        )


@pytest.mark.parametrize(
    "attack",
    [
        "missing",
        "pad",
        "tampered",
        "row_identity",
        "comparator",
        "candidate_formula",
        "membership",
        "nonfinite",
        "protocol",
        "benchmark",
        "barrier",
        "manifest",
        "incumbent",
        "source",
        "data",
    ],
)
def test_global_barrier_failures_precede_any_truth_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    run = _build_synthetic_run(tmp_path, monkeypatch)
    identity = gate._all_fold_identities()[0]
    shard_path = run.shard_dir / gate._field_shard_name(*identity)
    if attack == "missing":
        shard_path.unlink()
    elif attack == "pad":
        (run.shard_dir / "pad_out_fold_0.json").write_text("{}\n", encoding="utf-8")
    elif attack == "tampered":
        shard_path.write_text(
            shard_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif attack == "row_identity":
        _reseal_prediction(
            run,
            identity,
            lambda arrays: arrays["row_index"].__setitem__(0, 3),
        )
    elif attack == "comparator":
        _reseal_prediction(
            run,
            identity,
            lambda arrays: arrays["base_prediction"].__setitem__(
                0, arrays["base_prediction"][0] + 1.0e-12
            ),
        )
    elif attack == "candidate_formula":
        _reseal_prediction(
            run,
            identity,
            lambda arrays: arrays["candidate_prediction"].__setitem__(
                0, arrays["candidate_prediction"][0] + 1.0e-9
            ),
        )
    elif attack == "membership":
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        shard["outer_role_sha256"]["validation_ids"] = "0" * 64
        _write_json(shard_path, shard)
        _seal(shard_path)
    elif attack == "nonfinite":
        _reseal_prediction(
            run,
            identity,
            lambda arrays: arrays["candidate_prediction"].__setitem__(0, np.nan),
        )
    elif attack == "protocol":
        run.protocol.write_text(
            run.protocol.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif attack == "benchmark":
        run.benchmark.write_text(
            run.benchmark.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif attack == "barrier":
        run.barrier.write_text(
            run.barrier.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif attack == "manifest":
        run.audit.exact_manifest.loc[0, "typewell_profile_hash"] = "changed_group"
    elif attack == "incumbent":
        run.audit.protocol["incumbent_pretruth_inventory"]["inventory_sha256"] = (
            "1" * 64
        )
    elif attack == "source":
        run.audit.protocol["source_sha256"][score.SCORER_SOURCE_FILES[0]] = "0" * 64
    else:
        first_data = run.audit.data_dir / "train" / next(iter(run.truths))
        first_data.write_text(
            first_data.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    calls = _install_truth_spy(run, monkeypatch)

    with pytest.raises(score.ScoreError):
        score.score_structural_field(
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
            run.output,
        )
    assert calls == []
    assert not run.output.exists()
    assert not score._per_well_path(run.output).exists()


def test_hardlink_alias_is_rejected_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_synthetic_run(tmp_path, monkeypatch)
    first, second = gate._all_fold_identities()[:2]
    source = run.shard_dir / gate._field_shard_name(*first)
    target = run.shard_dir / gate._field_shard_name(*second)
    target.unlink()
    try:
        os.link(source, target)
    except OSError:
        pytest.skip("hard links are unavailable on this test filesystem")
    calls = _install_truth_spy(run, monkeypatch)

    with pytest.raises(score.ScoreError, match="alias"):
        score.score_structural_field(
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
            run.output,
        )
    assert calls == []


def test_output_inside_shard_directory_is_rejected_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _build_synthetic_run(tmp_path, monkeypatch)
    calls = _install_truth_spy(run, monkeypatch)
    unsafe_output = run.shard_dir / "fresh_score.json"

    with pytest.raises(score.ScoreError, match="outside candidate shard"):
        score.score_structural_field(
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
            unsafe_output,
        )
    assert calls == []
    assert not unsafe_output.exists()
    assert not score._per_well_path(unsafe_output).exists()
    with pytest.raises(score.ScoreError, match="aliases sealed lineage"):
        score.score_structural_field(
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
            run.barrier,
        )
    assert calls == []


def test_postscore_drift_and_nonfinite_truth_never_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drift_root = tmp_path / "drift"
    run = _build_synthetic_run(drift_root, monkeypatch)
    first = run.shard_dir / gate._field_shard_name(*gate._all_fold_identities()[0])

    def mutate() -> None:
        first.write_text(first.read_text(encoding="utf-8") + " ", encoding="utf-8")

    calls = _install_truth_spy(run, monkeypatch, mutate_after_first=mutate)
    with pytest.raises(score.ScoreError):
        score.score_structural_field(
            run.protocol,
            run.benchmark,
            run.shard_dir,
            run.barrier,
            run.output,
        )
    assert calls
    assert not run.output.exists()
    assert not score._per_well_path(run.output).exists()

    nan_root = tmp_path / "nan"
    nan_run = _build_synthetic_run(nan_root, monkeypatch)
    nan_calls = _install_truth_spy(nan_run, monkeypatch, nonfinite=True)
    with pytest.raises(score.ScoreError, match="nonfinite"):
        score.score_structural_field(
            nan_run.protocol,
            nan_run.benchmark,
            nan_run.shard_dir,
            nan_run.barrier,
            nan_run.output,
        )
    assert nan_calls
    assert not nan_run.output.exists()


def test_coupled_repeat_primary_is_not_concatenated_pooling() -> None:
    rows = _metric_frame(
        [
            {
                "repeat": 0,
                "well": "well",
                "equality_group": "group",
                "n_rows": 10_000,
                "joint_sse": 1_000_000.0,
                "candidate_sse": 0.0,
            },
            {
                "repeat": 1,
                "well": "well",
                "equality_group": "group",
                "n_rows": 1,
                "joint_sse": 0.0,
                "candidate_sse": 900.0,
            },
        ]
    )

    assert score._pooled_gain(rows) > 9.0
    assert score._mean_repeat_pooled_gain(rows) == pytest.approx(-10.0)
    bootstrap = score._exact_group_bootstrap(rows, draws=100, seed=7)
    assert bootstrap["gain_ci95_low_ft"] == pytest.approx(-10.0)
    assert bootstrap["gain_ci95_high_ft"] == pytest.approx(-10.0)
    assert bootstrap["p90_population"] == "coupled_two_repeat_well_records"


def test_paired_median_pools_repeat_sse_per_well() -> None:
    rows = _metric_frame(
        [
            {
                "repeat": 0,
                "well": "a",
                "equality_group": "ga",
                "n_rows": 1,
                "joint_sse": 100.0,
                "candidate_sse": 0.0,
            },
            {
                "repeat": 1,
                "well": "a",
                "equality_group": "ga",
                "n_rows": 9,
                "joint_sse": 0.0,
                "candidate_sse": 144.0,
            },
            {
                "repeat": 0,
                "well": "b",
                "equality_group": "gb",
                "n_rows": 1,
                "joint_sse": 4.0,
                "candidate_sse": 0.0,
            },
            {
                "repeat": 1,
                "well": "b",
                "equality_group": "gb",
                "n_rows": 9,
                "joint_sse": 0.0,
                "candidate_sse": 9.0,
            },
        ]
    )
    paired = score._paired_exact_wells(rows)
    observed = float(np.median(paired["rmse_gain_ft"]))
    mean_repeat_rmse_then_median = 1.75
    median_over_four_repeat_well_gains = 0.5

    assert observed == pytest.approx(
        np.median([np.sqrt(10.0) - np.sqrt(14.4), np.sqrt(0.4) - np.sqrt(0.9)])
    )
    assert observed != pytest.approx(mean_repeat_rmse_then_median)
    assert observed != pytest.approx(median_over_four_repeat_well_gains)


def test_exhaustive_region_bootstrap_and_replicated_quantile() -> None:
    rows = _metric_frame(
        [
            {
                "mode": "region",
                "repeat": 0,
                "fold": fold,
                "well": f"well_{fold}_{index}",
                "n_rows": 4,
                "joint_sse": 16.0,
                "candidate_sse": 4.0,
            }
            for fold in range(5)
            for index in range(2)
        ]
    )
    result = score._region_exhaustive_bootstrap(rows)
    assert result["draws"] == 5**5
    assert result["gain_ci95_low_ft"] == pytest.approx(1.0)
    assert result["gain_ci95_high_ft"] == pytest.approx(1.0)
    assert result["p90_worsening_one_sided_95_upper_ft"] == pytest.approx(-1.0)

    values = np.array([1.0, 2.0, 9.0])
    multiplicity = np.array([2.0, 0.0, 3.0])
    expected = np.quantile(np.repeat(values, multiplicity.astype(int)), 0.9)
    assert score._replicated_linear_quantile(
        values, multiplicity, 0.9
    ) == pytest.approx(expected)


def test_top10_influence_and_p90_tail_are_independent_guards() -> None:
    influence_rows = _metric_frame(
        [
            {
                "mode": "region",
                "fold": index % 5,
                "well": f"well_{index:02d}",
                "n_rows": 1,
                "joint_sse": 100.0 if index < 10 else 1.0,
                "candidate_sse": 0.0 if index < 10 else 4.0,
            }
            for index in range(20)
        ]
    )
    influence = score._top_positive_removal(influence_rows, exact_repeats=False)
    assert influence["removed_count"] == 10
    assert influence["remaining_gain_ft"] == pytest.approx(-1.0)
    assert influence["passed"] is False

    tail_rows = _metric_frame(
        [
            {
                "mode": "region",
                "fold": index % 5,
                "well": f"tail_{index:02d}",
                "n_rows": 1,
                "joint_sse": 1.0,
                "candidate_sse": 1.0 if index < 15 else 9.0,
            }
            for index in range(20)
        ]
    )
    p90 = score._p90_worsening(tail_rows)
    assert p90["worsening_ft"] == pytest.approx(2.0)
    assert p90["worsening_ft"] > score.P90_POINT_MAX_WORSENING_FT
