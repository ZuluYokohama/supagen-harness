"""Focused deterministic tests for the frozen repeated exact-group gate."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.repeated_group_gate as gate
from research.repeated_group_gate import (
    METHOD,
    OUTER_REPEATS,
    OUTER_SEEDS,
    OUTER_SPLITS,
    ProtocolError,
    _assign_outer_folds,
    _base_manifest,
    _build_fold_artifact_inventory,
    _coefficient_stability,
    _fit_base_fold,
    _frozen_ordered_settings,
    _group_cluster_bootstrap,
    _joint_vs_best_component_bootstrap,
    _load_inference_well,
    _logical_array_hash,
    _persist_aggregate_artifacts,
    _repeat_pooled_gains,
    _run_one_fold,
    _score_sealed_predictions,
    _scored_sse_path,
    _summary,
    _top_positive_sse_removal,
    _validate_manifest,
    _validate_fold_artifact_inventory_payload,
    aggregate_folds,
    audit_protocol,
    freeze_protocol,
    frozen_research_params,
    parse_args,
    sha256_file,
)


def _base_rows(n_wells: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(n_wells):
        # Twenty exact profiles, each shared by two wells.
        group = f"{index // 2:064x}"
        rows.append(
            {
                "well": f"{index:08x}",
                "typewell_profile_hash": group,
                "rows": 100 + index,
                "prefix_rows": 60,
                "suffix_rows": 40 + index,
                "gr_valid_fraction": 0.8,
                "horizontal_sha256": f"{index + 1000:064x}",
                "typewell_sha256": f"{index // 2 + 2000:064x}",
                "horizontal_file": f"{index:08x}__horizontal_well.csv",
                "typewell_file": f"{index:08x}__typewell.csv",
            }
        )
    return pd.DataFrame(rows)


def _write_synthetic_corpus(root: Path, n_wells: int = 12) -> list[Path]:
    train = root / "train"
    train.mkdir(parents=True)
    horizontal_paths = []
    md = np.arange(70.0)
    typewell_tvt = np.arange(60.0)
    for index in range(n_wells):
        wid = f"{index + 0x10000000:08x}"
        # Wells 0 and 1 deliberately share an exact typewell profile.
        profile = 0 if index < 2 else index
        typewell_gr = 70.0 + profile + 8.0 * np.sin(typewell_tvt / 5.0)
        typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
        typewell_path = train / f"{wid}__typewell.csv"
        typewell.to_csv(typewell_path, index=False)

        tvt = 20.0 + 0.05 * md
        tvt_input = tvt.copy()
        tvt_input[55:] = np.nan
        horizontal = pd.DataFrame(
            {
                "MD": md,
                "X": 1000.0 + md,
                "Y": 2000.0 + 0.5 * md,
                "Z": -9000.0 - 0.02 * md,
                "GR": np.interp(tvt, typewell_tvt, typewell_gr),
                "TVT_input": tvt_input,
                "TVT": tvt,
                "EGFDU": -9500.0 + 0.01 * md,
            }
        )
        horizontal_path = train / f"{wid}__horizontal_well.csv"
        horizontal.to_csv(horizontal_path, index=False)
        horizontal_paths.append(horizontal_path)
    return horizontal_paths


def _write_fake_validated_fold_artifacts(
    root: Path,
) -> list[tuple[int, int, Path, dict, dict[str, np.ndarray]]]:
    root.mkdir(parents=True, exist_ok=True)
    validated = []
    for repeat in range(OUTER_REPEATS):
        for fold in range(OUTER_SPLITS):
            shard_path = root / gate._fold_shard_name(repeat, fold)
            prediction_path = gate._prediction_path(shard_path)
            arrays = {
                "row_index": np.array([repeat * OUTER_SPLITS + fold], dtype=np.int32),
                "joint_prediction": np.array([100.0 + repeat + fold], dtype=np.float64),
            }
            gate._atomic_write_npz(prediction_path, arrays)
            shard = {
                "repeat": repeat,
                "outer_fold": fold,
                "prediction_file": prediction_path.name,
                "prediction_sha256": sha256_file(prediction_path),
                "prediction_logical_sha256": gate._logical_array_hash(arrays),
            }
            gate._atomic_write_json(shard_path, shard)
            gate._write_hash_sidecar(shard_path)
            validated.append((repeat, fold, shard_path, shard, arrays))
    return validated


def test_outer_plan_is_deterministic_distinct_and_exact_group_safe() -> None:
    first = _assign_outer_folds(_base_rows())
    second = _assign_outer_folds(_base_rows().sample(frac=1.0, random_state=7))
    _validate_manifest(first)
    pd.testing.assert_frame_equal(first, second)

    assert first.groupby("well").size().eq(OUTER_REPEATS).all()
    assert (
        first.groupby(["repeat", "typewell_profile_hash"])["outer_fold"]
        .nunique()
        .eq(1)
        .all()
    )
    assignments = first.pivot(index="well", columns="repeat", values="outer_fold")
    assert (assignments[0] != assignments[1]).any()
    assert sorted(first["seed"].unique()) == sorted(OUTER_SEEDS)
    assert sorted(first["outer_fold"].unique()) == list(range(OUTER_SPLITS))


def test_freeze_audit_and_one_byte_input_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    horizontal_paths = _write_synthetic_corpus(tmp_path)
    monkeypatch.setattr(gate, "EXPECTED_ELIGIBLE_WELLS", 12)
    monkeypatch.setattr(gate, "EXPECTED_TYPEWELL_GROUPS", 11)
    protocol_path = tmp_path / "artifacts" / "protocol.json"
    protocol, manifest, sidecar = freeze_protocol(tmp_path, protocol_path)

    assert protocol.is_file() and manifest.is_file() and sidecar.is_file()
    bundle = audit_protocol(protocol_path, verify_data=True)
    assert bundle.manifest["well"].nunique() == 12
    assert bundle.manifest["typewell_profile_hash"].nunique() == 11

    original = horizontal_paths[0].read_bytes()
    horizontal_paths[0].write_bytes(original + b"\n")
    with pytest.raises(ProtocolError, match="dataset hash drift"):
        audit_protocol(protocol_path, verify_data=True)
    horizontal_paths[0].write_bytes(original)

    protocol_path.write_bytes(protocol_path.read_bytes() + b" ")
    with pytest.raises(ProtocolError, match="protocol hash drift"):
        audit_protocol(protocol_path, verify_data=False)


def test_freeze_validates_all_source_hashes_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_synthetic_corpus(tmp_path)
    protocol_path = tmp_path / "artifacts" / "protocol.json"

    def fail_source_inventory() -> dict[str, str]:
        raise ProtocolError("required test source is missing")

    monkeypatch.setattr(gate, "_source_hashes", fail_source_inventory)
    with pytest.raises(ProtocolError, match="required test source"):
        freeze_protocol(tmp_path, protocol_path)
    assert not protocol_path.exists()
    assert not gate._manifest_path(protocol_path).exists()
    assert not gate._protocol_sidecar(protocol_path).exists()


def test_freeze_manifest_uses_inference_view_and_never_opens_suffix_truth(
    tmp_path: Path,
) -> None:
    horizontal = _write_synthetic_corpus(tmp_path, n_wells=1)[0]
    full = pd.read_csv(horizontal)
    full.loc[full["TVT_input"].isna(), "TVT"] = np.nan
    full.to_csv(horizontal, index=False)

    manifest = _base_manifest(tmp_path)
    assert len(manifest) == 1
    source = inspect.getsource(_base_manifest)
    assert "load_well" not in source
    assert '["truth"]' not in source


def test_inference_loader_never_reads_truth_or_surfaces(tmp_path: Path) -> None:
    horizontal = _write_synthetic_corpus(tmp_path, n_wells=1)[0]
    well = _load_inference_well(str(horizontal))

    assert well["truth"] is None
    assert list(well["df"].columns) == ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    assert "TVT" not in well["df"] and "EGFDU" not in well["df"]
    assert list(well["tw"].columns) == ["TVT", "GR"]

    # A suffix-truth/surface sentinel mutation must be invisible to the
    # inference view used for every outer-test prediction.
    full = pd.read_csv(horizontal)
    full.loc[full["TVT_input"].isna(), "TVT"] += 1_000_000.0
    full["EGFDU"] -= 2_000_000.0
    full.to_csv(horizontal, index=False)
    mutated = _load_inference_well(str(horizontal))
    pd.testing.assert_frame_equal(mutated["df"], well["df"])
    pd.testing.assert_frame_equal(mutated["tw"], well["tw"])
    np.testing.assert_array_equal(mutated["known"], well["known"])
    np.testing.assert_array_equal(mutated["tail"], well["tail"])


def test_outer_test_rows_are_rejected_from_static_fit_before_lightgbm() -> None:
    matrix = (
        pd.DataFrame({"feature": [1.0, 2.0]}),
        np.array([0.0, 1.0]),
        np.array(["aaaaaaaa", "bbbbbbbb"]),
    )
    with pytest.raises(ProtocolError, match="non-training wells"):
        _fit_base_fold(
            ["train/aaaaaaaa__horizontal_well.csv"],
            ["train/bbbbbbbb__horizontal_well.csv"],
            {"aaaaaaaa": "group-a", "bbbbbbbb": "group-b"},
            matrix,
        )


def test_outer_prediction_record_uses_sanitized_inference_well_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_ids = ["aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd"]
    test_id = "eeeeeeee"
    train_files = [f"train/{wid}__horizontal_well.csv" for wid in train_ids]
    test_files = [f"train/{test_id}__horizontal_well.csv"]
    matrix = (
        pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array(train_ids),
    )

    class FakeRegressor:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, _features: pd.DataFrame, _target: np.ndarray):
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.full(len(features), 0.25)

    def labeled_well(path: str) -> dict:
        wid = gate.well_id(path)
        return {
            "well": wid,
            "known": np.array([True, False]),
            "tvt_prefix": np.array([10.0, np.nan]),
            "truth": np.array([10.0, 11.0]),
        }

    def inference_well(_path: str) -> dict:
        return {
            "well": test_id,
            "known": np.array([True, False]),
            "tvt_prefix": np.array([10.0, np.nan]),
            "truth": None,
        }

    def point_frame_stub(well: dict, stride: int = 1):
        del stride
        truth = None if well["truth"] is None else np.array([1.0])
        return pd.DataFrame({"feature": [1.0]}), np.array([1]), truth

    monkeypatch.setattr(gate.lgb, "LGBMRegressor", FakeRegressor)
    monkeypatch.setattr(gate, "load_well", labeled_well)
    monkeypatch.setattr(gate, "_load_inference_well", inference_well)
    monkeypatch.setattr(gate, "point_frame", point_frame_stub)
    monkeypatch.setattr(gate, "calibrate_shrink", lambda _oof, _target: 1.0)
    _, test_records, _ = _fit_base_fold(
        train_files,
        test_files,
        {wid: f"group-{index}" for index, wid in enumerate(train_ids + [test_id])},
        matrix,
    )
    assert [record.well for record in test_records] == [test_id]


def test_run_phase_has_no_truth_scoring_and_aggregate_owns_it() -> None:
    run_source = inspect.getsource(_run_one_fold)
    score_source = inspect.getsource(_score_sealed_predictions)
    aggregate_source = inspect.getsource(aggregate_folds)

    assert ".truth" not in run_source
    assert "base_sse" not in run_source and "joint_sse" not in run_source
    assert 'well["truth"]' in score_source
    assert "base_sse" in score_source and "joint_sse" in score_source
    assert aggregate_source.index("validated.append") < aggregate_source.index(
        "fold_artifact_inventory = _build_fold_artifact_inventory"
    )
    assert aggregate_source.index(
        "fold_artifact_inventory = _build_fold_artifact_inventory"
    ) < aggregate_source.index("for repeat, fold, _, shard, arrays in validated")
    assert aggregate_source.index(
        "for repeat, fold, _, shard, arrays in validated"
    ) < aggregate_source.index("_score_sealed_predictions")
    assert aggregate_source.rindex("audit_protocol") < aggregate_source.index(
        "_persist_aggregate_artifacts"
    )
    assert METHOD == "equal_ordered_joint_v2"


def test_logical_prediction_hash_covers_schema_shape_dtype_and_values() -> None:
    arrays = {
        "row_index": np.array([1, 2, 3], dtype=np.int32),
        "joint_prediction": np.array([10.0, 11.0, 12.0], dtype=np.float64),
    }
    reversed_order = dict(reversed(list(arrays.items())))
    assert _logical_array_hash(arrays) == _logical_array_hash(reversed_order)

    changed = {name: value.copy() for name, value in arrays.items()}
    changed["joint_prediction"][1] += 1e-9
    assert _logical_array_hash(arrays) != _logical_array_hash(changed)


def test_fold_artifact_inventory_is_complete_unique_and_hash_bound(
    tmp_path: Path,
) -> None:
    validated = _write_fake_validated_fold_artifacts(tmp_path / "folds")
    inventory = _build_fold_artifact_inventory(validated)

    assert inventory["fold_count"] == OUTER_REPEATS * OUTER_SPLITS
    assert len(inventory["items"]) == OUTER_REPEATS * OUTER_SPLITS
    _validate_fold_artifact_inventory_payload(inventory)
    first = inventory["items"][0]
    assert Path(first["shard_json"]["path"]).name == first["shard_json"]["name"]
    assert (
        first["prediction_npz"]["logical_array_sha256"]
        == validated[0][3]["prediction_logical_sha256"]
    )

    with pytest.raises(ProtocolError, match="incomplete or unexpected"):
        _build_fold_artifact_inventory(validated[:-1])
    with pytest.raises(ProtocolError, match="duplicate identities"):
        _build_fold_artifact_inventory(validated[:-1] + [validated[0]])

    prediction_path = gate._prediction_path(validated[0][2])
    prediction_path.write_bytes(prediction_path.read_bytes() + b"tamper")
    with pytest.raises(ProtocolError, match="byte hash drifted"):
        _build_fold_artifact_inventory(validated)


def test_aggregate_artifacts_are_hash_sealed_and_write_once(tmp_path: Path) -> None:
    inventory = _build_fold_artifact_inventory(
        _write_fake_validated_fold_artifacts(tmp_path / "folds")
    )
    scored = pd.DataFrame(
        {
            "repeat": [0, 1],
            "outer_fold": [2, 3],
            "well": ["w1", "w1"],
            "typewell_profile_hash": ["g1", "g1"],
            "n_rows": [10, 10],
            "base_sse": [100.0, 110.0],
            "typewell_sse": [95.0, 100.0],
            "ordered_sse": [94.0, 99.0],
            "joint_sse": [90.0, 95.0],
        }
    )
    output = tmp_path / "result.json"
    sealed = _persist_aggregate_artifacts(
        output,
        scored,
        {
            "status": "test",
            "protocol_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "fold_artifact_inventory": inventory,
        },
    )
    scored_path = _scored_sse_path(output)

    assert output.is_file() and scored_path.is_file()
    assert gate._protocol_sidecar(output).is_file()
    assert gate._protocol_sidecar(scored_path).is_file()
    assert sealed["scored_sse_artifact"]["sha256"] == sha256_file(scored_path)
    assert (
        sealed["artifact_lineage"]["fold_artifact_inventory_sha256"]
        == inventory["inventory_sha256"]
    )
    assert gate._protocol_sidecar(output).read_text(encoding="ascii").split()[
        0
    ] == sha256_file(output)
    assert gate._protocol_sidecar(scored_path).read_text(encoding="ascii").split()[
        0
    ] == sha256_file(scored_path)
    with pytest.raises(ProtocolError, match="refusing to overwrite"):
        _persist_aggregate_artifacts(
            output,
            scored,
            {
                "status": "second write",
                "protocol_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "fold_artifact_inventory": inventory,
            },
        )


def test_global_sse_summary_and_repeat_coupled_bootstrap_are_order_invariant() -> None:
    one_repeat = pd.DataFrame(
        {
            "well": ["w1", "w2", "w3", "w4", "w5"],
            "typewell_profile_hash": ["a", "a", "b", "c", "c"],
            "n_rows": [10, 30, 20, 15, 25],
            "base_sse": [100.0, 300.0, 500.0, 225.0, 400.0],
            "joint_sse": [90.0, 270.0, 450.0, 180.0, 350.0],
        }
    )
    repeat_zero = one_repeat.assign(repeat=0)
    repeat_one = one_repeat.assign(
        repeat=1,
        base_sse=one_repeat["base_sse"] * 1.2,
        joint_sse=one_repeat["joint_sse"] * 1.1,
    )
    rows = pd.concat([repeat_zero, repeat_one], ignore_index=True)
    summary = _summary(rows)
    assert summary["base_pooled_row_rmse"] == pytest.approx(
        np.sqrt(rows["base_sse"].sum() / rows["n_rows"].sum())
    )
    repeat_gains = _repeat_pooled_gains(rows)

    first = _group_cluster_bootstrap(rows, draws=500, seed=17)
    shuffled = _group_cluster_bootstrap(
        rows.sample(frac=1.0, random_state=9), draws=500, seed=17
    )
    grouped = rows.groupby(["repeat", "typewell_profile_hash"], as_index=False).sum(
        numeric_only=True
    )
    collapsed = _group_cluster_bootstrap(grouped, draws=500, seed=17)
    assert first == shuffled == collapsed
    assert first["n_groups"] == 3
    assert first["observed_mean_repeat_pooled_rmse_gain_ft"] == pytest.approx(
        np.mean(list(repeat_gains.values()))
    )


def test_median_well_gain_is_paired_not_difference_of_medians() -> None:
    rows = pd.DataFrame(
        {
            "well": ["w1", "w2", "w3"],
            "n_rows": [1, 1, 1],
            "base_sse": np.square([1.0, 10.0, 100.0]),
            "joint_sse": np.square([0.0, 90.0, 99.0]),
        }
    )
    summary = _summary(rows)
    assert summary["median_well_rmse_gain_ft"] == pytest.approx(1.0)
    assert summary["median_well_rmse_gain_ft"] != pytest.approx(
        summary["base_median_well_rmse"] - summary["candidate_median_well_rmse"]
    )


def test_joint_best_component_bootstrap_and_top_positive_sse_removal() -> None:
    records = []
    for repeat in range(OUTER_REPEATS):
        for index in range(14):
            base = 200.0 + 10.0 * index + 5.0 * repeat
            # The first eleven wells carry the largest positive joint benefits;
            # the remainder make the post-removal gain slightly positive.
            benefit = 20.0 - index if index < 11 else 1.0
            joint = base - benefit
            records.append(
                {
                    "repeat": repeat,
                    "well": f"w{index:02d}",
                    "typewell_profile_hash": f"g{index // 2:02d}",
                    "n_rows": 10,
                    "base_sse": base,
                    "typewell_sse": joint + 4.0,
                    "ordered_sse": joint + 2.0,
                    "joint_sse": joint,
                }
            )
    rows = pd.DataFrame(records)
    comparison = _joint_vs_best_component_bootstrap(rows, draws=500, seed=31)
    assert comparison["observed_joint_vs_best_component_gain_ft"] > 0.0
    assert comparison["ci95_low_ft"] > 0.0

    removal = _top_positive_sse_removal(rows)
    assert removal["removed_count"] == 10
    assert removal["removed_wells"] == [f"w{index:02d}" for index in range(10)]
    assert removal["remaining_mean_repeat_pooled_rmse_gain_ft"] > 0.0
    assert removal["passed"] is True


def test_coefficient_stability_detects_sign_flips_and_repeated_bound_hits() -> None:
    learned = []
    for repeat in range(OUTER_REPEATS):
        for fold in range(OUTER_SPLITS):
            index = repeat * OUTER_SPLITS + fold
            learned.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "typewell_shrink": 0.5 + 0.01 * index,
                    "ordered_shrink": 0.7 + 0.01 * index,
                    "joint_coefficients": [0.4 + 0.01 * index, 0.8 - 0.01 * index],
                }
            )
    stable = _coefficient_stability(learned)
    assert stable["passed"] is True

    sign_flip = [dict(row) for row in learned]
    sign_flip[0] = {**sign_flip[0], "joint_coefficients": [-0.2, 0.8]}
    flipped = _coefficient_stability(sign_flip)
    assert flipped["coefficients"]["joint_typewell_coefficient"]["sign_flip"]
    assert flipped["passed"] is False

    repeated_bound = [dict(row) for row in learned]
    repeated_bound[0] = {**repeated_bound[0], "typewell_shrink": 1.5}
    repeated_bound[1] = {**repeated_bound[1], "typewell_shrink": 1.5}
    bounded = _coefficient_stability(repeated_bound)
    assert bounded["coefficients"]["typewell_shrink"]["upper_bound_hits"] == 2
    assert bounded["passed"] is False


def test_runtime_mutation_guards_frozen_parameter_maps() -> None:
    original_params = dict(gate.PARAMS)
    original_settings = dict(gate.FROZEN_SETTINGS)
    try:
        gate.PARAMS["n_estimators"] = 901
        with pytest.raises(ProtocolError, match="production model parameters drifted"):
            frozen_research_params()
        gate.PARAMS.clear()
        gate.PARAMS.update(original_params)

        gate.FROZEN_SETTINGS["window_samples"] = 99
        with pytest.raises(ProtocolError, match="ordered-transport settings drifted"):
            _frozen_ordered_settings()
    finally:
        gate.PARAMS.clear()
        gate.PARAMS.update(original_params)
        gate.FROZEN_SETTINGS.clear()
        gate.FROZEN_SETTINGS.update(original_settings)


def test_frozen_lightgbm_and_cli_contract() -> None:
    params = frozen_research_params()
    assert params["random_state"] == 20260810
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert params["n_jobs"] == 4
    evaluation = gate._frozen_evaluation_protocol()
    assert evaluation["support_criteria"][
        "primary_mean_repeat_pooled_gain_at_least_ft"
    ] == pytest.approx(0.2)
    assert evaluation["aggregate_artifacts"]["overwrite"] == "forbidden"

    args = parse_args(["run", "--protocol", "p.json", "--all-folds", "--resume"])
    assert args.command == "run" and args.all_folds and args.resume
    args = parse_args(["aggregate", "--protocol", "p.json", "--output", "result.json"])
    assert args.command == "aggregate"


def test_all_folds_rejects_fold_selector(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_args(
            [
                "run",
                "--protocol",
                "p.json",
                "--all-folds",
                "--fold",
                "2",
            ]
        )
    assert "--fold cannot be used with --all-folds" in capsys.readouterr().err
