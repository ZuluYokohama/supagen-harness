from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest

from research.structural_field import (
    FieldConfig,
    InferenceSafetyError,
    StructuralFieldError,
    TrainingWell,
    _segments_intersect,
    build_edge_one_form_operator,
    build_face_edge_incidence,
    build_face_circulation_operator,
    build_graph_incidence,
    build_graph_laplacian,
    fit_structural_field,
    predict_structural_field,
)


AFFINE_GRADIENT = np.array([0.018, -0.011])
AFFINE_INTERCEPT = 10_000.0


def _line_well(
    well_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    gradient: np.ndarray = AFFINE_GRADIENT,
    intercept: float = AFFINE_INTERCEPT,
    rows: int = 161,
) -> TrainingWell:
    md = np.linspace(0.0, 16_000.0, rows)
    fraction = md / md[-1]
    x = start[0] + fraction * (end[0] - start[0])
    y = start[1] + fraction * (end[1] - start[1])
    z = -9_000.0 + 0.025 * md
    structural = intercept + gradient[0] * x + gradient[1] * y
    tvt = structural - z
    return TrainingWell(well_id, md, x, y, z, tvt)


def _affine_wells() -> list[TrainingWell]:
    wells = []
    for index, coordinate in enumerate((-7_500.0, -3_750.0, 0.0, 3_750.0, 7_500.0)):
        wells.append(
            _line_well(
                f"east_{index}",
                (-10_000.0, coordinate),
                (10_000.0, coordinate),
            )
        )
        wells.append(
            _line_well(
                f"north_{index}",
                (coordinate, -10_000.0),
                (coordinate, 10_000.0),
            )
        )
    return wells


def _config(**changes: float | int) -> FieldConfig:
    values: dict[str, float | int] = {
        "inducing_cell_ft": 2_500.0,
        "support_length_ft": 7_500.0,
        "graph_max_edge_ft": 10_000.0,
        "cut_fallback_radius_ft": 400.0,
        "laplacian_strength": 0.3,
        "circulation_strength": 0.1,
    }
    values.update(changes)
    return FieldConfig(**values)


def _target(
    *,
    curved: bool = False,
    far: bool = False,
) -> tuple[np.ndarray, ...]:
    md = np.arange(0.0, 12_000.0 + 100.0, 100.0)
    if curved:
        theta = np.linspace(-0.8 * np.pi, 0.8 * np.pi, len(md))
        x = 6_000.0 * np.cos(theta)
        y = 6_000.0 * np.sin(theta)
    else:
        x = np.linspace(-8_000.0, 8_000.0, len(md))
        y = np.linspace(-2_500.0, 2_500.0, len(md))
    if far:
        x += 200_000.0
        y += 200_000.0
    z = -9_100.0 + 0.02 * md
    structural = AFFINE_INTERCEPT + AFFINE_GRADIENT[0] * x + AFFINE_GRADIENT[1] * y
    truth = structural - z
    prefix_rows = 31
    tvt_input = np.full(len(md), np.nan)
    tvt_input[:prefix_rows] = truth[:prefix_rows]
    policy = truth.copy()
    policy[prefix_rows:] += 18.0
    return md, x, y, z, truth, tvt_input, policy


@pytest.fixture(scope="module")
def affine_model():
    return fit_structural_field(_affine_wells(), _config())


def test_defined_graph_operators_annihilate_constant_gradient() -> None:
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    faces = np.array([[0, 1, 2]])
    conductance = np.array([1.0, 2.0, 3.0])
    incidence = build_graph_incidence(3, edges)
    laplacian = build_graph_laplacian(3, edges, conductance)
    edge_form = build_edge_one_form_operator(nodes, edges)
    face_edge = build_face_edge_incidence(nodes, edges, faces)
    circulation = build_face_circulation_operator(nodes, edges, faces)
    constant_scalar = np.ones(3)
    constant_gradient = np.tile(np.array([0.3, -0.2]), 3)
    np.testing.assert_allclose(incidence @ constant_scalar, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(laplacian @ constant_scalar, 0.0, atol=1.0e-14)
    np.testing.assert_allclose((face_edge @ incidence).toarray(), 0.0, atol=1.0e-14)
    assert edge_form.shape == (3, 6)
    np.testing.assert_allclose(circulation @ constant_gradient, 0.0, atol=1.0e-14)
    rotational_gradient = np.array([0.0, 0.0, 0.0, 2.0, -2.0, 0.0])
    expected = 4.0 / (4.0 + np.sqrt(8.0))
    assert float((circulation @ rotational_gradient)[0]) == pytest.approx(expected)


def test_segment_intersection_rejects_collinear_disjoint_segments() -> None:
    assert not _segments_intersect(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([3.0, 0.0]),
    )
    assert _segments_intersect(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([3.0, 0.0]),
    )
    offset = np.array([2_000_000.0, 4_000_000.0])
    assert not _segments_intersect(
        offset + np.array([0.0, 0.0]),
        offset + np.array([1.0, 0.0]),
        offset + np.array([2.0, 0.0]),
        offset + np.array([3.0, 0.0]),
    )


def test_affine_dip_recovery_and_exact_anchor(affine_model) -> None:
    assert affine_model.diagnostics.discontinuity_candidates == 0
    np.testing.assert_allclose(
        np.median(affine_model.gradients_xy, axis=0),
        AFFINE_GRADIENT,
        atol=2.0e-4,
    )
    md, x, y, z, truth, tvt_input, policy = _target()
    result = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    prefix_rows = int(np.flatnonzero(~np.isfinite(tvt_input))[0])
    np.testing.assert_allclose(
        result.field_tvt[prefix_rows:], truth[prefix_rows:], atol=0.25
    )
    assert result.field_tvt[prefix_rows - 1] + z[prefix_rows - 1] == pytest.approx(
        tvt_input[prefix_rows - 1] + z[prefix_rows - 1], abs=1.0e-12
    )
    field_error = np.sqrt(
        np.mean((result.field_tvt[prefix_rows:] - truth[prefix_rows:]) ** 2)
    )
    policy_error = np.sqrt(np.mean((policy[prefix_rows:] - truth[prefix_rows:]) ** 2))
    assert field_error < 0.02 * policy_error
    assert result.diagnostics.mean_confidence > 0.0


def test_anchor_is_invariant_to_vertical_coordinate_shift(affine_model) -> None:
    md, x, y, z, _, tvt_input, policy = _target()
    base = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    shift = 725.0
    shifted_input = tvt_input.copy()
    shifted_input[np.isfinite(shifted_input)] -= shift
    shifted = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z + shift,
        tvt_input=shifted_input,
        policy_tvt=policy - shift,
    )
    np.testing.assert_allclose(shifted.field_tvt, base.field_tvt - shift, atol=1.0e-10)
    np.testing.assert_allclose(
        shifted.predicted_tvt, base.predicted_tvt - shift, atol=1.0e-10
    )
    np.testing.assert_array_equal(shifted.confidence, base.confidence)
    assert shifted.diagnostics.anchor_s == pytest.approx(base.diagnostics.anchor_s)


def test_prefix_bias_component_is_explicit_and_reconstructs_field(affine_model) -> None:
    md, x, y, z, truth, _, policy = _target()
    biased_truth = truth + 0.004 * md
    tvt_input = np.full(len(md), np.nan)
    tvt_input[:31] = biased_truth[:31]
    result = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    assert result.diagnostics.prefix_bias == pytest.approx(0.004, abs=2.0e-4)
    assert np.max(np.abs(result.prefix_bias_delta_tvt)) > 1.0
    supported = result.support_mask
    np.testing.assert_allclose(
        result.field_tvt[supported],
        (
            policy
            + result.field_delta_without_prefix_bias_tvt
            + result.prefix_bias_delta_tvt
        )[supported],
        atol=1.0e-10,
    )


def test_curved_path_integrates_differential_field(affine_model) -> None:
    md, x, y, z, truth, tvt_input, policy = _target(curved=True)
    result = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    prefix_rows = int(np.flatnonzero(~np.isfinite(tvt_input))[0])
    np.testing.assert_allclose(
        result.field_tvt[prefix_rows:], truth[prefix_rows:], atol=0.35
    )
    assert result.diagnostics.prefix_bias_intervals > 0


def test_sparse_support_falls_back_exactly_to_policy(affine_model) -> None:
    md, x, y, z, _, tvt_input, policy = _target(far=True)
    result = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    prefix_rows = int(np.flatnonzero(~np.isfinite(tvt_input))[0])
    np.testing.assert_array_equal(
        result.predicted_tvt[prefix_rows:], policy[prefix_rows:]
    )
    np.testing.assert_array_equal(result.confidence[prefix_rows:], 0.0)
    assert result.diagnostics.fallback_fraction == 1.0


def test_dense_raw_rows_use_bounded_knots_and_returned_blend_identity(
    affine_model,
) -> None:
    md = np.arange(0.0, 12_000.0 + 5.0, 5.0)
    x = np.linspace(-8_000.0, 8_000.0, len(md))
    y = np.linspace(-2_500.0, 2_500.0, len(md))
    z = -9_100.0 + 0.02 * md
    structural = AFFINE_INTERCEPT + AFFINE_GRADIENT[0] * x + AFFINE_GRADIENT[1] * y
    target_tvt = structural - z
    prefix_rows = 401
    tvt_input = np.full(len(md), np.nan)
    tvt_input[:prefix_rows] = target_tvt[:prefix_rows]
    policy = target_tvt.copy()
    policy[prefix_rows:] += 10.0
    started = perf_counter()
    result = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    elapsed = perf_counter() - started
    assert elapsed < 5.0
    assert result.diagnostics.rows == len(md)
    assert result.diagnostics.evaluation_rows < 130
    supported = result.support_mask
    np.testing.assert_allclose(
        result.field_tvt[supported],
        (
            policy
            + result.field_delta_without_prefix_bias_tvt
            + result.prefix_bias_delta_tvt
        )[supported],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.predicted_tvt,
        policy
        + affine_model.config.blend_alpha
        * result.confidence
        * (result.field_tvt - policy),
        atol=1.0e-12,
    )


def test_support_gap_forces_permanent_downstream_fallback(affine_model) -> None:
    md = np.arange(0.0, 12_000.0 + 100.0, 100.0)
    x = np.concatenate(
        (
            np.linspace(-5_000.0, 4_000.0, 55),
            np.full(12, 150_000.0),
            np.linspace(4_000.0, 0.0, len(md) - 67),
        )
    )
    y = np.zeros(len(md))
    z = np.full(len(md), -9_000.0)
    structural = AFFINE_INTERCEPT + AFFINE_GRADIENT[0] * x
    truth = structural - z
    tvt_input = np.full(len(md), np.nan)
    tvt_input[:31] = truth[:31]
    policy = truth + 9.0
    result = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    suffix_confidence = result.confidence[31:]
    positive = np.flatnonzero(suffix_confidence > 0.0)
    assert len(positive) > 0
    first_gap = int(positive[-1] + 1)
    assert first_gap < len(suffix_confidence)
    np.testing.assert_array_equal(suffix_confidence[first_gap:], 0.0)
    np.testing.assert_array_equal(
        result.predicted_tvt[31 + first_gap :], policy[31 + first_gap :]
    )


def test_parallel_support_is_query_direction_specific() -> None:
    gradient = np.array([0.020, 0.030])
    wells = [
        _line_well(
            f"parallel_{index}",
            (-8_000.0, y_value),
            (8_000.0, y_value),
            gradient=gradient,
        )
        for index, y_value in enumerate((-4_000.0, -2_000.0, 0.0, 2_000.0, 4_000.0))
    ]
    model = fit_structural_field(
        wells,
        _config(
            inducing_cell_ft=2_000.0,
            support_length_ft=7_000.0,
            graph_max_edge_ft=9_000.0,
        ),
    )
    md = np.arange(121, dtype=float) * 100.0
    z = np.full(len(md), -9_000.0)

    east_x = np.linspace(-7_000.0, 7_000.0, len(md))
    east_y = np.zeros(len(md))
    east_truth = AFFINE_INTERCEPT + gradient[0] * east_x - z
    east_input = np.full(len(md), np.nan)
    east_input[:31] = east_truth[:31]
    east_policy = east_truth + 10.0
    east = predict_structural_field(
        model,
        md=md,
        x=east_x,
        y=east_y,
        z=z,
        tvt_input=east_input,
        policy_tvt=east_policy,
    )
    assert east.diagnostics.azimuth_condition_median == float("inf")
    assert east.diagnostics.query_direction_observability_mean > 0.99
    assert east.diagnostics.mean_confidence > 0.9

    north_x = np.zeros(len(md))
    north_y = np.linspace(-5_000.0, 5_000.0, len(md))
    north_truth = AFFINE_INTERCEPT + gradient[1] * north_y - z
    north_input = np.full(len(md), np.nan)
    north_input[:31] = north_truth[:31]
    north_policy = north_truth + 10.0
    north = predict_structural_field(
        model,
        md=md,
        x=north_x,
        y=north_y,
        z=z,
        tvt_input=north_input,
        policy_tvt=north_policy,
    )
    np.testing.assert_array_equal(north.confidence[31:], 0.0)
    np.testing.assert_array_equal(north.predicted_tvt[31:], north_policy[31:])


def test_smooth_curved_field_does_not_create_cut_candidates() -> None:
    def smooth_well(
        well_id: str, start: tuple[float, float], end: tuple[float, float]
    ) -> TrainingWell:
        md = np.linspace(0.0, 16_000.0, 161)
        fraction = md / md[-1]
        x = start[0] + fraction * (end[0] - start[0])
        y = start[1] + fraction * (end[1] - start[1])
        z = -9_000.0 + 0.025 * md
        structural = 10_000.0 + 0.010 * x - 0.005 * y + 1.0e-7 * x * x
        return TrainingWell(well_id, md, x, y, z, structural - z)

    wells: list[TrainingWell] = []
    for index, coordinate in enumerate((-7_500.0, -3_750.0, 0.0, 3_750.0, 7_500.0)):
        wells.append(
            smooth_well(
                f"smooth_east_{index}",
                (-10_000.0, coordinate),
                (10_000.0, coordinate),
            )
        )
        wells.append(
            smooth_well(
                f"smooth_north_{index}",
                (coordinate, -10_000.0),
                (coordinate, 10_000.0),
            )
        )
    model = fit_structural_field(wells, _config())
    assert model.diagnostics.discontinuity_candidates == 0


def _discontinuous_wells() -> list[TrainingWell]:
    wells: list[TrainingWell] = []
    left_gradient = np.array([0.030, -0.004])
    right_gradient = np.array([-0.030, -0.004])
    for side, x_range, gradient in (
        ("left", (-8_000.0, -750.0), left_gradient),
        ("right", (750.0, 8_000.0), right_gradient),
    ):
        for index, y_value in enumerate((-4_000.0, 0.0, 4_000.0)):
            wells.append(
                _line_well(
                    f"{side}_east_{index}",
                    (x_range[0], y_value),
                    (x_range[1], y_value),
                    gradient=gradient,
                )
            )
        x_value = -3_000.0 if side == "left" else 3_000.0
        for index in range(2):
            wells.append(
                _line_well(
                    f"{side}_north_{index}",
                    (x_value + 300.0 * index, -6_000.0),
                    (x_value + 300.0 * index, 6_000.0),
                    gradient=gradient,
                )
            )
    return wells


def test_discontinuity_candidate_suppresses_field_correction() -> None:
    model = fit_structural_field(
        _discontinuous_wells(),
        _config(
            inducing_cell_ft=1_500.0,
            support_length_ft=2_800.0,
            graph_max_edge_ft=3_500.0,
            discontinuity_mad_threshold=3.0,
            discontinuity_absolute_floor=1.0e-3,
            cut_fallback_radius_ft=500.0,
        ),
    )
    assert model.diagnostics.discontinuity_candidates > 0
    cut_midpoint = model.nodes_xy[model.edges[model.cut_edge_mask]].mean(axis=1)
    assert np.median(np.abs(cut_midpoint[:, 0])) < 500.0
    np.testing.assert_array_equal(model.edge_conductance[model.cut_edge_mask], 0.0)
    cut_keys = {tuple(edge) for edge in model.edges[model.cut_edge_mask]}
    for face in model.faces:
        sides = {
            tuple(sorted((int(face[0]), int(face[1])))),
            tuple(sorted((int(face[1]), int(face[2])))),
            tuple(sorted((int(face[2]), int(face[0])))),
        }
        assert not sides & cut_keys
    md = np.arange(0.0, 14_000.0 + 100.0, 100.0)
    x = np.linspace(-7_000.0, 7_000.0, len(md))
    y = np.zeros(len(md))
    z = np.full(len(md), -9_000.0)
    structural = AFFINE_INTERCEPT + 0.03 * x
    truth = structural - z
    tvt_input = np.full(len(md), np.nan)
    tvt_input[:41] = truth[:41]
    policy = truth + 12.0
    result = predict_structural_field(
        model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    prefix_rows = 41
    fallback = result.confidence[prefix_rows:] == 0.0
    assert fallback.any()
    np.testing.assert_array_equal(
        result.predicted_tvt[prefix_rows:][fallback], policy[prefix_rows:][fallback]
    )
    assert result.diagnostics.cut_edge_crossings > 0


def test_suffix_truth_mutation_is_invisible_and_rejected_as_input(affine_model) -> None:
    md, x, y, z, truth, tvt_input, policy = _target()
    mutated_truth = truth.copy()
    mutated_truth[~np.isfinite(tvt_input)] += np.linspace(
        -1_000.0, 1_000.0, np.sum(~np.isfinite(tvt_input))
    )
    first_input = np.where(np.isfinite(tvt_input), truth, np.nan)
    second_input = np.where(np.isfinite(tvt_input), mutated_truth, np.nan)
    first = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=first_input,
        policy_tvt=policy,
    )
    second = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=second_input,
        policy_tvt=policy,
    )
    np.testing.assert_array_equal(first.predicted_tvt, second.predicted_tvt)
    changed_prefix = first_input.copy()
    changed_prefix[np.isfinite(changed_prefix)] += 5.0
    changed = predict_structural_field(
        affine_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=changed_prefix,
        policy_tvt=policy,
    )
    prefix_rows = int(np.flatnonzero(~np.isfinite(first_input))[0])
    assert not np.array_equal(
        first.predicted_tvt[prefix_rows:], changed.predicted_tvt[prefix_rows:]
    )
    with pytest.raises(InferenceSafetyError, match="suffix_truth"):
        predict_structural_field(
            affine_model,
            md=md,
            x=x,
            y=y,
            z=z,
            tvt_input=tvt_input,
            policy_tvt=policy,
            suffix_truth=truth,
        )


def test_fit_and_prediction_are_deterministic() -> None:
    wells = _affine_wells()
    first_model = fit_structural_field(wells, _config())
    second_model = fit_structural_field(reversed(wells), _config())
    for first, second in (
        (first_model.nodes_xy, second_model.nodes_xy),
        (first_model.gradients_xy, second_model.gradients_xy),
        (first_model.edges, second_model.edges),
        (first_model.faces, second_model.faces),
        (first_model.cut_edge_mask, second_model.cut_edge_mask),
    ):
        np.testing.assert_array_equal(first, second)
    md, x, y, z, _, tvt_input, policy = _target(curved=True)
    first = predict_structural_field(
        first_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    second = predict_structural_field(
        second_model,
        md=md,
        x=x,
        y=y,
        z=z,
        tvt_input=tvt_input,
        policy_tvt=policy,
    )
    np.testing.assert_array_equal(first.predicted_tvt, second.predicted_tvt)


def test_md_subdivision_does_not_change_resampled_equal_well_fit() -> None:
    coarse: list[TrainingWell] = []
    dense: list[TrainingWell] = []
    for index, coordinate in enumerate((-7_500.0, -3_750.0, 0.0, 3_750.0, 7_500.0)):
        for label, start, end in (
            ("east", (-10_000.0, coordinate), (10_000.0, coordinate)),
            ("north", (coordinate, -10_000.0), (coordinate, 10_000.0)),
        ):
            well_id = f"{label}_{index}"
            coarse.append(_line_well(well_id, start, end, rows=81))
            dense.append(_line_well(well_id, start, end, rows=321))
    coarse_model = fit_structural_field(coarse, _config())
    dense_model = fit_structural_field(dense, _config())
    np.testing.assert_array_equal(coarse_model.nodes_xy, dense_model.nodes_xy)
    np.testing.assert_array_equal(coarse_model.gradients_xy, dense_model.gradients_xy)


def test_lsmr_nonconvergence_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_lsmr(matrix, _rhs, **_kwargs):
        return (np.zeros(matrix.shape[1]), 7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    monkeypatch.setattr("research.structural_field.lsmr", failed_lsmr)
    with pytest.raises(StructuralFieldError, match="stop code 7"):
        fit_structural_field(_affine_wells()[:2], _config())


def test_invalid_inputs_fail_closed(affine_model) -> None:
    with pytest.raises(StructuralFieldError, match="support_length_ft"):
        replace(_config(), support_length_ft=0.0)
    duplicate = _affine_wells()[:2]
    duplicate[1] = replace(duplicate[1], well_id=duplicate[0].well_id)
    with pytest.raises(StructuralFieldError, match="unique"):
        fit_structural_field(duplicate, _config())
    with pytest.raises(StructuralFieldError, match="max_resampled_intervals"):
        fit_structural_field(
            _affine_wells()[:2],
            replace(_config(), max_resampled_intervals_per_well=10),
        )
    md, x, y, z, _, tvt_input, policy = _target()
    bounded_model = replace(
        affine_model,
        config=replace(affine_model.config, max_prediction_rows=100),
    )
    with pytest.raises(StructuralFieldError, match="max_prediction_rows"):
        predict_structural_field(
            bounded_model,
            md=md,
            x=x,
            y=y,
            z=z,
            tvt_input=tvt_input,
            policy_tvt=policy,
        )
    with pytest.raises(InferenceSafetyError, match="fully finite"):
        predict_structural_field(
            affine_model,
            md=md,
            x=x,
            y=y,
            z=z,
            tvt_input=np.nan_to_num(tvt_input, nan=0.0),
            policy_tvt=policy,
        )
    noncontiguous = tvt_input.copy()
    noncontiguous[-1] = noncontiguous[0]
    with pytest.raises(InferenceSafetyError, match="contiguous"):
        predict_structural_field(
            affine_model,
            md=md,
            x=x,
            y=y,
            z=z,
            tvt_input=noncontiguous,
            policy_tvt=policy,
        )
    with pytest.raises(StructuralFieldError, match="strictly increasing"):
        predict_structural_field(
            affine_model,
            md=md[::-1],
            x=x,
            y=y,
            z=z,
            tvt_input=tvt_input,
            policy_tvt=policy,
        )
