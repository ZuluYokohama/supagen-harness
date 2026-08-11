"""Anchored differential structural-field research primitive.

The model fits a regularized horizontal latent for ``S = TVT + Z`` from
labeled *training* trajectories.  Where azimuth coverage is rank deficient,
only its observable projection along the target direction is an estimand; the
full two-component vector is not claimed as identified.  A target prediction
integrates that projection from the exact last-known prefix anchor.  Absolute
training-well structural datums are never transported to a target.

The inference boundary is deliberately narrow: ``MD, X, Y, Z``, a contiguous
``TVT_input`` prefix, and a separately-produced policy TVT path.  Formation
labels, interpreted geology, images, and target suffix truth are not inputs.

This is a phase-one research implementation, not a frozen competition gate.
Its topological quantities are explicitly constructed:

* ``B`` is oriented graph incidence and ``L = B.T @ W @ B`` is its Laplacian;
* ``P`` maps nodal gradients to trapezoidal edge one-forms; and
* ``C @ P`` is normalized face circulation on Delaunay triangles.

No spectral-gap or holonomy interpretation is made.  Graph edges with large
training-only derivative incompatibility are called discontinuity candidates,
not faults.  Predictions near or across those cut edges fall back to policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr
from scipy.spatial import Delaunay, QhullError, cKDTree


_FORBIDDEN_PREDICTION_INPUTS = frozenset(
    {
        "suffix_truth",
        "suffix_tvt",
        "true_tvt",
        "truth_tvt",
        "formation",
        "formations",
        "formation_surfaces",
        "geology",
        "geology_labels",
        "train_image",
        "png",
    }
)


class StructuralFieldError(ValueError):
    """Raised when the structural-field construction is not well defined."""


class InferenceSafetyError(StructuralFieldError):
    """Raised when prediction input could expose unavailable target labels."""


@dataclass(frozen=True)
class FieldConfig:
    """Deterministic, CPU-bounded phase-one settings.

    The regularization strengths are trace normalized against the data design,
    so their numeric values are dimensionless ratios rather than raw matrix
    coefficients.  These defaults are hypotheses; a later gate must freeze or
    select them using training-only inner folds.
    """

    resample_step_md: float = 100.0
    lateral_max_abs_dz_dmd: float = 0.15
    min_horizontal_speed: float = 1.0e-4
    inducing_cell_ft: float = 5_000.0
    max_nodes: int = 2_000
    max_resampled_intervals_per_well: int = 20_000
    max_training_observations: int = 250_000
    max_prediction_rows: int = 100_000
    max_support_neighbors: int = 4_096
    max_distinct_support_wells: int = 16
    interpolation_neighbors: int = 6
    support_length_ft: float = 15_000.0
    graph_neighbors: int = 6
    graph_max_edge_ft: float = 22_500.0
    laplacian_strength: float = 0.3
    circulation_strength: float = 0.1
    ridge_strength: float = 1.0e-6
    huber_delta: float = 1.5
    irls_iterations: int = 4
    solver_max_iterations: int = 2_000
    discontinuity_mad_threshold: float = 4.0
    discontinuity_absolute_floor: float = 1.0e-4
    cut_fallback_radius_ft: float = 500.0
    prefix_bias_window_md: float = 1_000.0
    prefix_bias_shrink: float = 1.0
    max_abs_prefix_bias: float = 0.10
    blend_alpha: float = 1.0
    min_effective_wells: float = 1.5
    min_directional_observability: float = 0.05

    def __post_init__(self) -> None:
        positive_float_names = (
            "resample_step_md",
            "lateral_max_abs_dz_dmd",
            "min_horizontal_speed",
            "inducing_cell_ft",
            "support_length_ft",
            "graph_max_edge_ft",
            "huber_delta",
            "discontinuity_mad_threshold",
            "discontinuity_absolute_floor",
            "cut_fallback_radius_ft",
            "prefix_bias_window_md",
            "max_abs_prefix_bias",
            "min_effective_wells",
            "min_directional_observability",
        )
        for name in positive_float_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise StructuralFieldError(f"{name} must be finite and positive")
        nonnegative_names = (
            "laplacian_strength",
            "circulation_strength",
            "ridge_strength",
            "prefix_bias_shrink",
            "blend_alpha",
        )
        for name in nonnegative_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise StructuralFieldError(f"{name} must be finite and nonnegative")
        if self.blend_alpha > 1.0:
            raise StructuralFieldError("blend_alpha must not exceed one")
        if self.min_directional_observability > 1.0:
            raise StructuralFieldError(
                "min_directional_observability must not exceed one"
            )
        for name in (
            "max_nodes",
            "max_resampled_intervals_per_well",
            "max_training_observations",
            "max_prediction_rows",
            "max_support_neighbors",
            "max_distinct_support_wells",
            "interpolation_neighbors",
            "graph_neighbors",
            "irls_iterations",
            "solver_max_iterations",
        ):
            if int(getattr(self, name)) < 1:
                raise StructuralFieldError(f"{name} must be at least one")
        if self.max_support_neighbors < self.max_distinct_support_wells:
            raise StructuralFieldError(
                "max_support_neighbors must cover max_distinct_support_wells"
            )


@dataclass(frozen=True)
class TrainingWell:
    """One labeled outer-training trajectory.

    ``tvt`` may be fully labeled here because this object is accepted only by
    :func:`fit_structural_field`, never by the target prediction function.
    """

    well_id: str
    md: ArrayLike
    x: ArrayLike
    y: ArrayLike
    z: ArrayLike
    tvt: ArrayLike


@dataclass(frozen=True)
class FitDiagnostics:
    """Training-only diagnostics for defined numerical quantities."""

    wells: int
    resampled_intervals: int
    inducing_nodes: int
    graph_edges: int
    graph_faces: int
    discontinuity_candidates: int
    graph_components_after_cuts: int
    actual_inducing_cell_ft: float
    robust_passes: int
    weighted_derivative_rmse: float
    derivative_residual_scale: float
    graph_smoothness_energy: float
    normalized_circulation_rms: float
    solver_stop_codes: tuple[int, ...]


@dataclass(frozen=True)
class StructuralFieldModel:
    """Compact inducing representation of the regularized differential latent."""

    config: FieldConfig
    nodes_xy: NDArray[np.float64]
    gradients_xy: NDArray[np.float64]
    edges: NDArray[np.int64]
    edge_conductance: NDArray[np.float64]
    faces: NDArray[np.int64]
    cut_edge_mask: NDArray[np.bool_]
    node_components: NDArray[np.int64]
    node_effective_wells: NDArray[np.float64]
    node_azimuth_gram: NDArray[np.float64]
    support_xy: NDArray[np.float64]
    support_unit_u: NDArray[np.float64]
    support_well_index: NDArray[np.int64]
    support_tree: cKDTree
    diagnostics: FitDiagnostics


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Inference-visible diagnostics for a target prediction."""

    status: str
    rows: int
    evaluation_rows: int
    prefix_rows: int
    suffix_rows: int
    anchor_s: float
    prefix_bias: float
    prefix_bias_intervals: int
    nearest_resampled_training_midpoint_distance_mean_ft: float
    nearest_resampled_training_midpoint_distance_max_ft: float
    effective_well_support_mean: float
    azimuth_condition_median: float
    query_direction_observability_mean: float
    cut_edge_crossings: int
    fallback_fraction: float
    mean_confidence: float
    max_abs_field_policy_difference_tvt: float


@dataclass(frozen=True)
class StructuralPrediction:
    """Anchored field proposal, policy blend, confidence, and diagnostics."""

    predicted_tvt: NDArray[np.float64]
    field_tvt: NDArray[np.float64]
    field_delta_without_prefix_bias_tvt: NDArray[np.float64]
    prefix_bias_delta_tvt: NDArray[np.float64]
    confidence: NDArray[np.float64]
    support_mask: NDArray[np.bool_]
    diagnostics: PredictionDiagnostics


@dataclass(frozen=True)
class _Observations:
    xy: NDArray[np.float64]
    u: NDArray[np.float64]
    q: NDArray[np.float64]
    well_index: NDArray[np.int64]
    well_ids: tuple[str, ...]
    base_weight: NDArray[np.float64]


def _as_vector(
    name: str,
    values: ArrayLike,
    *,
    expected_length: int | None = None,
    finite: bool = True,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise StructuralFieldError(f"{name} must be one-dimensional")
    if expected_length is not None and len(result) != expected_length:
        raise StructuralFieldError(
            f"{name} must have length {expected_length}, got {len(result)}"
        )
    if finite and not np.isfinite(result).all():
        raise StructuralFieldError(f"{name} must contain only finite values")
    return result.copy()


def _validate_md(md: NDArray[np.float64]) -> None:
    if len(md) < 2:
        raise StructuralFieldError("trajectory needs at least two rows")
    if not np.all(np.diff(md) > 0.0):
        raise StructuralFieldError("MD must be finite and strictly increasing")


def _resampling_knots(
    md: NDArray[np.float64], step: float, max_intervals: int
) -> NDArray[np.float64]:
    """Return endpoint-preserving knots on a global deterministic MD lattice."""

    estimated_intervals = int(np.ceil((md[-1] - md[0]) / step)) + 1
    if estimated_intervals > max_intervals:
        raise StructuralFieldError(
            "resampled intervals exceed max_resampled_intervals_per_well"
        )
    first_multiple = np.ceil(md[0] / step) * step
    interior = np.arange(first_multiple, md[-1] + 0.5 * step, step, dtype=float)
    interior = interior[(interior > md[0]) & (interior < md[-1])]
    return np.unique(np.concatenate(([md[0]], interior, [md[-1]])))


def _well_observations(
    well: TrainingWell,
    config: FieldConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    md = _as_vector(f"{well.well_id}.MD", well.md)
    _validate_md(md)
    n_rows = len(md)
    x = _as_vector(f"{well.well_id}.X", well.x, expected_length=n_rows)
    y = _as_vector(f"{well.well_id}.Y", well.y, expected_length=n_rows)
    z = _as_vector(f"{well.well_id}.Z", well.z, expected_length=n_rows)
    tvt = _as_vector(f"{well.well_id}.TVT", well.tvt, expected_length=n_rows)
    knots = _resampling_knots(
        md, config.resample_step_md, config.max_resampled_intervals_per_well
    )
    xk = np.interp(knots, md, x)
    yk = np.interp(knots, md, y)
    zk = np.interp(knots, md, z)
    sk = np.interp(knots, md, tvt + z)
    delta_md = np.diff(knots)
    delta_x = np.diff(xk)
    delta_y = np.diff(yk)
    dz_dmd = np.diff(zk) / delta_md
    u = np.column_stack((delta_x / delta_md, delta_y / delta_md))
    horizontal_speed = np.linalg.norm(u, axis=1)
    lateral = (np.abs(dz_dmd) <= config.lateral_max_abs_dz_dmd) & (
        horizontal_speed >= config.min_horizontal_speed
    )
    xy = np.column_stack(((xk[:-1] + xk[1:]) / 2.0, (yk[:-1] + yk[1:]) / 2.0))
    q = np.diff(sk) / delta_md
    finite = np.isfinite(q) & np.isfinite(u).all(axis=1) & np.isfinite(xy).all(axis=1)
    keep = lateral & finite
    return xy[keep], u[keep], q[keep], delta_md[keep]


def _collect_observations(
    wells: Iterable[TrainingWell],
    config: FieldConfig,
) -> _Observations:
    ordered = sorted(wells, key=lambda item: item.well_id)
    if len(ordered) < 2:
        raise StructuralFieldError("at least two training wells are required")
    ids = [well.well_id for well in ordered]
    if any(not well_id for well_id in ids):
        raise StructuralFieldError("well_id must be nonempty")
    if len(set(ids)) != len(ids):
        raise StructuralFieldError("training well_id values must be unique")

    xy_parts: list[NDArray[np.float64]] = []
    u_parts: list[NDArray[np.float64]] = []
    q_parts: list[NDArray[np.float64]] = []
    index_parts: list[NDArray[np.int64]] = []
    interval_parts: list[NDArray[np.float64]] = []
    retained_ids: list[str] = []
    running_observations = 0
    for well in ordered:
        xy, u, q, interval_md = _well_observations(well, config)
        if len(q) == 0:
            continue
        running_observations += len(q)
        if running_observations > config.max_training_observations:
            raise StructuralFieldError(
                "resampled observations exceed max_training_observations"
            )
        well_index = len(retained_ids)
        xy_parts.append(xy)
        u_parts.append(u)
        q_parts.append(q)
        index_parts.append(np.full(len(q), well_index, dtype=np.int64))
        interval_parts.append(interval_md)
        retained_ids.append(well.well_id)
    if len(retained_ids) < 2:
        raise StructuralFieldError("fewer than two wells have usable lateral intervals")

    xy = np.concatenate(xy_parts)
    u = np.concatenate(u_parts)
    q = np.concatenate(q_parts)
    well_index = np.concatenate(index_parts)
    interval_md = np.concatenate(interval_parts)
    n_obs = len(q)
    n_wells = len(retained_ids)
    base_weight = np.empty(n_obs, dtype=float)
    for index in range(n_wells):
        rows = well_index == index
        retained_md = interval_md[rows]
        # Equal total mass per well; within a well, each retained segment is
        # weighted by represented MD so endpoint subdivision cannot dominate.
        base_weight[rows] = n_obs * retained_md / (n_wells * float(retained_md.sum()))
    return _Observations(
        xy=xy,
        u=u,
        q=q,
        well_index=well_index,
        well_ids=tuple(retained_ids),
        base_weight=base_weight,
    )


def _inducing_nodes(
    xy: NDArray[np.float64],
    config: FieldConfig,
) -> tuple[NDArray[np.float64], float]:
    origin = np.min(xy, axis=0)
    cell = float(config.inducing_cell_ft)
    for _ in range(20):
        keys = np.floor((xy - origin) / cell).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        # NumPy has shipped both 1-D and (n, 1) shapes for the axis= form of
        # return_inverse; bincount and add.at below require 1-D either way.
        inverse = np.reshape(inverse, -1)
        if len(unique) <= config.max_nodes:
            break
        cell *= max(1.05, np.sqrt(len(unique) / config.max_nodes) * 1.01)
    else:
        raise StructuralFieldError(
            "could not satisfy max_nodes with deterministic cells"
        )
    nodes = np.zeros((len(unique), 2), dtype=float)
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    np.add.at(nodes[:, 0], inverse, xy[:, 0])
    np.add.at(nodes[:, 1], inverse, xy[:, 1])
    nodes /= counts[:, None]
    order = np.lexsort((nodes[:, 1], nodes[:, 0]))
    return nodes[order], cell


def _compact_weights(
    points: NDArray[np.float64],
    nodes: NDArray[np.float64],
    config: FieldConfig,
    *,
    allow_training_nearest: bool,
) -> tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]
]:
    k = min(config.interpolation_neighbors, len(nodes))
    distances, indices = cKDTree(nodes).query(points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    scaled = distances / config.support_length_ft
    inside = scaled < 1.0
    weights = np.where(inside, (1.0 - scaled) ** 4 * (4.0 * scaled + 1.0), 0.0)
    totals = weights.sum(axis=1)
    supported = totals > 0.0
    if allow_training_nearest and not supported.all():
        rows = np.flatnonzero(~supported)
        weights[rows, 0] = 1.0
        totals[rows] = 1.0
    safe_totals = np.where(totals > 0.0, totals, 1.0)
    weights /= safe_totals[:, None]
    return indices.astype(np.int64), weights, supported, distances[:, 0]


def _design_matrix(
    u: NDArray[np.float64],
    node_indices: NDArray[np.int64],
    node_weights: NDArray[np.float64],
    n_nodes: int,
) -> csr_matrix:
    n_obs, k = node_indices.shape
    row = np.repeat(np.arange(n_obs, dtype=np.int64), 2 * k)
    columns = np.empty((n_obs, k, 2), dtype=np.int64)
    columns[:, :, 0] = 2 * node_indices
    columns[:, :, 1] = 2 * node_indices + 1
    data = np.empty((n_obs, k, 2), dtype=float)
    data[:, :, 0] = node_weights * u[:, [0]]
    data[:, :, 1] = node_weights * u[:, [1]]
    return sparse.coo_matrix(
        (data.reshape(-1), (row, columns.reshape(-1))),
        shape=(n_obs, 2 * n_nodes),
    ).tocsr()


def _ccw_face(
    nodes: NDArray[np.float64], triangle: NDArray[np.int64]
) -> tuple[int, int, int]:
    a, b, c = (int(value) for value in triangle)
    first = nodes[b] - nodes[a]
    second = nodes[c] - nodes[a]
    cross = first[0] * second[1] - first[1] * second[0]
    return (a, b, c) if cross > 0.0 else (a, c, b)


def _graph_geometry(
    nodes: NDArray[np.float64],
    config: FieldConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    edge_set: set[tuple[int, int]] = set()
    face_candidates: list[tuple[int, int, int]] = []
    if len(nodes) >= 3 and np.linalg.matrix_rank(nodes - nodes.mean(axis=0)) == 2:
        try:
            triangulation = Delaunay(nodes)
            for raw_face in triangulation.simplices:
                face = _ccw_face(nodes, raw_face)
                face_candidates.append(face)
                for start, end in (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                ):
                    edge_set.add((min(start, end), max(start, end)))
        except QhullError:
            face_candidates = []

    if len(nodes) >= 2:
        k = min(config.graph_neighbors + 1, len(nodes))
        distance, neighbor = cKDTree(nodes).query(nodes, k=k)
        if k == 1:
            distance = distance[:, None]
            neighbor = neighbor[:, None]
        for start in range(len(nodes)):
            for dist, end in zip(distance[start, 1:], neighbor[start, 1:], strict=True):
                if dist <= config.graph_max_edge_ft:
                    edge_set.add((min(start, int(end)), max(start, int(end))))

    kept_edges = [
        edge
        for edge in sorted(edge_set)
        if np.linalg.norm(nodes[edge[1]] - nodes[edge[0]]) <= config.graph_max_edge_ft
    ]
    edges = np.asarray(kept_edges, dtype=np.int64).reshape(-1, 2)
    edge_lookup = {edge: index for index, edge in enumerate(kept_edges)}
    faces = []
    for face in sorted(set(face_candidates)):
        keys = (
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0])),
        )
        if all(key in edge_lookup for key in keys):
            faces.append(face)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(edges):
        length = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
        conductance = np.exp(-((length / config.support_length_ft) ** 2))
    else:
        conductance = np.empty(0, dtype=float)
    return edges, face_array, conductance


def build_graph_incidence(n_nodes: int, edges: ArrayLike) -> csr_matrix:
    """Build oriented edge-node incidence ``B`` using stored edge orientation."""

    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if n_nodes < 1:
        raise StructuralFieldError("n_nodes must be positive")
    if len(edge_array) == 0:
        return sparse.csr_matrix((0, n_nodes), dtype=float)
    if edge_array.min() < 0 or edge_array.max() >= n_nodes:
        raise StructuralFieldError("edge endpoint outside node range")
    if np.any(edge_array[:, 0] == edge_array[:, 1]):
        raise StructuralFieldError("self edges are not allowed")
    rows = np.repeat(np.arange(len(edge_array)), 2)
    columns = edge_array.reshape(-1)
    values = np.tile(np.array([-1.0, 1.0]), len(edge_array))
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(edge_array), n_nodes)
    ).tocsr()


def build_graph_laplacian(
    n_nodes: int,
    edges: ArrayLike,
    conductance: ArrayLike,
) -> csr_matrix:
    """Return the explicitly defined scalar graph Laplacian ``B.T W B``."""

    incidence = build_graph_incidence(n_nodes, edges)
    weights = _as_vector("conductance", conductance, expected_length=incidence.shape[0])
    if np.any(weights < 0.0):
        raise StructuralFieldError("conductance must be nonnegative")
    return (incidence.T @ sparse.diags(weights) @ incidence).tocsr()


def build_edge_one_form_operator(nodes_xy: ArrayLike, edges: ArrayLike) -> csr_matrix:
    """Map nodal gradients to trapezoidal edge one-forms.

    For oriented edge ``e=(u,v)``, the row computes
    ``0.5 * (g_u + g_v) dot (r_v - r_u)``.
    """

    nodes = np.asarray(nodes_xy, dtype=float)
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    build_graph_incidence(len(nodes), edge_array)
    if len(edge_array) == 0:
        return sparse.csr_matrix((0, 2 * len(nodes)), dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for edge_index, (start, end) in enumerate(edge_array):
        delta = nodes[end] - nodes[start]
        for node in (int(start), int(end)):
            rows.extend((edge_index, edge_index))
            columns.extend((2 * node, 2 * node + 1))
            values.extend((0.5 * float(delta[0]), 0.5 * float(delta[1])))
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(edge_array), 2 * len(nodes))
    ).tocsr()


def build_face_edge_incidence(
    nodes_xy: ArrayLike,
    edges: ArrayLike,
    faces: ArrayLike,
) -> csr_matrix:
    """Build oriented face-edge incidence ``C``, normalized by perimeter."""

    nodes = np.asarray(nodes_xy, dtype=float)
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    lookup = {
        (int(min(start, end)), int(max(start, end))): index
        for index, (start, end) in enumerate(edge_array)
    }
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for face_index, raw_face in enumerate(face_array):
        face = _ccw_face(nodes, raw_face)
        perimeter = sum(
            float(np.linalg.norm(nodes[end] - nodes[start]))
            for start, end in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        )
        if not np.isfinite(perimeter) or perimeter <= 0.0:
            raise StructuralFieldError("face perimeter must be finite and positive")
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(start, end), max(start, end))
            if key not in lookup:
                raise StructuralFieldError("each face side must exist in edges")
            edge_index = lookup[key]
            stored_start, stored_end = edge_array[edge_index]
            sign = 1.0 if (stored_start, stored_end) == (start, end) else -1.0
            rows.append(face_index)
            columns.append(edge_index)
            values.append(sign / perimeter)
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(face_array), len(edge_array))
    ).tocsr()


def build_face_circulation_operator(
    nodes_xy: ArrayLike,
    edges: ArrayLike,
    faces: ArrayLike,
) -> csr_matrix:
    """Return the defined normalized circulation operator ``C @ P``."""

    return (
        build_face_edge_incidence(nodes_xy, edges, faces)
        @ build_edge_one_form_operator(nodes_xy, edges)
    ).tocsr()


def _regularization_operators(
    nodes: NDArray[np.float64],
    edges: NDArray[np.int64],
    conductance: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> tuple[csr_matrix, csr_matrix]:
    incidence = build_graph_incidence(len(nodes), edges)
    weighted_incidence = sparse.diags(np.sqrt(conductance)) @ incidence
    smooth = sparse.kron(weighted_incidence, sparse.eye(2), format="csr")
    circulation = build_face_circulation_operator(nodes, edges, faces)
    return smooth, circulation


def _weighted_median(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    index = min(
        int(np.searchsorted(np.cumsum(sorted_weights), cutoff)), len(values) - 1
    )
    return float(sorted_values[index])


def _mad_scale(
    values: NDArray[np.float64],
    *,
    floor: float = 1.0e-8,
    weights: NDArray[np.float64] | None = None,
) -> float:
    if len(values) == 0:
        return floor
    if weights is None:
        center = float(np.median(values))
        deviation = float(np.median(np.abs(values - center)))
    else:
        center = _weighted_median(values, weights)
        deviation = _weighted_median(np.abs(values - center), weights)
    scale = 1.4826 * deviation
    return max(floor, scale)


def _scaled_regularizer(
    operator: csr_matrix,
    strength: float,
    data_trace: float,
) -> csr_matrix:
    if strength <= 0.0 or operator.shape[0] == 0:
        return sparse.csr_matrix((0, operator.shape[1]), dtype=float)
    operator_trace = float(np.square(operator.data).sum())
    if operator_trace <= 0.0:
        return sparse.csr_matrix((0, operator.shape[1]), dtype=float)
    return operator * np.sqrt(strength * data_trace / operator_trace)


def _robust_solve(
    design: csr_matrix,
    target: NDArray[np.float64],
    base_weight: NDArray[np.float64],
    smooth: csr_matrix,
    circulation: csr_matrix,
    config: FieldConfig,
) -> tuple[NDArray[np.float64], float, int]:
    n_parameters = design.shape[1]
    solution = np.zeros(n_parameters, dtype=float)
    stop_code = 0
    scale = _mad_scale(target, weights=base_weight)
    base_weighted_design = sparse.diags(np.sqrt(base_weight)) @ design
    # Freeze trace normalization before IRLS; changing robust residual weights
    # must not silently retune the declared graph/circulation strengths.
    base_data_trace = max(float(np.square(base_weighted_design.data).sum()), 1.0e-12)
    smooth_scaled = _scaled_regularizer(
        smooth, config.laplacian_strength, base_data_trace
    )
    circulation_scaled = _scaled_regularizer(
        circulation, config.circulation_strength, base_data_trace
    )
    ridge_scale = np.sqrt(
        config.ridge_strength * base_data_trace / max(n_parameters, 1)
    )
    ridge = sparse.eye(n_parameters, format="csr") * ridge_scale
    for _ in range(config.irls_iterations):
        residual = target - design @ solution
        scale = _mad_scale(residual, weights=base_weight)
        standardized = np.abs(residual) / (config.huber_delta * scale)
        robust_weight = np.ones_like(standardized)
        outside = standardized > 1.0
        robust_weight[outside] = 1.0 / standardized[outside]
        row_weight = np.sqrt(base_weight * robust_weight)
        weighted_design = sparse.diags(row_weight) @ design
        weighted_target = row_weight * target
        augmented = sparse.vstack(
            (weighted_design, smooth_scaled, circulation_scaled, ridge),
            format="csr",
        )
        rhs = np.concatenate(
            (
                weighted_target,
                np.zeros(
                    smooth_scaled.shape[0] + circulation_scaled.shape[0] + n_parameters
                ),
            )
        )
        solved = lsmr(
            augmented,
            rhs,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=config.solver_max_iterations,
        )
        solution = np.asarray(solved[0], dtype=float)
        stop_code = int(solved[1])
        numerical_summary = np.asarray(solved[2:8], dtype=float)
        if stop_code not in {0, 1, 2, 4, 5}:
            raise StructuralFieldError(
                f"LSMR failed closed with unacceptable stop code {stop_code}"
            )
        if not np.isfinite(solution).all() or not np.isfinite(numerical_summary).all():
            raise StructuralFieldError("LSMR produced nonfinite solution diagnostics")
        solved_residual = target - design @ solution
        if not np.isfinite(solved_residual).all():
            raise StructuralFieldError("LSMR produced a nonfinite data residual")
    return solution, _mad_scale(solved_residual, weights=base_weight), stop_code


def _node_training_diagnostics(
    observations: _Observations,
    node_indices: NDArray[np.int64],
    node_weights: NDArray[np.float64],
    n_nodes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    gram = np.zeros((n_nodes, 2, 2), dtype=float)
    if len(observations.q) == 0 or node_indices.size == 0:
        return np.zeros(n_nodes, dtype=float), gram

    # Flatten (observation, node) pairs once. np.add.at applies in index order,
    # which is the same order the nested loop accumulated in, so the Gram sums
    # match the sequential version rather than merely approximating it.
    per_obs = node_indices.shape[1]
    obs_of = np.repeat(np.arange(len(observations.q)), per_obs)
    flat_nodes = np.asarray(node_indices).reshape(-1)
    flat_weights = np.asarray(node_weights).reshape(-1)

    keep = flat_weights > 0.0
    obs_of = obs_of[keep]
    flat_nodes = flat_nodes[keep].astype(np.int64)
    flat_weights = flat_weights[keep]

    u = observations.u[obs_of]
    weights = observations.base_weight[obs_of] * flat_weights
    np.add.at(gram, flat_nodes, weights[:, None, None] * (u[:, :, None] * u[:, None, :]))

    wells = np.asarray(observations.well_index)[obs_of].astype(np.int64)
    unique_pairs = np.unique(np.stack((flat_nodes, wells), axis=1), axis=0)
    effective_wells = np.bincount(
        unique_pairs[:, 0], minlength=n_nodes
    ).astype(float)
    return effective_wells, gram


def _cut_candidates(
    edges: NDArray[np.int64],
    first_pass_gradient: NDArray[np.float64],
    node_azimuth_gram: NDArray[np.float64],
    effective_wells: NDArray[np.float64],
    config: FieldConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    if len(edges) == 0:
        return np.empty(0, dtype=bool), np.empty(0, dtype=float)
    delta = first_pass_gradient[edges[:, 1]] - first_pass_gradient[edges[:, 0]]
    score = np.zeros(len(edges), dtype=float)
    for edge_index, (start, end) in enumerate(edges):
        local_gram = node_azimuth_gram[start] + node_azimuth_gram[end]
        trace = float(np.trace(local_gram))
        if trace > 0.0:
            # Only the component visible to nearby training directions can
            # establish an incompatibility; rank-null directions are ignored.
            score[edge_index] = np.sqrt(
                max(0.0, float(delta[edge_index] @ local_gram @ delta[edge_index]))
                / trace
            )
    # A single trajectory cannot establish a cross-well incompatibility.
    eligible = (effective_wells[edges[:, 0]] >= config.min_effective_wells) & (
        effective_wells[edges[:, 1]] >= config.min_effective_wells
    )
    eligible_score = score[eligible]
    if len(eligible_score) == 0:
        return np.zeros(len(edges), dtype=bool), score
    center = float(np.median(eligible_score))
    robust_scale = _mad_scale(eligible_score, floor=0.0)
    threshold = max(
        config.discontinuity_absolute_floor,
        center + config.discontinuity_mad_threshold * robust_scale,
    )
    return eligible & (score > threshold), score


def _faces_without_cut_edges(
    faces: NDArray[np.int64],
    edges: NDArray[np.int64],
    cut_mask: NDArray[np.bool_],
) -> NDArray[np.int64]:
    if len(faces) == 0 or not cut_mask.any():
        return faces.copy()
    cut_keys = {(int(start), int(end)) for start, end in edges[cut_mask]}
    kept = []
    for face in faces:
        sides = {
            (min(int(face[0]), int(face[1])), max(int(face[0]), int(face[1]))),
            (min(int(face[1]), int(face[2])), max(int(face[1]), int(face[2]))),
            (min(int(face[2]), int(face[0])), max(int(face[2]), int(face[0]))),
        }
        if not sides & cut_keys:
            kept.append(tuple(int(value) for value in face))
    return np.asarray(kept, dtype=np.int64).reshape(-1, 3)


def _component_labels(
    n_nodes: int,
    edges: NDArray[np.int64],
    keep_mask: NDArray[np.bool_],
) -> tuple[int, NDArray[np.int64]]:
    kept = edges[keep_mask]
    if len(kept) == 0:
        return n_nodes, np.arange(n_nodes, dtype=np.int64)
    values = np.ones(2 * len(kept), dtype=float)
    adjacency = sparse.coo_matrix(
        (
            values,
            (
                np.concatenate((kept[:, 0], kept[:, 1])),
                np.concatenate((kept[:, 1], kept[:, 0])),
            ),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    return int(count), labels.astype(np.int64)


def fit_structural_field(
    wells: Iterable[TrainingWell],
    config: FieldConfig | None = None,
) -> StructuralFieldModel:
    """Fit an inducing-node differential latent from labeled training wells.

    The solve performs two deterministic robust passes.  Pass one identifies
    training-only derivative-incompatibility candidates.  Pass two removes
    those graph restrictions, downweights other incompatible conductances,
    and refits.  Each well has equal total data weight.
    """

    settings = config if config is not None else FieldConfig()
    observations = _collect_observations(wells, settings)
    nodes, actual_cell = _inducing_nodes(observations.xy, settings)
    node_indices, node_weights, _, _ = _compact_weights(
        observations.xy,
        nodes,
        settings,
        allow_training_nearest=True,
    )
    design = _design_matrix(observations.u, node_indices, node_weights, len(nodes))
    edges, faces, conductance = _graph_geometry(nodes, settings)
    smooth, circulation = _regularization_operators(nodes, edges, conductance, faces)
    initial, _, initial_stop = _robust_solve(
        design,
        observations.q,
        observations.base_weight,
        smooth,
        circulation,
        settings,
    )
    effective_wells, gram = _node_training_diagnostics(
        observations, node_indices, node_weights, len(nodes)
    )
    initial_gradient = initial.reshape(-1, 2)
    cut_mask, incompatibility = _cut_candidates(
        edges, initial_gradient, gram, effective_wells, settings
    )
    retained = ~cut_mask
    retained_edges = edges[retained]
    retained_conductance = conductance[retained].copy()
    if len(retained_conductance):
        retained_score = incompatibility[retained]
        score_scale = max(
            settings.discontinuity_absolute_floor,
            float(np.median(retained_score)) + _mad_scale(retained_score, floor=0.0),
        )
        retained_conductance /= 1.0 + (retained_score / score_scale) ** 2
        retained_conductance = np.maximum(
            retained_conductance, 0.05 * conductance[retained]
        )
    retained_faces = _faces_without_cut_edges(faces, edges, cut_mask)
    smooth_final, circulation_final = _regularization_operators(
        nodes, retained_edges, retained_conductance, retained_faces
    )
    final, residual_scale, final_stop = _robust_solve(
        design,
        observations.q,
        observations.base_weight,
        smooth_final,
        circulation_final,
        settings,
    )
    gradient = final.reshape(-1, 2)
    residual = observations.q - design @ final
    weighted_rmse = float(
        np.sqrt(np.average(np.square(residual), weights=observations.base_weight))
    )
    smooth_energy = float(np.square(smooth_final @ final).sum())
    circulation_values = circulation_final @ final
    circulation_rms = (
        float(np.sqrt(np.mean(np.square(circulation_values))))
        if len(circulation_values)
        else 0.0
    )
    component_count, labels = _component_labels(len(nodes), edges, retained)
    final_edge_conductance = np.zeros(len(edges), dtype=float)
    final_edge_conductance[retained] = retained_conductance
    diagnostics = FitDiagnostics(
        wells=len(observations.well_ids),
        resampled_intervals=len(observations.q),
        inducing_nodes=len(nodes),
        graph_edges=len(edges),
        graph_faces=len(retained_faces),
        discontinuity_candidates=int(cut_mask.sum()),
        graph_components_after_cuts=component_count,
        actual_inducing_cell_ft=actual_cell,
        robust_passes=2,
        weighted_derivative_rmse=weighted_rmse,
        derivative_residual_scale=residual_scale,
        graph_smoothness_energy=smooth_energy,
        normalized_circulation_rms=circulation_rms,
        solver_stop_codes=(initial_stop, final_stop),
    )
    # ``initial`` is intentionally computed even though only the final pass is
    # returned: it establishes the fixed pass-one residual/conductance state.
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise StructuralFieldError("structural solve produced nonfinite parameters")
    return StructuralFieldModel(
        config=settings,
        nodes_xy=nodes,
        gradients_xy=gradient,
        edges=edges,
        edge_conductance=final_edge_conductance,
        faces=retained_faces,
        cut_edge_mask=cut_mask,
        node_components=labels,
        node_effective_wells=effective_wells,
        node_azimuth_gram=gram,
        support_xy=observations.xy,
        support_unit_u=(
            observations.u / np.linalg.norm(observations.u, axis=1)[:, None]
        ),
        support_well_index=observations.well_index,
        support_tree=cKDTree(observations.xy),
        diagnostics=diagnostics,
    )


def _known_prefix_rows(tvt_input: NDArray[np.float64]) -> int:
    if np.isinf(tvt_input).any():
        raise InferenceSafetyError(
            "TVT_input may contain finite prefix values and NaNs only"
        )
    finite = np.isfinite(tvt_input)
    missing = np.flatnonzero(np.isnan(tvt_input))
    if len(missing) == 0:
        raise InferenceSafetyError(
            "TVT_input must end in NaNs; fully finite target TVT could expose suffix truth"
        )
    prefix_rows = int(missing[0])
    if prefix_rows < 2:
        raise StructuralFieldError("TVT_input needs at least two finite prefix rows")
    if not finite[:prefix_rows].all() or not np.isnan(tvt_input[prefix_rows:]).all():
        raise InferenceSafetyError(
            "TVT_input must be one contiguous finite prefix followed only by NaNs"
        )
    return prefix_rows


def _interpolate_model(
    model: StructuralFieldModel,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    indices, weights, supported, _ = _compact_weights(
        points,
        model.nodes_xy,
        model.config,
        allow_training_nearest=False,
    )
    gradient = np.sum(model.gradients_xy[indices] * weights[:, :, None], axis=1)
    mixed_component = np.zeros(len(points), dtype=bool)
    for row in np.flatnonzero(supported):
        active = weights[row] > 1.0e-8
        components = np.unique(model.node_components[indices[row, active]])
        mixed_component[row] = len(components) > 1
    return gradient, supported & ~mixed_component


def _trajectory_directions(
    md: NDArray[np.float64], xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    direction = np.column_stack(
        (
            np.gradient(xy[:, 0], md),
            np.gradient(xy[:, 1], md),
        )
    )
    norm = np.linalg.norm(direction, axis=1)
    unit = np.zeros_like(direction)
    nonzero = norm > 0.0
    unit[nonzero] = direction[nonzero] / norm[nonzero, None]
    return unit


def _support_metrics(
    model: StructuralFieldModel,
    points: NDArray[np.float64],
    query_directions: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Measure trajectory distance, per-well n_eff, condition, observability.

    Resampled training intervals, not inducing centroids, define support.  A
    well contributes only its nearest local midpoint and compact-kernel mass.
    Thus dense or long sampling from one trajectory cannot masquerade as
    multiple wells.  Adaptive querying stops when the radius is exhausted or
    the declared nearest-M distinct wells have been captured; metrics are exact
    for that explicitly bounded support definition.
    """

    if query_directions.shape != points.shape:
        raise StructuralFieldError("query_directions must match points")
    support_count = len(model.support_xy)
    maximum_k = min(model.config.max_support_neighbors, support_count)
    nearest = np.full(len(points), np.inf, dtype=float)
    effective_wells = np.zeros(len(points), dtype=float)
    condition = np.full(len(points), np.inf, dtype=float)
    directional = np.zeros(len(points), dtype=float)
    declared_well_count = min(
        model.config.max_distinct_support_wells,
        int(np.unique(model.support_well_index).size),
    )
    for row, point in enumerate(points):
        k = min(32, maximum_k)
        while True:
            distances, indices = model.support_tree.query(point, k=k)
            distances = np.atleast_1d(np.asarray(distances, dtype=float))
            indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
            radius_exhausted = (
                k == support_count
                or float(distances[-1]) >= model.config.support_length_ft
            )
            inside_query = distances < model.config.support_length_ft
            distinct_wells = np.unique(
                model.support_well_index[indices[inside_query]]
            ).size
            sufficient_wells = distinct_wells >= declared_well_count
            if radius_exhausted or sufficient_wells or k == maximum_k:
                break
            k = min(2 * k, maximum_k)
        nearest[row] = float(distances[0])
        if not radius_exhausted and not sufficient_wells:
            # The bounded query could not see the complete support ball.  Do
            # not report truncated well diversity as if it were the defined
            # per-well n_eff; conservative policy fallback is the residue.
            continue
        inside = distances < model.config.support_length_ft
        if not inside.any():
            continue
        distances = distances[inside]
        indices = indices[inside]
        scaled = distances / model.config.support_length_ft
        kernel = (1.0 - scaled) ** 4 * (4.0 * scaled + 1.0)
        neighbor_wells = model.support_well_index[indices]
        nearest_wells: list[int] = []
        seen_wells: set[int] = set()
        for well_index in neighbor_wells:
            well_int = int(well_index)
            if well_int not in seen_wells:
                seen_wells.add(well_int)
                nearest_wells.append(well_int)
                if len(nearest_wells) == declared_well_count:
                    break
        well_strengths: list[float] = []
        gram = np.zeros((2, 2), dtype=float)
        for well_index in nearest_wells:
            same_well = neighbor_wells == well_index
            local_kernel = kernel[same_well]
            nearest_local = int(np.argmax(local_kernel))
            strength = float(local_kernel[nearest_local])
            if strength <= 0.0:
                continue
            local_unit = model.support_unit_u[indices[same_well]][nearest_local]
            gram += strength * np.outer(local_unit, local_unit)
            well_strengths.append(strength)
        if not well_strengths:
            continue
        strengths = np.asarray(well_strengths, dtype=float)
        effective_wells[row] = float(
            np.square(strengths.sum()) / np.square(strengths).sum()
        )
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[-1] <= 0.0:
            continue
        if eigenvalues[0] > eigenvalues[-1] * 1.0e-12:
            condition[row] = float(eigenvalues[-1] / eigenvalues[0])
        query = query_directions[row]
        query_norm = float(np.linalg.norm(query))
        if query_norm > 0.0:
            unit_query = query / query_norm
            directional[row] = float(
                np.clip(unit_query @ gram @ unit_query / eigenvalues[-1], 0.0, 1.0)
            )
    return nearest, effective_wells, condition, directional


def _support_confidence(
    model: StructuralFieldModel,
    nearest: NDArray[np.float64],
    effective_wells: NDArray[np.float64],
    directional: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    distance_factor = np.clip(1.0 - nearest / model.config.support_length_ft, 0.0, 1.0)
    well_factor = np.clip(effective_wells / model.config.min_effective_wells, 0.0, 1.0)
    direction_factor = np.clip(directional, 0.0, 1.0)
    supported = (
        (nearest < model.config.support_length_ft)
        & (effective_wells >= model.config.min_effective_wells)
        & (directional >= model.config.min_directional_observability)
    )
    return distance_factor * well_factor * direction_factor, supported


def _huber_location(values: NDArray[np.float64], delta: float) -> float:
    if len(values) == 0:
        return 0.0
    location = float(np.median(values))
    scale = _mad_scale(values)
    for _ in range(5):
        residual = values - location
        absolute = np.abs(residual)
        weights = np.ones_like(values)
        outside = absolute > delta * scale
        weights[outside] = delta * scale / absolute[outside]
        location = float(np.sum(weights * values) / np.sum(weights))
    return location


def _prefix_bias(
    model: StructuralFieldModel,
    md: NDArray[np.float64],
    xy: NDArray[np.float64],
    z: NDArray[np.float64],
    tvt_input: NDArray[np.float64],
    prefix_rows: int,
) -> tuple[float, int]:
    anchor_md = md[prefix_rows - 1]
    start = int(
        np.searchsorted(md, anchor_md - model.config.prefix_bias_window_md, side="left")
    )
    interval_start = np.arange(start, prefix_rows - 1)
    if len(interval_start) == 0:
        return 0.0, 0
    interval_end = interval_start + 1
    delta_md = md[interval_end] - md[interval_start]
    midpoint = (xy[interval_start] + xy[interval_end]) / 2.0
    gradient, interpolation_supported = _interpolate_model(model, midpoint)
    cut_fallback, _ = _cut_fallback_mask(model, midpoint)
    u = (xy[interval_end] - xy[interval_start]) / delta_md[:, None]
    nearest, effective_wells, _, directional = _support_metrics(model, midpoint, u)
    _, evidence_supported = _support_confidence(
        model, nearest, effective_wells, directional
    )
    structural = tvt_input[:prefix_rows] + z[:prefix_rows]
    q = (structural[interval_end] - structural[interval_start]) / delta_md
    residual = q - np.sum(gradient * u, axis=1)
    valid = (
        interpolation_supported
        & evidence_supported
        & ~cut_fallback
        & np.isfinite(residual)
    )
    # Bias must be supported continuously up to the anchor; an earlier valid
    # island cannot bridge a later evidence gap.
    valid &= np.logical_and.accumulate(valid[::-1])[::-1]
    if not valid.any():
        return 0.0, 0
    bias = _huber_location(residual[valid], model.config.huber_delta)
    bias *= model.config.prefix_bias_shrink
    bias = float(
        np.clip(
            bias, -model.config.max_abs_prefix_bias, model.config.max_abs_prefix_bias
        )
    )
    return bias, int(valid.sum())


def _segments_intersect(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> bool:
    def cross(
        p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]
    ) -> float:
        delta_q = q - p
        delta_r = r - p
        return float(delta_q[0] * delta_r[1] - delta_q[1] * delta_r[0])

    coordinate_scale = max(
        1.0,
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(c - a)),
        float(np.linalg.norm(d - a)),
    )
    linear_tolerance = 1.0e-10 * coordinate_scale
    orientation_tolerance = 1.0e-10 * coordinate_scale * coordinate_scale
    if (
        max(a[0], b[0]) + linear_tolerance < min(c[0], d[0])
        or max(c[0], d[0]) + linear_tolerance < min(a[0], b[0])
        or max(a[1], b[1]) + linear_tolerance < min(c[1], d[1])
        or max(c[1], d[1]) + linear_tolerance < min(a[1], b[1])
    ):
        return False
    ab_c = cross(a, b, c)
    ab_d = cross(a, b, d)
    cd_a = cross(c, d, a)
    cd_b = cross(c, d, b)
    opposite_ab = (
        ab_c <= orientation_tolerance and ab_d >= -orientation_tolerance
    ) or (ab_d <= orientation_tolerance and ab_c >= -orientation_tolerance)
    opposite_cd = (
        cd_a <= orientation_tolerance and cd_b >= -orientation_tolerance
    ) or (cd_b <= orientation_tolerance and cd_a >= -orientation_tolerance)
    return opposite_ab and opposite_cd


def _point_segment_distance(
    points: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> NDArray[np.float64]:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator == 0.0:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
    projection = start + fraction[:, None] * delta
    return np.linalg.norm(points - projection, axis=1)


def _cut_fallback_mask(
    model: StructuralFieldModel,
    xy: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], int]:
    mask = np.zeros(len(xy), dtype=bool)
    cut_edges = model.edges[model.cut_edge_mask]
    crossings = 0
    if len(xy) == 0 or len(cut_edges) == 0:
        return mask, crossings

    # The scan below is O(cut edges x rows). An edge whose bounding box misses
    # the trajectory's own box, grown by the fallback radius, can be neither
    # within that radius of any row nor crossed by any segment -- box gap on
    # some axis lower-bounds the true distance. Dropping those edges first
    # leaves both mask and crossings bit-identical.
    radius = float(model.config.cut_fallback_radius_ft)
    starts_all = model.nodes_xy[cut_edges[:, 0]]
    ends_all = model.nodes_xy[cut_edges[:, 1]]
    low = xy.min(axis=0) - radius
    high = xy.max(axis=0) + radius
    edge_low = np.minimum(starts_all, ends_all)
    edge_high = np.maximum(starts_all, ends_all)
    near = np.all((edge_high >= low) & (edge_low <= high), axis=1)

    for start, end in zip(starts_all[near], ends_all[near], strict=True):
        mask |= (
            _point_segment_distance(xy, start, end)
            <= model.config.cut_fallback_radius_ft
        )
        for row in range(1, len(xy)):
            if _segments_intersect(xy[row - 1], xy[row], start, end):
                crossings += 1
                # With no train-supported jump amplitude, integration beyond a
                # cut crossing has an unknown datum and cannot safely resume.
                mask[row - 1 :] = True
    return mask, crossings


def _prediction_knots(
    md: NDArray[np.float64],
    anchor_md: float,
    config: FieldConfig,
) -> NDArray[np.float64]:
    """Bound support/integration work on deterministic target MD knots."""

    start = max(float(md[0]), anchor_md - config.prefix_bias_window_md)
    step = config.resample_step_md
    first_multiple = np.ceil(start / step) * step
    interior = np.arange(first_multiple, md[-1] + 0.5 * step, step, dtype=float)
    interior = interior[(interior > start) & (interior < md[-1])]
    knots = np.unique(np.concatenate(([start, anchor_md], interior, [md[-1]])))
    if len(knots) > config.max_prediction_rows:
        raise StructuralFieldError("evaluation knots exceed max_prediction_rows")
    return knots


def predict_structural_field(
    model: StructuralFieldModel,
    *,
    md: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    tvt_input: ArrayLike,
    policy_tvt: ArrayLike,
    **unexpected: ArrayLike,
) -> StructuralPrediction:
    """Predict a target using only inference-visible arrays.

    The exact last finite ``TVT_input + Z`` value is the integration anchor.
    Support and integration are evaluated on deterministic 100-ft MD knots;
    field delta is interpolated to raw rows and confidence uses the conservative
    minimum of its bracketing knots.
    Confidence is a deterministic function of resampled training-trajectory
    distance, per-well effective support, query-direction observability,
    connected-component agreement, and cut-edge proximity.  Full azimuth
    condition is diagnostic only.  Zero confidence returns ``policy_tvt``.
    """

    if unexpected:
        forbidden = sorted(_FORBIDDEN_PREDICTION_INPUTS & unexpected.keys())
        if forbidden:
            raise InferenceSafetyError(
                f"prediction forbids target-label inputs: {', '.join(forbidden)}"
            )
        raise TypeError(
            f"unexpected prediction inputs: {', '.join(sorted(unexpected))}"
        )
    raw_md = _as_vector("MD", md)
    _validate_md(raw_md)
    raw_rows = len(raw_md)
    if raw_rows > model.config.max_prediction_rows:
        # The expensive path is knot-bounded below, so this cap protects output
        # allocation only.  It can be raised explicitly for a known corpus.
        raise StructuralFieldError("rows exceed max_prediction_rows")
    raw_x = _as_vector("X", x, expected_length=raw_rows)
    raw_y = _as_vector("Y", y, expected_length=raw_rows)
    raw_z = _as_vector("Z", z, expected_length=raw_rows)
    raw_input = _as_vector(
        "TVT_input", tvt_input, expected_length=raw_rows, finite=False
    )
    raw_policy = _as_vector("policy_tvt", policy_tvt, expected_length=raw_rows)
    raw_prefix_rows = _known_prefix_rows(raw_input)
    raw_anchor_index = raw_prefix_rows - 1
    anchor_md_value = float(raw_md[raw_anchor_index])

    md_array = _prediction_knots(raw_md, anchor_md_value, model.config)
    n_rows = len(md_array)
    x_array = np.interp(md_array, raw_md, raw_x)
    y_array = np.interp(md_array, raw_md, raw_y)
    z_array = np.interp(md_array, raw_md, raw_z)
    policy_array = np.interp(md_array, raw_md, raw_policy)
    prefix_rows = int(np.searchsorted(md_array, anchor_md_value, side="right"))
    input_array = np.full(n_rows, np.nan, dtype=float)
    input_array[:prefix_rows] = np.interp(
        md_array[:prefix_rows],
        raw_md[:raw_prefix_rows],
        raw_input[:raw_prefix_rows],
    )
    xy = np.column_stack((x_array, y_array))
    gradient, interpolation_supported = _interpolate_model(model, xy)
    query_direction = _trajectory_directions(md_array, xy)
    nearest, effective_wells, condition, directional = _support_metrics(
        model, xy, query_direction
    )
    base_confidence, evidence_supported = _support_confidence(
        model, nearest, effective_wells, directional
    )
    bias, bias_intervals = _prefix_bias(
        model, md_array, xy, z_array, input_array, prefix_rows
    )

    anchor_index = prefix_rows - 1
    anchor_s = float(input_array[anchor_index] + z_array[anchor_index])
    field_s_without_bias = policy_array + z_array
    field_s_without_bias[anchor_index] = anchor_s
    for row in range(prefix_rows, n_rows):
        delta_xy = xy[row] - xy[row - 1]
        trapezoid_gradient = 0.5 * (gradient[row - 1] + gradient[row])
        field_s_without_bias[row] = field_s_without_bias[row - 1] + float(
            trapezoid_gradient @ delta_xy
        )
    field_delta_without_bias = field_s_without_bias - z_array - policy_array
    prefix_bias_delta = np.zeros(n_rows, dtype=float)
    prefix_bias_delta[prefix_rows:] = bias * (
        md_array[prefix_rows:] - md_array[anchor_index]
    )
    field_delta_without_bias[:anchor_index] = 0.0
    field_tvt = policy_array + field_delta_without_bias + prefix_bias_delta
    field_tvt[:prefix_rows] = input_array[:prefix_rows]

    supported = interpolation_supported & evidence_supported
    cut_fallback = np.zeros(n_rows, dtype=bool)
    local_cut_fallback, crossings = _cut_fallback_mask(model, xy[anchor_index:])
    cut_fallback[anchor_index:] = local_cut_fallback
    # An anchored integral cannot resume after a support gap without a new
    # structural anchor.  Keep all downstream rows on policy in that case.
    path_supported = supported & ~cut_fallback
    continuous_support = np.logical_and.accumulate(path_supported[anchor_index:])
    evaluation_support_mask = np.zeros(n_rows, dtype=bool)
    evaluation_support_mask[anchor_index:] = continuous_support
    confidence = base_confidence * supported.astype(float)
    confidence[anchor_index:] *= continuous_support.astype(float)
    confidence[:prefix_rows] = 0.0
    invalid_downstream = ~continuous_support
    downstream_indices = anchor_index + np.flatnonzero(invalid_downstream)
    field_delta_without_bias[downstream_indices] = 0.0
    prefix_bias_delta[downstream_indices] = 0.0
    field_tvt[downstream_indices] = policy_array[downstream_indices]
    evaluation_suffix = slice(prefix_rows, None)
    suffix_condition = condition[evaluation_suffix]
    finite_suffix_condition = suffix_condition[np.isfinite(suffix_condition)]
    right = np.searchsorted(md_array, raw_md, side="left")
    right = np.clip(right, 0, len(md_array) - 1)
    left = np.maximum(right - 1, 0)
    # Confidence is the conservative minimum of bracketing knot values.  A
    # hard unsupported knot therefore suppresses its entire incoming raw-row
    # segment and cannot reactivate downstream.
    raw_confidence = np.minimum(confidence[left], confidence[right])
    raw_support_mask = evaluation_support_mask[left] & evaluation_support_mask[right]
    raw_field_delta_without_bias = np.interp(raw_md, md_array, field_delta_without_bias)
    raw_prefix_bias_delta = np.interp(raw_md, md_array, prefix_bias_delta)
    raw_field_delta_without_bias[~raw_support_mask] = 0.0
    raw_prefix_bias_delta[~raw_support_mask] = 0.0
    raw_field_tvt = raw_policy + raw_field_delta_without_bias + raw_prefix_bias_delta
    raw_predicted = raw_policy + model.config.blend_alpha * raw_confidence * (
        raw_field_tvt - raw_policy
    )
    raw_field_tvt[:raw_prefix_rows] = raw_input[:raw_prefix_rows]
    raw_predicted[:raw_prefix_rows] = raw_input[:raw_prefix_rows]
    raw_confidence[:raw_prefix_rows] = 0.0
    raw_field_delta_without_bias[:raw_prefix_rows] = 0.0
    raw_prefix_bias_delta[:raw_prefix_rows] = 0.0
    raw_support_mask[:raw_prefix_rows] = False
    raw_suffix = slice(raw_prefix_rows, None)
    diagnostics = PredictionDiagnostics(
        status="anchored_field_100ft_knots_with_policy_fallback",
        rows=raw_rows,
        evaluation_rows=n_rows,
        prefix_rows=raw_prefix_rows,
        suffix_rows=raw_rows - raw_prefix_rows,
        anchor_s=anchor_s,
        prefix_bias=bias,
        prefix_bias_intervals=bias_intervals,
        nearest_resampled_training_midpoint_distance_mean_ft=float(
            np.mean(nearest[evaluation_suffix])
        ),
        nearest_resampled_training_midpoint_distance_max_ft=float(
            np.max(nearest[evaluation_suffix])
        ),
        effective_well_support_mean=float(np.mean(effective_wells[evaluation_suffix])),
        azimuth_condition_median=(
            float(np.median(finite_suffix_condition))
            if len(finite_suffix_condition)
            else float("inf")
        ),
        query_direction_observability_mean=float(
            np.mean(directional[evaluation_suffix])
        ),
        cut_edge_crossings=crossings,
        fallback_fraction=float(np.mean(raw_confidence[raw_suffix] <= 1.0e-12)),
        mean_confidence=float(np.mean(raw_confidence[raw_suffix])),
        max_abs_field_policy_difference_tvt=float(
            np.max(np.abs(raw_field_tvt[raw_suffix] - raw_policy[raw_suffix]))
        ),
    )
    return StructuralPrediction(
        predicted_tvt=raw_predicted,
        field_tvt=raw_field_tvt,
        field_delta_without_prefix_bias_tvt=raw_field_delta_without_bias,
        prefix_bias_delta_tvt=raw_prefix_bias_delta,
        confidence=raw_confidence,
        support_mask=raw_support_mask,
        diagnostics=diagnostics,
    )


API_FIELDS: Mapping[str, tuple[str, ...]] = {
    "training": ("well_id", "MD", "X", "Y", "Z", "TVT"),
    "prediction": ("MD", "X", "Y", "Z", "TVT_input", "policy_tvt"),
}
