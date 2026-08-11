"""Inference-safe ordered gamma-ray transport for one horizontal well.

This module deliberately has no file I/O and no training-label inputs.  The
only TVT observations it accepts are a contiguous, finite prefix followed by
NaNs.  Formation surfaces, geology labels, images, and suffix TVT truth are
not part of the API and are rejected if supplied as keyword arguments.

The implementation freezes the research settings measured in August 2026:

* 13 ordered gamma-ray samples across +/-90 MD;
* both forward and reverse local traversal orientations;
* robust capped-Huber emissions with weight 4;
* a 0.5-TVT state grid and a Viterbi path that permits stay, positive, and
  negative transitions (hence crossings and reversals);
* 15-MD inference nodes and at most 10 state bins per transition.

``proposed_tvt_path`` is an inference-generated structural/geometry proposal.
It is a reference carrier, not a label.  Callers are responsible for ensuring
that it was not derived from suffix truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


WINDOW_HALF_WIDTH_MD = 90.0
WINDOW_SAMPLES = 13
EMISSION_WEIGHT = 4.0
SMOOTHING_SAMPLES = 7

STATE_STEP_TVT = 0.5
STATE_HALF_WIDTH_TVT = 150.0
NODE_STEP_MD = 15.0
MAX_TRANSITION_BINS = 10
TRANSITION_WEIGHT = 0.15
TRANSITION_SCALE_TVT = 1.2

ANCHOR_WEIGHT = 10.0
GEOMETRY_SIGMA_INITIAL = 1.5
GEOMETRY_SIGMA_GROWTH_PER_MD = 0.015
HUBER_CAP = 4.0
# The anchor prior runs its own, much looser cap than the emission term's
# HUBER_CAP; it is a separate frozen setting, not a reuse of that one.
ANCHOR_HUBER_CAP = 100.0

FROZEN_SETTINGS: Mapping[str, float | int] = {
    "window_half_width_md": WINDOW_HALF_WIDTH_MD,
    "window_samples": WINDOW_SAMPLES,
    "emission_weight": EMISSION_WEIGHT,
    "smoothing_samples": SMOOTHING_SAMPLES,
    "state_step_tvt": STATE_STEP_TVT,
    "state_half_width_tvt": STATE_HALF_WIDTH_TVT,
    "node_step_md": NODE_STEP_MD,
    "max_transition_bins": MAX_TRANSITION_BINS,
    "transition_weight": TRANSITION_WEIGHT,
    "transition_scale_tvt": TRANSITION_SCALE_TVT,
    "anchor_weight": ANCHOR_WEIGHT,
    "geometry_sigma_initial": GEOMETRY_SIGMA_INITIAL,
    "geometry_sigma_growth_per_md": GEOMETRY_SIGMA_GROWTH_PER_MD,
    "huber_cap": HUBER_CAP,
    "anchor_huber_cap": ANCHOR_HUBER_CAP,
}

_FORBIDDEN_INPUT_NAMES = frozenset(
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
        "png",
        "train_image",
    }
)


class InferenceSafetyError(ValueError):
    """Raised when an input would expose information unavailable at inference."""


@dataclass(frozen=True)
class TransportDiagnostics:
    """Label-free diagnostics for one corrected path."""

    status: str
    rows: int
    prefix_rows: int
    suffix_rows: int
    viterbi_nodes: int
    states: int
    observed_gr_fraction: float
    observed_window_nodes: int
    calibration_pairs: int
    calibration_slope: float
    calibration_offset: float
    calibration_residual_scale: float
    forward_orientation_nodes: int
    reverse_orientation_nodes: int
    orientation_tie_nodes: int
    stay_steps: int
    positive_tvt_steps: int
    negative_tvt_steps: int
    reversal_count: int
    boundary_state_nodes: int
    final_path_cost: float
    mean_abs_correction_tvt: float
    max_abs_correction_tvt: float
    settings: tuple[tuple[str, float | int], ...]


@dataclass(frozen=True)
class TransportResult:
    """Corrected path and inference-safe diagnostics."""

    corrected_tvt: NDArray[np.float64]
    diagnostics: TransportDiagnostics


def robust_huber_cost(values: ArrayLike, cap: float = HUBER_CAP) -> NDArray[np.float64]:
    """Return a capped unit-threshold Huber cost elementwise."""

    z = np.abs(np.asarray(values, dtype=float))
    cost = np.where(z <= 1.0, 0.5 * z * z, z - 0.5)
    return np.minimum(cost, float(cap))


def _as_vector(name: str, values: ArrayLike, expected_length: int | None = None) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(result)}")
    return result.copy()


def _validate_known_prefix(known_prefix_tvt: NDArray[np.float64]) -> int:
    finite = np.isfinite(known_prefix_tvt)
    missing = np.flatnonzero(~finite)
    if len(missing) == 0:
        raise InferenceSafetyError(
            "known_prefix_tvt must end in NaNs; a fully finite TVT vector could expose suffix truth"
        )
    prefix_rows = int(missing[0])
    if prefix_rows < 2:
        raise ValueError("known_prefix_tvt needs at least two finite prefix rows")
    if not finite[:prefix_rows].all() or finite[prefix_rows:].any():
        raise InferenceSafetyError(
            "known_prefix_tvt must be one contiguous finite prefix followed only by NaNs"
        )
    return prefix_rows


def _sort_unique_curve(
    typewell_tvt: NDArray[np.float64], typewell_gr: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    valid = np.isfinite(typewell_tvt) & np.isfinite(typewell_gr)
    if valid.sum() < 2:
        raise ValueError("typewell needs at least two finite TVT/GR pairs")
    tvt = typewell_tvt[valid]
    gr = typewell_gr[valid]
    order = np.argsort(tvt, kind="stable")
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, inverse = np.unique(tvt, return_inverse=True)
    if len(unique_tvt) != len(tvt):
        sums = np.zeros(len(unique_tvt), dtype=float)
        counts = np.zeros(len(unique_tvt), dtype=float)
        np.add.at(sums, inverse, gr)
        np.add.at(counts, inverse, 1.0)
        gr = sums / counts
        tvt = unique_tvt
    if len(tvt) < 2 or not np.all(np.diff(tvt) > 0.0):
        raise ValueError("typewell TVT must contain at least two distinct values")
    return tvt, gr


def _fill_and_smooth_gr(horizontal_gr: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    valid = np.isfinite(horizontal_gr)
    if not valid.any():
        return np.zeros_like(horizontal_gr), valid
    indices = np.arange(len(horizontal_gr), dtype=float)
    filled = np.interp(indices, indices[valid], horizontal_gr[valid])
    radius = SMOOTHING_SAMPLES // 2
    padded = np.pad(filled, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, SMOOTHING_SAMPLES)
    return np.median(windows, axis=1), valid


def _calibrate_typewell_gr(
    horizontal_gr: NDArray[np.float64],
    known_prefix_tvt: NDArray[np.float64],
    prefix_rows: int,
    typewell_tvt: NDArray[np.float64],
    typewell_gr: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int, float, float, float]:
    type_at_prefix = np.interp(known_prefix_tvt[:prefix_rows], typewell_tvt, typewell_gr)
    observed = horizontal_gr[:prefix_rows]
    valid = np.isfinite(observed) & np.isfinite(type_at_prefix)
    pairs = int(valid.sum())
    if pairs >= 2:
        design = np.column_stack((type_at_prefix[valid], np.ones(pairs)))
        slope = float(np.linalg.lstsq(design, observed[valid], rcond=None)[0][0])
        slope = float(np.clip(slope, 0.2, 3.0))
        offset = float(np.median(observed[valid] - slope * type_at_prefix[valid]))
        residual = observed[valid] - (slope * type_at_prefix[valid] + offset)
        median_residual = float(np.median(residual))
        residual_scale = max(5.0, 1.4826 * float(np.median(np.abs(residual - median_residual))))
    else:
        slope = 1.0
        offset = 0.0
        residual_scale = 20.0
    return slope * typewell_gr + offset, pairs, slope, offset, residual_scale


def _node_indices(horizontal_md: NDArray[np.float64], suffix_start: int) -> NDArray[np.int64]:
    target_md = np.arange(horizontal_md[suffix_start], horizontal_md[-1] + NODE_STEP_MD, NODE_STEP_MD)
    indices = np.searchsorted(horizontal_md, target_md, side="left")
    indices = np.clip(indices, suffix_start, len(horizontal_md) - 1)
    return np.unique(np.concatenate((np.array([suffix_start]), indices, np.array([len(horizontal_md) - 1]))))


def _ordered_emission(
    state_tvt: NDArray[np.float64],
    center_index: int,
    horizontal_md: NDArray[np.float64],
    smoothed_gr: NDArray[np.float64],
    gr_valid: NDArray[np.bool_],
    proposed_tvt_path: NDArray[np.float64],
    typewell_tvt: NDArray[np.float64],
    calibrated_typewell_gr: NDArray[np.float64],
    residual_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.int8], bool]:
    offsets = np.linspace(-WINDOW_HALF_WIDTH_MD, WINDOW_HALF_WIDTH_MD, WINDOW_SAMPLES)
    positions = np.clip(
        horizontal_md[center_index] + offsets,
        horizontal_md[0],
        horizontal_md[-1],
    )
    observed = np.interp(positions, horizontal_md, smoothed_gr)
    observed_support = np.interp(positions, horizontal_md, gr_valid.astype(float)) >= 0.5
    if not observed_support.any():
        return np.zeros(len(state_tvt)), np.zeros(len(state_tvt), dtype=np.int8), False

    proposed_window = np.interp(positions, horizontal_md, proposed_tvt_path)
    local_delta = proposed_window - proposed_tvt_path[center_index]
    forward_tvt = state_tvt[:, None] + local_delta[None, :]
    reverse_tvt = state_tvt[:, None] - local_delta[None, :]
    forward_gr = np.interp(forward_tvt.ravel(), typewell_tvt, calibrated_typewell_gr).reshape(
        forward_tvt.shape
    )
    reverse_gr = np.interp(reverse_tvt.ravel(), typewell_tvt, calibrated_typewell_gr).reshape(
        reverse_tvt.shape
    )
    support = observed_support.astype(float)[None, :]
    denominator = float(observed_support.sum())
    forward_cost = np.sum(
        robust_huber_cost((observed[None, :] - forward_gr) / residual_scale) * support,
        axis=1,
    ) / denominator
    reverse_cost = np.sum(
        robust_huber_cost((observed[None, :] - reverse_gr) / residual_scale) * support,
        axis=1,
    ) / denominator
    tied = np.isclose(forward_cost, reverse_cost, rtol=1e-10, atol=1e-12)
    orientation = np.where(tied, 0, np.where(forward_cost < reverse_cost, 1, -1)).astype(np.int8)
    return np.minimum(forward_cost, reverse_cost), orientation, True


def ordered_reversible_interval_transport(
    horizontal_md: ArrayLike,
    horizontal_gr: ArrayLike,
    typewell_tvt: ArrayLike,
    typewell_gr: ArrayLike,
    known_prefix_tvt: ArrayLike,
    proposed_tvt_path: ArrayLike,
    **unsafe_inputs: object,
) -> TransportResult:
    """Correct one proposed TVT path using ordered interval-scale GR evidence.

    Parameters are arrays for exactly one well. ``known_prefix_tvt`` must be a
    contiguous finite prefix followed by NaNs. The complete horizontal GR log
    is observable at inference and may therefore be finite in the suffix.

    The function rejects suffix truth, formations, geology, and image-derived
    inputs. It never calculates a truth-based score; diagnostics compare only
    the corrected path with the proposed inference path.
    """

    if unsafe_inputs:
        names = set(unsafe_inputs)
        forbidden = sorted(names & _FORBIDDEN_INPUT_NAMES)
        if forbidden:
            raise InferenceSafetyError(
                "inference-unsafe inputs are forbidden: " + ", ".join(forbidden)
            )
        raise TypeError("unexpected keyword inputs: " + ", ".join(sorted(names)))

    md = _as_vector("horizontal_md", horizontal_md)
    if len(md) < 3:
        raise ValueError("horizontal well needs at least three rows")
    if not np.isfinite(md).all() or not np.all(np.diff(md) > 0.0):
        raise ValueError("horizontal_md must be finite and strictly increasing")
    gr = _as_vector("horizontal_gr", horizontal_gr, len(md))
    prefix = _as_vector("known_prefix_tvt", known_prefix_tvt, len(md))
    proposal = _as_vector("proposed_tvt_path", proposed_tvt_path, len(md))
    if not np.isfinite(proposal).all():
        raise ValueError("proposed_tvt_path must be finite for all rows")
    prefix_rows = _validate_known_prefix(prefix)

    tw_tvt = _as_vector("typewell_tvt", typewell_tvt)
    tw_gr = _as_vector("typewell_gr", typewell_gr, len(tw_tvt))
    tw_tvt, tw_gr = _sort_unique_curve(tw_tvt, tw_gr)
    calibrated_tw_gr, pairs, slope, offset, residual_scale = _calibrate_typewell_gr(
        gr, prefix, prefix_rows, tw_tvt, tw_gr
    )
    smoothed_gr, gr_valid = _fill_and_smooth_gr(gr)

    anchor = float(prefix[prefix_rows - 1])
    lower = max(float(tw_tvt[0]), anchor - STATE_HALF_WIDTH_TVT)
    upper = min(float(tw_tvt[-1]), anchor + STATE_HALF_WIDTH_TVT)
    lower = np.ceil(lower / STATE_STEP_TVT) * STATE_STEP_TVT
    upper = np.floor(upper / STATE_STEP_TVT) * STATE_STEP_TVT
    state_tvt = np.arange(lower, upper + 0.5 * STATE_STEP_TVT, STATE_STEP_TVT)
    if len(state_tvt) < 2:
        raise ValueError("typewell TVT does not overlap the anchored state window")

    suffix_start = prefix_rows
    nodes = _node_indices(md, suffix_start)
    state_count = len(state_tvt)
    transition_bins = np.arange(-MAX_TRANSITION_BINS, MAX_TRANSITION_BINS + 1)
    source = np.arange(state_count)[None, :] - transition_bins[:, None]
    source_valid = (source >= 0) & (source < state_count)
    source_clipped = np.clip(source, 0, state_count - 1)

    previous_cost = ANCHOR_WEIGHT * robust_huber_cost(
        (state_tvt - anchor) / STATE_STEP_TVT, cap=ANCHOR_HUBER_CAP
    )
    backpointer = np.empty((len(nodes), state_count), dtype=np.int32)
    orientation_by_state = np.empty((len(nodes), state_count), dtype=np.int8)
    observed_node = np.zeros(len(nodes), dtype=bool)

    for node_number, row in enumerate(nodes):
        prior_reference = anchor if node_number == 0 else proposal[nodes[node_number - 1]]
        proposed_increment = proposal[row] - prior_reference
        transition_cost = previous_cost[source_clipped] + TRANSITION_WEIGHT * (
            (STATE_STEP_TVT * transition_bins[:, None] - proposed_increment)
            / TRANSITION_SCALE_TVT
        ) ** 2
        transition_cost[~source_valid] = np.inf
        best_transition = np.argmin(transition_cost, axis=0)
        column = np.arange(state_count)
        best_cost = transition_cost[best_transition, column]
        best_source = source_clipped[best_transition, column]

        elapsed_md = max(0.0, float(md[row] - md[prefix_rows - 1]))
        geometry_sigma = GEOMETRY_SIGMA_INITIAL + GEOMETRY_SIGMA_GROWTH_PER_MD * elapsed_md
        unary = robust_huber_cost((state_tvt - proposal[row]) / geometry_sigma)
        emission, orientation, observed = _ordered_emission(
            state_tvt,
            int(row),
            md,
            smoothed_gr,
            gr_valid,
            proposal,
            tw_tvt,
            calibrated_tw_gr,
            residual_scale,
        )
        unary = unary + EMISSION_WEIGHT * emission
        previous_cost = best_cost + unary
        backpointer[node_number] = best_source
        orientation_by_state[node_number] = orientation
        observed_node[node_number] = observed

    node_states = np.empty(len(nodes), dtype=np.int32)
    node_states[-1] = int(np.argmin(previous_cost))
    for node_number in range(len(nodes) - 1, 0, -1):
        node_states[node_number - 1] = backpointer[node_number, node_states[node_number]]
    node_tvt = state_tvt[node_states]

    corrected = proposal.copy()
    corrected[:prefix_rows] = prefix[:prefix_rows]
    corrected[prefix_rows:] = np.interp(md[prefix_rows:], md[nodes], node_tvt)

    chosen_orientation = orientation_by_state[np.arange(len(nodes)), node_states]
    step = np.diff(node_tvt)
    tolerance = 0.5 * STATE_STEP_TVT
    step_sign = np.where(step > tolerance, 1, np.where(step < -tolerance, -1, 0))
    nonzero_sign = step_sign[step_sign != 0]
    reversal_count = int(np.sum(nonzero_sign[1:] != nonzero_sign[:-1])) if len(nonzero_sign) > 1 else 0
    suffix_correction = corrected[prefix_rows:] - proposal[prefix_rows:]
    diagnostics = TransportDiagnostics(
        status="ok" if observed_node.any() else "geometry_only_no_observed_gr",
        rows=len(md),
        prefix_rows=prefix_rows,
        suffix_rows=len(md) - prefix_rows,
        viterbi_nodes=len(nodes),
        states=state_count,
        observed_gr_fraction=float(np.mean(gr_valid)),
        observed_window_nodes=int(observed_node.sum()),
        calibration_pairs=pairs,
        calibration_slope=slope,
        calibration_offset=offset,
        calibration_residual_scale=residual_scale,
        forward_orientation_nodes=int(np.sum(chosen_orientation == 1)),
        reverse_orientation_nodes=int(np.sum(chosen_orientation == -1)),
        orientation_tie_nodes=int(np.sum(chosen_orientation == 0)),
        stay_steps=int(np.sum(step_sign == 0)),
        positive_tvt_steps=int(np.sum(step_sign > 0)),
        negative_tvt_steps=int(np.sum(step_sign < 0)),
        reversal_count=reversal_count,
        boundary_state_nodes=int(np.sum((node_states == 0) | (node_states == state_count - 1))),
        final_path_cost=float(previous_cost[node_states[-1]]),
        mean_abs_correction_tvt=float(np.mean(np.abs(suffix_correction))),
        max_abs_correction_tvt=float(np.max(np.abs(suffix_correction))),
        settings=tuple(FROZEN_SETTINGS.items()),
    )
    return TransportResult(corrected_tvt=corrected, diagnostics=diagnostics)


__all__ = [
    "FROZEN_SETTINGS",
    "InferenceSafetyError",
    "TransportDiagnostics",
    "TransportResult",
    "ordered_reversible_interval_transport",
    "robust_huber_cost",
]
