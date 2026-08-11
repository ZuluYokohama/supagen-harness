"""Deterministic tests for the inference-safe ordered transport research path."""

from __future__ import annotations

import numpy as np
import pytest

from research.ordered_transport import (
    FROZEN_SETTINGS,
    InferenceSafetyError,
    ordered_reversible_interval_transport,
)


def _typewell_curve() -> tuple[np.ndarray, np.ndarray]:
    tvt = np.arange(20.0, 140.01, 0.25)
    gr = (
        72.0
        + 18.0 * np.sin(0.31 * tvt)
        + 9.0 * np.sin(0.083 * tvt + 0.4)
        + 14.0 * np.exp(-0.5 * ((tvt - 73.0) / 2.3) ** 2)
        - 11.0 * np.exp(-0.5 * ((tvt - 98.0) / 3.1) ** 2)
    )
    return tvt, gr


def _reversing_well() -> tuple[np.ndarray, ...]:
    md = np.arange(0.0, 361.0)
    cutoff = 121
    prefix_tvt = 58.0 + 0.085 * md[:cutoff]
    suffix_md = md[cutoff:] - md[cutoff - 1]
    # Traverse down-section, reverse, and then traverse up-section.
    suffix_tvt = prefix_tvt[-1] + np.where(
        suffix_md <= 105.0,
        0.075 * suffix_md,
        0.075 * 105.0 - 0.095 * (suffix_md - 105.0),
    )
    latent_tvt = np.concatenate((prefix_tvt, suffix_tvt))
    proposed = latent_tvt.copy()
    proposed[cutoff:] += np.linspace(0.0, 10.0, len(md) - cutoff)
    known = np.full(len(md), np.nan)
    known[:cutoff] = latent_tvt[:cutoff]
    type_tvt, type_gr = _typewell_curve()
    horizontal_gr = np.interp(latent_tvt, type_tvt, type_gr)
    horizontal_gr += 0.35 * np.sin(0.17 * md)  # deterministic lateral/tool variation
    return md, horizontal_gr, type_tvt, type_gr, known, proposed, latent_tvt


def test_frozen_settings_and_prefix_are_preserved() -> None:
    md, gr, tw_tvt, tw_gr, known, proposed, latent = _reversing_well()
    result = ordered_reversible_interval_transport(md, gr, tw_tvt, tw_gr, known, proposed)
    cutoff = np.flatnonzero(~np.isfinite(known))[0]

    assert FROZEN_SETTINGS["window_samples"] == 13
    assert FROZEN_SETTINGS["window_half_width_md"] == 90.0
    assert FROZEN_SETTINGS["emission_weight"] == 4.0
    np.testing.assert_array_equal(result.corrected_tvt[:cutoff], known[:cutoff])
    assert result.diagnostics.prefix_rows == cutoff
    assert result.diagnostics.observed_window_nodes == result.diagnostics.viterbi_nodes
    assert result.diagnostics.positive_tvt_steps > 0
    assert result.diagnostics.negative_tvt_steps > 0
    assert result.diagnostics.reversal_count >= 1

    proposal_rmse = np.sqrt(np.mean((proposed[cutoff:] - latent[cutoff:]) ** 2))
    corrected_rmse = np.sqrt(np.mean((result.corrected_tvt[cutoff:] - latent[cutoff:]) ** 2))
    assert corrected_rmse < proposal_rmse


def test_reverse_orientation_is_available_and_used() -> None:
    md = np.arange(0.0, 301.0)
    cutoff = 101
    prefix = 65.0 + 0.08 * md[:cutoff]
    true_suffix = prefix[-1] - 0.08 * (md[cutoff:] - md[cutoff - 1])
    latent = np.concatenate((prefix, true_suffix))
    proposal = np.concatenate(
        (prefix, prefix[-1] + 0.08 * (md[cutoff:] - md[cutoff - 1]))
    )
    known = np.full(len(md), np.nan)
    known[:cutoff] = prefix
    tw_tvt, tw_gr = _typewell_curve()
    gr = np.interp(latent, tw_tvt, tw_gr)

    result = ordered_reversible_interval_transport(md, gr, tw_tvt, tw_gr, known, proposal)
    assert result.diagnostics.reverse_orientation_nodes > 0
    assert result.diagnostics.negative_tvt_steps > 0
    assert result.diagnostics.max_abs_correction_tvt > 0.0


def test_suffix_truth_and_geology_inputs_are_forbidden() -> None:
    md, gr, tw_tvt, tw_gr, known, proposed, latent = _reversing_well()
    with pytest.raises(InferenceSafetyError, match="suffix_truth"):
        ordered_reversible_interval_transport(
            md,
            gr,
            tw_tvt,
            tw_gr,
            known,
            proposed,
            suffix_truth=latent,
        )
    with pytest.raises(InferenceSafetyError, match="formation_surfaces"):
        ordered_reversible_interval_transport(
            md,
            gr,
            tw_tvt,
            tw_gr,
            known,
            proposed,
            formation_surfaces=np.zeros_like(md),
        )
    with pytest.raises(InferenceSafetyError, match="geology"):
        ordered_reversible_interval_transport(
            md,
            gr,
            tw_tvt,
            tw_gr,
            known,
            proposed,
            geology=np.array(["unit"] * len(md)),
        )


def test_fully_finite_tvt_vector_is_rejected_as_possible_truth() -> None:
    md, gr, tw_tvt, tw_gr, _, proposed, latent = _reversing_well()
    with pytest.raises(InferenceSafetyError, match="suffix truth"):
        ordered_reversible_interval_transport(md, gr, tw_tvt, tw_gr, latent, proposed)
