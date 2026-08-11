"""Small deterministic checks for the interval measurement instrument."""
from pathlib import Path

import numpy as np

from research.interval_gate import (
    EXCLUDED_TEST_OVERLAP,
    OFFSETS,
    _assert_inference_safe_feature_surface,
    _interp_atlas,
    _landscape,
    _make_atlas,
    _runtime_mutation_audit,
)


def test_repeated_identical_md_samples_do_not_change_cell_quartiles() -> None:
    single = _make_atlas(
        np.array([10.0, 11.0, 12.0]), np.array([40.0, 80.0, 60.0]), 1.0
    )
    repeated = _make_atlas(
        np.r_[np.repeat(10.0, 500), 11.0, 12.0],
        np.r_[np.repeat(40.0, 500), 80.0, 60.0],
        1.0,
    )
    np.testing.assert_allclose(single.quartiles, repeated.quartiles)
    assert repeated.counts[0] == 500


def test_interval_landscape_recovers_known_shift() -> None:
    reference_tvt = np.arange(0.0, 100.01, 0.5)
    reference_gr = (
        70.0
        + 15.0 * np.sin(reference_tvt / 4.0)
        + 8.0 * (reference_tvt > 45.0)
        - 12.0 * (reference_tvt > 68.0)
    )
    base_tvt = np.arange(20.0, 75.01, 1.0)
    true_shift = 6.0
    horizontal_gr = np.interp(base_tvt + true_shift, reference_tvt, reference_gr)
    query = _make_atlas(base_tvt, horizontal_gr, 1.0)
    reference = _make_atlas(reference_tvt, reference_gr, 1.0)
    costs, _ = _landscape(query, reference, width=1.0, scale=20.0)
    recovered = OFFSETS[int(np.argmin(costs))]
    assert abs(recovered - true_shift) <= 0.5


def test_interpolation_does_not_bridge_large_missing_interval() -> None:
    atlas = _make_atlas(
        np.array([0.0, 1.0, 20.0, 21.0]),
        np.array([10.0, 11.0, 30.0, 31.0]),
        1.0,
    )
    # This point is close to the left endpoint but is bracketed by a 19-ft gap.
    assert np.isnan(_interp_atlas(atlas, np.array([2.0]), 1.0)).all()


def test_leakage_guards_are_present_and_feature_surface_passes() -> None:
    assert EXCLUDED_TEST_OVERLAP == {"000d7d20", "00bbac68", "00e12e8b"}
    _assert_inference_safe_feature_surface()


def test_real_well_features_ignore_mutated_targets_and_surfaces() -> None:
    path = Path(r"C:\PRIMEdEV-1\GeoSteerN-Codex\train\015fe0d2__horizontal_well.csv")
    if path.exists():
        _runtime_mutation_audit(str(path))
