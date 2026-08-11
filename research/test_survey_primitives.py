"""Guards for the windowed coordinate -> directional survey primitives.

Two failure modes would silently poison everything downstream and are tested
hardest: a window short enough to measure coordinate rounding instead of
geometry, and an inclination sign fold that maps toe-up onto toe-down.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from research.survey_primitives import (
    COORD_QUANTUM_FT,
    DEFAULT_WINDOW_FT,
    MIN_WINDOW_FT,
    SurveyPrimitiveError,
    azimuth_error_multiplier,
    derive_section_azimuth,
    unwrap_azimuth,
    vertical_section,
    windowed_survey,
)

N = 400


def _straight(direction: tuple[float, float, float], n: int = N):
    """Unit-speed straight hole along `direction`, on a 1 ft MD grid."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    md = np.arange(n, dtype=float)
    pts = md[:, None] * d[None, :]
    return md, pts[:, 0], pts[:, 1], pts[:, 2]


# --------------------------------------------------------------------------
# the two poisoning failure modes
# --------------------------------------------------------------------------

def test_sub_window_request_is_refused_with_the_artifact_it_would_fabricate():
    md, x, y, z = _straight((0, 1, 0))
    with pytest.raises(SurveyPrimitiveError, match="below MIN_WINDOW_FT"):
        windowed_survey(md, x, y, z, window_ft=1)


@pytest.mark.parametrize("window", [1, 5, MIN_WINDOW_FT - 1])
def test_every_window_under_the_floor_is_refused(window):
    md, x, y, z = _straight((0, 1, 0))
    with pytest.raises(SurveyPrimitiveError):
        windowed_survey(md, x, y, z, window_ft=window)


def test_toe_up_inclination_exceeds_ninety_and_is_not_folded():
    """A hole climbing while near-horizontal must read >90, never its mirror."""
    md, x, y, z = _straight((0, np.cos(np.radians(2.0)), np.sin(np.radians(2.0))))
    s = windowed_survey(md, x, y, z)
    assert s.inclination_deg.min() > 90.0
    assert np.allclose(s.inclination_deg, 92.0, atol=1e-6)


def test_toe_down_and_toe_up_are_distinguishable_not_mirrored():
    up = windowed_survey(*_straight((0, np.cos(np.radians(2.0)), np.sin(np.radians(2.0)))))
    dn = windowed_survey(*_straight((0, np.cos(np.radians(2.0)), -np.sin(np.radians(2.0)))))
    assert up.inclination_deg.mean() > 90.0 > dn.inclination_deg.mean()
    assert not np.isclose(up.inclination_deg.mean(), dn.inclination_deg.mean())


# --------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------

def test_vertical_hole_reads_zero_inclination():
    s = windowed_survey(*_straight((0, 0, -1)))
    assert np.allclose(s.inclination_deg, 0.0, atol=1e-9)


def test_horizontal_hole_reads_ninety_inclination():
    s = windowed_survey(*_straight((0, 1, 0)))
    assert np.allclose(s.inclination_deg, 90.0, atol=1e-9)


@pytest.mark.parametrize(
    "direction,expected",
    [((0, 1, 0), 0.0), ((1, 0, 0), 90.0), ((0, -1, 0), 180.0), ((-1, 0, 0), 270.0)],
)
def test_azimuth_is_compass_from_grid_north(direction, expected):
    s = windowed_survey(*_straight(direction))
    assert np.allclose(s.azimuth_deg, expected, atol=1e-9)


def test_straight_hole_has_no_dogleg():
    s = windowed_survey(*_straight((0, 1, 0)))
    assert np.allclose(s.dls_deg_per_100ft, 0.0, atol=1e-9)


def test_circular_arc_recovers_its_build_rate():
    """A known-curvature arc must report that curvature, not the window's."""
    target = 8.0                              # deg/100ft
    radius = np.degrees(1.0) / (target / 100.0)
    md = np.arange(N, dtype=float)
    theta = md / radius
    x = np.zeros_like(md)
    y = radius * np.sin(theta)
    z = -radius * np.cos(theta)
    s = windowed_survey(md, x, y, z)
    assert np.allclose(np.median(s.dls_deg_per_100ft[1:]), target, rtol=0.02)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_non_uniform_md_spacing_is_rejected():
    md, x, y, z = _straight((0, 1, 0))
    md = md.copy()
    md[200:] += 7.0
    with pytest.raises(SurveyPrimitiveError, match="not uniform"):
        windowed_survey(md, x, y, z)


def test_mismatched_lengths_are_rejected():
    md, x, y, z = _straight((0, 1, 0))
    with pytest.raises(SurveyPrimitiveError, match="same length"):
        windowed_survey(md, x, y, z[:-1])


def test_too_few_rows_for_the_window_is_rejected():
    md, x, y, z = _straight((0, 1, 0), n=DEFAULT_WINDOW_FT)
    with pytest.raises(SurveyPrimitiveError, match="finite rows"):
        windowed_survey(md, x, y, z)


def test_artifact_floor_matches_the_rounding_that_produces_it():
    s = windowed_survey(*_straight((0, 1, 0)))
    expected = np.degrees(COORD_QUANTUM_FT / DEFAULT_WINDOW_FT) * 100.0
    assert np.isclose(s.artifact_floor_deg_per_100ft, expected)
    # the 1 ft case is the 57.3 deg/100ft that invalidated the geometry features
    assert np.isclose(np.degrees(COORD_QUANTUM_FT / 1) * 100.0, 57.2957, atol=1e-3)


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------

def test_unwrap_azimuth_removes_the_wrap_discontinuity():
    wrapped = np.array([358.0, 359.0, 0.0, 1.0, 2.0])
    out = unwrap_azimuth(wrapped)
    assert np.all(np.abs(np.diff(out)) < 2.0)


def test_vertical_section_projects_along_and_ignores_across():
    east = np.array([0.0, 0.0]); north = np.array([0.0, 100.0])
    assert np.isclose(vertical_section(east, north, 0.0)[1], 100.0)
    assert np.isclose(vertical_section(east, north, 90.0)[1], 0.0, atol=1e-9)


@pytest.mark.parametrize("azimuth", [0.0, 35.0, 142.2, 170.0])
def test_derive_section_azimuth_recovers_the_heading(azimuth):
    theta = np.radians(azimuth)
    t = np.arange(N, dtype=float)
    x = t * np.sin(theta)
    y = t * np.cos(theta)
    z = np.zeros_like(t)
    assert np.isclose(derive_section_azimuth(x, y, z), azimuth % 180.0, atol=0.5)


def test_derive_section_azimuth_is_indifferent_to_drilling_direction():
    """Pads drill both ways along one line; the azimuth must not flip."""
    theta = np.radians(140.0)
    t = np.arange(N, dtype=float)
    x, y, z = t * np.sin(theta), t * np.cos(theta), np.zeros_like(t)
    forward = derive_section_azimuth(x, y, z)
    reverse = derive_section_azimuth(x[::-1], y[::-1], z)
    assert np.isclose(forward, reverse, atol=1e-6)


def test_derive_section_azimuth_needs_enough_points():
    with pytest.raises(SurveyPrimitiveError, match="at least 50"):
        derive_section_azimuth(np.zeros(10), np.zeros(10), np.zeros(10))


def test_azimuth_trust_is_zero_north_south_and_one_east_west():
    inc = np.array([90.0, 90.0, 90.0, 90.0])
    azi = np.array([0.0, 90.0, 180.0, 270.0])
    w = azimuth_error_multiplier(azi, inc, mag_to_grid_deg=0.0)
    assert np.allclose(w, [0.0, 1.0, 0.0, 1.0], atol=1e-9)


def test_azimuth_trust_applies_the_magnetic_correction():
    """N/S immunity is relative to MAGNETIC north, not grid north."""
    inc = np.array([90.0])
    w_grid = azimuth_error_multiplier(np.array([8.7414]), inc, mag_to_grid_deg=8.7414)
    assert np.isclose(w_grid[0], 0.0, atol=1e-9)


# --------------------------------------------------------------------------
# label-free validation against the real corpus
# --------------------------------------------------------------------------

def test_build_section_sweeps_zero_to_ninety_on_real_wells():
    """The build must trace 0->90 with no labels involved.

    If the extractor is wrong, this fails without needing any target value --
    inclination is forced by the physics of landing a horizontal well.
    """
    data_dir = os.environ.get("GEOSTEERN_DATA_DIR")
    if not data_dir:
        pytest.skip("GEOSTEERN_DATA_DIR is unset; no corpus to validate against")
    import pandas as pd

    files = sorted(Path(data_dir).glob("train/*horizontal_well*.csv"))[:20]
    if not files:
        pytest.skip(f"no train wells under {data_dir}")

    swept = 0
    for path in files:
        frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z"]).to_numpy(float)
        frame = frame[np.isfinite(frame).all(axis=1)]
        if len(frame) < 1500:
            continue
        survey = windowed_survey(*frame.T)
        inc = survey.inclination_deg
        if inc.min() < 10.0 and inc.max() > 88.0:
            swept += 1
        # toe-up must survive on real data too
        assert inc.max() < 180.0
    assert swept >= 10, f"only {swept} wells traced a full build sweep"
