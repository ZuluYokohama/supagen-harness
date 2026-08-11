"""Windowed coordinate -> directional survey primitives.

The corpus ships X/Y/Z on a 1 ft grid rounded to 0.01 ft. Differencing that at
1 ft yields 0.01/1 rad = 0.573 deg per step = 57.3 deg/100ft of pure rounding
artifact; measured apparent dogleg at 1 ft baseline is 57.15 against that 57.30
prediction, so the per-foot direction signal carries no geometry whatsoever.

Every directional quantity here is therefore computed over a window long enough
that quantisation is negligible. ``MIN_WINDOW_FT`` is enforced, not advisory.

The trajectory is also piecewise minimum-curvature between survey stations at
roughly one stand: arc-fit phase contrast peaks at 95-96 ft (0.65 against a
0.02 no-station null), and within a course the path fits a planar circular arc
to 1.04x the rounding floor. There is no information between stations, so
``DEFAULT_WINDOW_FT`` matches the course rather than undercutting it.

Sign convention: Z is elevation (more negative = deeper), so the downward
component is ``-dz``. Inclination is measured from vertical-down and **may
exceed 90 degrees** -- 60.6% of lateral samples in this corpus do, because the
laterals are drilled toe-up. Folding that with ``abs()`` destroys the steering
state and is the single easiest way to silently break this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

COORD_QUANTUM_FT = 0.01
MIN_WINDOW_FT = 30
DEFAULT_WINDOW_FT = 96

#: Usable window range, from a rolling sweep over 40 wells (lateral only),
#: bounded on both sides rather than chosen:
#:
#:   lower -- per-foot jitter must clear the coordinate-rounding artifact by
#:            2x. At 32 ft the ratio is 0.97, i.e. the jitter *is* the rounding;
#:            at 64 ft it is still only 1.54. 96 ft is the first window to clear.
#:   upper -- keep >=95% of the p5-p95 signal range. 128 ft holds 97.1% of
#:            inclination and 95.7% of azimuth; 160 ft drops azimuth to 92%.
#:
#: The band floor coincides with the measured survey course (95-96 ft from
#: arc-fit phase contrast), which is the expected result: below the course there
#: is nothing real to resolve, above it genuine steering changes get averaged.
#: Maximising signal-to-jitter alone is degenerate -- it is optimised by
#: smoothing the signal away, and picks 512 ft.
OPERATING_BAND_FT = (96, 128)


class SurveyPrimitiveError(ValueError):
    """Raised when inputs cannot support a trustworthy directional estimate."""


@dataclass(frozen=True)
class WindowedSurvey:
    """Directional quantities at window centres, in measured-depth order."""

    md: NDArray[np.float64]
    inclination_deg: NDArray[np.float64]   # 0 = down, 90 = horizontal, >90 = toe-up
    azimuth_deg: NDArray[np.float64]       # compass, from +Y (grid north), clockwise
    dls_deg_per_100ft: NDArray[np.float64]
    window_ft: int

    @property
    def artifact_floor_deg_per_100ft(self) -> float:
        """Dogleg that pure coordinate rounding alone would manufacture."""
        return float(np.degrees(COORD_QUANTUM_FT / self.window_ft) * 100.0)


def windowed_survey(
    md: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    window_ft: int = DEFAULT_WINDOW_FT,
) -> WindowedSurvey:
    """Inclination, azimuth and dogleg from coordinates over a fixed window.

    ``window_ft`` is a measured-depth baseline in feet. The corpus is on a 1 ft
    grid, so it is also an index stride; a non-uniform grid is rejected rather
    than silently treated as uniform.
    """
    md, x, y, z = (np.asarray(a, dtype=float) for a in (md, x, y, z))
    if not (len(md) == len(x) == len(y) == len(z)):
        raise SurveyPrimitiveError("md, x, y and z must be the same length")
    if window_ft < MIN_WINDOW_FT:
        raise SurveyPrimitiveError(
            f"window {window_ft} ft is below MIN_WINDOW_FT={MIN_WINDOW_FT}: "
            f"rounding alone would fabricate "
            f"{np.degrees(COORD_QUANTUM_FT / max(window_ft, 1e-9)) * 100:.1f} deg/100ft"
        )
    finite = np.isfinite(md) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    md, x, y, z = md[finite], x[finite], y[finite], z[finite]
    if len(md) <= window_ft + 1:
        raise SurveyPrimitiveError(
            f"need more than window_ft+1={window_ft + 1} finite rows, got {len(md)}"
        )
    spacing = np.diff(md)
    if not np.allclose(spacing, spacing[0], atol=1e-6):
        raise SurveyPrimitiveError("md spacing is not uniform")
    sample_ft = float(spacing[0])
    if sample_ft <= 0:
        raise SurveyPrimitiveError(f"md must increase; spacing is {sample_ft}")

    # window_ft is a physical MD baseline, not an index count. Deriving the
    # stride keeps the reported window (and its artifact floor) honest on any
    # grid: on a 2 ft grid a stride of 96 would span 192 ft while still
    # claiming 96.
    stride_exact = window_ft / sample_ft
    stride = int(round(stride_exact))
    if abs(stride_exact - stride) > 1e-6 or stride < 1:
        raise SurveyPrimitiveError(
            f"window {window_ft} ft is not a whole multiple of the "
            f"{sample_ft} ft md spacing"
        )
    if len(md) <= stride + 1:
        raise SurveyPrimitiveError(
            f"need more than {stride + 1} rows for a {window_ft} ft window "
            f"on {sample_ft} ft spacing, got {len(md)}"
        )

    points = np.column_stack((x, y, z))
    step = points[stride:] - points[:-stride]
    length = np.linalg.norm(step, axis=1)
    usable = length > COORD_QUANTUM_FT * 10
    if not usable.any():
        raise SurveyPrimitiveError("no window advances far enough to have a direction")

    step, length = step[usable], length[usable]
    centre = 0.5 * (md[:-stride][usable] + md[stride:][usable])

    # Down is -z. Do NOT take abs(): >90 deg is toe-up and must survive.
    inclination = np.degrees(np.arccos(np.clip(-step[:, 2] / length, -1.0, 1.0)))
    # Compass azimuth: 0 = +Y (grid north), increasing clockwise toward +X (east).
    azimuth = np.degrees(np.arctan2(step[:, 0], step[:, 1])) % 360.0

    unit = step / length[:, None]
    dls = np.zeros(len(unit), dtype=float)
    if len(unit) > 1:
        turn = np.degrees(
            np.arccos(np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0))
        )
        # Consecutive windows are offset by one MD step, not by window_ft: each
        # chord direction is the tangent at its own midpoint, so the turn between
        # neighbours spans the *centre* advance. Dividing by window_ft instead
        # understates dogleg by exactly that factor.
        advance = np.diff(centre)
        dls[1:] = np.where(advance > 0, turn * 100.0 / np.maximum(advance, 1e-9), 0.0)

    return WindowedSurvey(centre, inclination, azimuth, dls, window_ft)


def unwrap_azimuth(azimuth_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Continuous azimuth, free of the 360->0 discontinuity."""
    return np.degrees(np.unwrap(np.radians(np.asarray(azimuth_deg, dtype=float))))


def derive_section_azimuth(
    md: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    survey: WindowedSurvey | None = None,
    landed_threshold_deg: float = 85.0,
) -> float:
    """This well's own vertical-section azimuth, from its coordinates alone.

    A wall plot projects onto the *plan's* section azimuth, but the corpus has
    no plans and every well is drilled on a slightly different heading, so a
    shared azimuth would mis-project every well but one. The lateral's own
    principal horizontal direction is the recoverable equivalent.

    Uses the principal axis of the landed portion rather than a heel-to-toe
    chord: the chord is thrown off by any dogleg near either end, and the
    principal axis is unaffected by which way the well was drilled along the
    line -- pads drill both directions on one azimuth.
    """
    md, x, y, z = (np.asarray(a, dtype=float) for a in (md, x, y, z))
    if not (len(md) == len(x) == len(y) == len(z)):
        raise SurveyPrimitiveError("md, x, y and z must be the same length")
    finite = np.isfinite(md) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    md, x, y = md[finite], x[finite], y[finite]
    if len(x) < 50:
        raise SurveyPrimitiveError("need at least 50 finite points for an azimuth")

    if survey is not None:
        # Survey values sit at window centres, so array position does not
        # correspond to coordinate row -- inferring it from lengths shifts the
        # selection by half a window. Interpolate onto md instead.
        inc_at_md = np.interp(md, survey.md, survey.inclination_deg)
        landed = inc_at_md > landed_threshold_deg
        if landed.sum() >= 50:
            x, y = x[landed], y[landed]

    pts = np.column_stack((x, y))
    pts = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    east, north = vt[0]
    return float(np.degrees(np.arctan2(east, north)) % 180.0)


def vertical_section(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    section_azimuth_deg: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Distance along the vertical-section plane, as a wall plot reports it."""
    theta = np.radians(section_azimuth_deg)
    east = np.asarray(x, dtype=float) - origin[0]
    north = np.asarray(y, dtype=float) - origin[1]
    return north * np.cos(theta) + east * np.sin(theta)


def azimuth_error_multiplier(
    azimuth_deg: NDArray[np.float64],
    inclination_deg: NDArray[np.float64],
    mag_to_grid_deg: float,
) -> NDArray[np.float64]:
    """Relative sensitivity of azimuth to axial magnetic interference.

    Axial interference enters azimuth as ``dBz*sin(I)*sin(A)/(B*cos(dip))``, so
    the geometry-dependent part is ``sin(I)*sin(A_magnetic)``.

    Returns 0..1 where **higher means worse**: 0 drilling due north or south
    (immune), 1 drilling due east or west (maximally sensitive). This is an
    error *sensitivity*, not a trust or confidence weight -- a consumer that
    treats it as trust weights every heading backwards. To weight by
    reliability, use ``1 - multiplier`` or restrict to low values.
    """
    magnetic = (np.asarray(azimuth_deg, dtype=float) - mag_to_grid_deg) % 360.0
    return np.abs(
        np.sin(np.radians(magnetic))
        * np.sin(np.radians(np.asarray(inclination_deg, dtype=float)))
    )
