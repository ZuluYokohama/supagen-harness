# flake8: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.spatial_split import (
    SplitConstraints,
    SplitConstructionError,
    build_pad_out_manifest,
    build_region_out_manifest,
    load_well_geometries,
    verify_manifest_sha256,
    write_manifest,
)


PERMISSIVE = SplitConstraints(
    min_validation_wells=1,
    min_validation_suffix_rows=1,
    min_training_wells=1,
    min_training_suffix_rows=1,
)


def _write_well(
    train_dir: Path,
    well_id: str,
    center_x: float,
    center_y: float,
    *,
    typewell_token: str | None = None,
    x_offsets: np.ndarray | None = None,
    y_offsets: np.ndarray | None = None,
) -> None:
    md = np.arange(0.0, 1_000.0, 100.0)
    if x_offsets is None:
        x_offsets = np.linspace(-200.0, 200.0, len(md))
    if y_offsets is None:
        y_offsets = np.zeros(len(md))
    tvt_input = np.full(len(md), np.nan)
    tvt_input[:3] = [10_000.0, 10_000.5, 10_001.0]
    # Forbidden model columns are intentionally present; the loader must select
    # only MD/X/Y/TVT_input and therefore cannot consume them.
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": center_x + x_offsets,
            "Y": center_y + y_offsets,
            "TVT_input": tvt_input,
            "TVT": np.linspace(1.0, 2.0, len(md)),
            "GR": np.linspace(80.0, 100.0, len(md)),
            "EGFDU": np.linspace(-9_000.0, -9_100.0, len(md)),
        }
    )
    frame.to_csv(train_dir / f"{well_id}__horizontal_well.csv", index=False)
    token = typewell_token if typewell_token is not None else f"unique:{well_id}"
    (train_dir / f"{well_id}__typewell.csv").write_bytes(token.encode("utf-8"))


def _fold_by_id(manifest: dict) -> dict[str, int]:
    return {
        well_id: int(fold["fold"])
        for fold in manifest["folds"]
        for well_id in fold["validation_ids"]
    }


def test_pad_out_is_deterministic_balanced_and_hash_indivisible(tmp_path: Path) -> None:
    for pad in range(10):
        for member in range(2):
            well_id = f"p{pad:02d}_{member}"
            token = "shared-typewell" if (pad, member) in {(0, 0), (1, 0)} else None
            _write_well(
                tmp_path,
                well_id,
                center_x=pad * 5_000.0 + member * 100.0,
                center_y=0.0,
                typewell_token=token,
            )
    wells = load_well_geometries(tmp_path, excluded_ids=())
    first = build_pad_out_manifest(wells, excluded_ids=(), constraints=PERMISSIVE)
    second = build_pad_out_manifest(
        list(reversed(wells)), excluded_ids=(), constraints=PERMISSIVE
    )
    assert first == second
    assert verify_manifest_sha256(first)
    assert first["resampled_polyline_spacing_ft"] == 100.0
    assert len(first["construction_provenance"]["source_sha256"]) == 64
    assert set(first["construction_provenance"]["packages"]) == {
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    }
    folds = _fold_by_id(first)
    assert folds["p00_0"] == folds["p01_0"]
    row_counts = [
        fold["diagnostics"]["validation_suffix_rows"] for fold in first["folds"]
    ]
    assert max(row_counts) - min(row_counts) <= 14
    for fold in first["folds"]:
        assert not set(fold["validation_ids"]) & set(fold["training_ids"])
        assert not fold["embargo_ids"]


def test_region_out_buffers_and_preserves_exact_groups(tmp_path: Path) -> None:
    cluster_centers = [
        (0.0, 0.0),
        (40_000.0, 0.0),
        (80_000.0, 0.0),
        (20_000.0, 40_000.0),
        (60_000.0, 40_000.0),
    ]
    for cluster, (cx, cy) in enumerate(cluster_centers):
        for member in range(4):
            token = "shared-local" if cluster == 0 and member < 2 else None
            _write_well(
                tmp_path,
                f"r{cluster}_{member}",
                center_x=cx + member * 300.0,
                center_y=cy + member * 150.0,
                typewell_token=token,
            )
    wells = load_well_geometries(tmp_path, excluded_ids=())
    manifest = build_region_out_manifest(
        wells,
        excluded_ids=(),
        centroid_buffer_ft=5_000.0,
        polyline_buffer_ft=1_500.0,
        constraints=PERMISSIVE,
    )
    reversed_manifest = build_region_out_manifest(
        list(reversed(wells)),
        excluded_ids=(),
        centroid_buffer_ft=5_000.0,
        polyline_buffer_ft=1_500.0,
        constraints=PERMISSIVE,
    )
    assert manifest == reversed_manifest
    assert verify_manifest_sha256(manifest)
    folds = _fold_by_id(manifest)
    assert folds["r0_0"] == folds["r0_1"]
    assert len(set(folds.values())) == 5
    universe = set(folds)
    for fold in manifest["folds"]:
        validation = set(fold["validation_ids"])
        training = set(fold["training_ids"])
        embargo = set(fold["embargo_ids"])
        assert validation | training | embargo == universe
        assert not validation & training
        assert not validation & embargo
        assert not training & embargo
        assert fold["diagnostics"]["nearest_training_centroid_ft"]["min"] > 5_000.0
        assert (
            fold["diagnostics"]["nearest_training_resampled_polyline_ft"]["min"]
            > 1_500.0
        )


def test_manifest_round_trip_sha256(tmp_path: Path) -> None:
    for index in range(10):
        _write_well(tmp_path, f"w{index}", index * 4_000.0, 0.0)
    wells = load_well_geometries(tmp_path, excluded_ids=())
    manifest = build_pad_out_manifest(wells, excluded_ids=(), constraints=PERMISSIVE)
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == manifest
    assert verify_manifest_sha256(loaded)
    loaded["parameters"]["centroid_component_radius_ft"] = 123.0
    assert not verify_manifest_sha256(loaded)


def test_conflicting_exact_typewell_group_fails(tmp_path: Path) -> None:
    for index in range(10):
        _write_well(
            tmp_path,
            f"same{index}",
            index * 10_000.0,
            0.0,
            typewell_token="one-typewell-for-every-well",
        )
    wells = load_well_geometries(tmp_path, excluded_ids=())
    with pytest.raises(SplitConstructionError, match="indivisible pad components"):
        build_pad_out_manifest(wells, excluded_ids=(), constraints=PERMISSIVE)


def test_invalid_geometry_parameters_fail_closed(tmp_path: Path) -> None:
    for index in range(10):
        _write_well(tmp_path, f"v{index}", index * 10_000.0, 0.0)
    with pytest.raises(SplitConstructionError, match="resample_spacing"):
        load_well_geometries(tmp_path, excluded_ids=(), resample_spacing=0.0)
    wells = load_well_geometries(tmp_path, excluded_ids=())
    with pytest.raises(SplitConstructionError, match="radius_ft"):
        build_pad_out_manifest(
            wells, excluded_ids=(), radius_ft=np.nan, constraints=PERMISSIVE
        )
    with pytest.raises(SplitConstructionError, match="centroid_buffer_ft"):
        build_region_out_manifest(
            wells,
            excluded_ids=(),
            centroid_buffer_ft=0.0,
            constraints=PERMISSIVE,
        )
    with pytest.raises(SplitConstructionError, match="excluded IDs remain"):
        build_pad_out_manifest(
            wells,
            excluded_ids=(wells[0].well_id,),
            constraints=PERMISSIVE,
        )
