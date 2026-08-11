"""Build a durable fold and source manifest for an interval-gate result."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from geosteern.data import find_typewell, well_id
from research.interval_gate import (
    EXCLUDED_TEST_OVERLAP,
    _filtered_files,
)
from research.ordered_transport import FROZEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(data_dir: Path, result_path: Path) -> tuple[Path, Path]:
    files, dev, hold, typewell_hashes = _filtered_files(str(data_dir))
    split = {well_id(path): "dev" for path in dev}
    split.update({well_id(path): "holdout" for path in hold})
    rows = []
    for number, horizontal in enumerate(files, 1):
        horizontal_path = Path(horizontal)
        typewell_path = Path(find_typewell(horizontal) or "")
        frame = pd.read_csv(horizontal_path, usecols=["TVT_input", "GR"])
        wid = well_id(horizontal)
        rows.append({
            "well": wid,
            "split": split[wid],
            "typewell_profile_hash": typewell_hashes[wid],
            "rows": int(len(frame)),
            "prefix_rows": int(frame["TVT_input"].notna().sum()),
            "suffix_rows": int(frame["TVT_input"].isna().sum()),
            "gr_valid_fraction": float(frame["GR"].notna().mean()),
            "horizontal_sha256": sha256_file(horizontal_path),
            "typewell_sha256": sha256_file(typewell_path),
            "horizontal_file": horizontal_path.name,
            "typewell_file": typewell_path.name,
        })
        if number % 100 == 0:
            print(f"hashed {number}/{len(files)} wells", flush=True)

    manifest = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    fold_path = result_path.with_name(result_path.stem + "_fold_manifest.csv")
    manifest.to_csv(fold_path, index=False)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int((manifest["split"] == "holdout").sum()) != result["holdout_wells"]:
        raise RuntimeError("fold manifest does not match result holdout")
    if int(manifest.loc[manifest["split"] == "holdout", "suffix_rows"].sum()) \
            != result["base_model"]["holdout"]["n_rows"]:
        raise RuntimeError("fold manifest suffix rows do not match scored rows")

    tracked = {
        "research/interval_gate.py": ROOT / "research" / "interval_gate.py",
        "research/ordered_transport.py": ROOT / "research" / "ordered_transport.py",
        "research/test_interval_gate.py": ROOT / "research" / "test_interval_gate.py",
        "research/test_ordered_transport.py": ROOT / "research" / "test_ordered_transport.py",
        result_path.name: result_path,
        fold_path.name: fold_path,
        "competition_pdf": data_dir / "AI_wellbore_geology_prediction_task_en.pdf",
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm"):
        packages[package] = importlib.metadata.version(package)
    provenance = {
        "status": "MEASURE_ONLY_NOT_OPEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_before_untracked_research": git_head,
        "result": str(result_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "excluded_test_overlap_ids": sorted(EXCLUDED_TEST_OVERLAP),
        "eligible_wells": int(len(manifest)),
        "unique_typewell_groups": int(manifest["typewell_profile_hash"].nunique()),
        "dev_wells": int((manifest["split"] == "dev").sum()),
        "holdout_wells": int((manifest["split"] == "holdout").sum()),
        "holdout_suffix_rows": int(
            manifest.loc[manifest["split"] == "holdout", "suffix_rows"].sum()
        ),
        "frozen_ordered_settings": dict(FROZEN_SETTINGS),
        "sha256": {name: sha256_file(path) for name, path in tracked.items()},
        "packages": packages,
        "notes": [
            "Formation surfaces, suffix TVT, Geology, PNGs, and ID lookup are forbidden features.",
            "Ordered diagnostic correction magnitudes in the result are pre-shrink raw solver values.",
            "The artifact is one deterministic outer holdout and is not an OPEN certificate.",
        ],
    }
    provenance_path = result_path.with_name(result_path.stem + "_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return fold_path, provenance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    default_data_dir = os.environ.get("GEOSTEERN_DATA_DIR")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data_dir) if default_data_dir else None,
        required=default_data_dir is None,
        help="directory containing train/ and test/; defaults to $GEOSTEERN_DATA_DIR",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fold, provenance = build(args.data_dir, args.result)
    print(f"wrote {fold}")
    print(f"wrote {provenance}")
