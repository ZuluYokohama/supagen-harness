"""Build multi-plane claim bundles from packs + optional scout drafts."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from parse_scout import parse_scout_text

HERE = Path(__file__).resolve().parent
PACKS = HERE / "packs"


def load_pack(name: str) -> dict[str, Any]:
    path = PACKS / f"{name}.json"
    if not path.is_file():
        # allow path-like names
        path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"pack not found: {name} (tried {PACKS / (name + '.json')})")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_scout_drafts(bundle: dict[str, Any], scout_text: str) -> dict[str, Any]:
    """Attach scout-parsed claims as DRAFT only; never upgrade pack HIGH claims."""
    out = deepcopy(bundle)
    drafts = parse_scout_text(scout_text)
    existing = {c.get("id") for c in out.get("claims") or []}
    for d in drafts:
        if d["id"] in existing:
            d["id"] = d["id"] + "-SCOUT"
        # force DRAFT
        d["confidence_requested"] = "DRAFT"
        if "from_scout" not in d.get("tags", []):
            d.setdefault("tags", []).append("from_scout")
        out.setdefault("claims", []).append(d)
    out.setdefault("meta", {})
    out["meta"]["scout_drafts_added"] = len(drafts)
    return out


def bundle_from_pack(
    pack_name: str,
    scout_text: str | None = None,
    bundle_id_suffix: str | None = None,
) -> dict[str, Any]:
    bundle = load_pack(pack_name)
    if bundle_id_suffix:
        bundle["bundle_id"] = f"{bundle.get('bundle_id', pack_name)}_{bundle_id_suffix}"
    if scout_text:
        bundle = merge_scout_drafts(bundle, scout_text)
    return bundle


def write_bundle(bundle: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path
