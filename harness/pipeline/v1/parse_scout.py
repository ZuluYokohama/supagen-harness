"""Heuristic parse of scout markdown → DRAFT claims (explore only)."""
from __future__ import annotations

import re
from typing import Any


def parse_scout_text(text: str) -> list[dict[str, Any]]:
    """Extract draft claim candidates from scout reply text.

    Looks for:
      - DRAFT-### / DRAFT_### / Item ID: ...
      - lines under a DRAFT claim heading
      - "DRAFT multi-plane claim" blocks
    Always confidence_requested=DRAFT (never auto-OPEN from scout prose).
    """
    claims: list[dict[str, Any]] = []
    if not text or not text.strip():
        return claims

    # Pattern: DRAFT-001 or DRAFT_001 or id: DRAFT-... or ID: 42 under DRAFT section
    for m in re.finditer(
        r"(?im)(?:item\s+id|claim\s+id|id)\s*[:\-]\s*(DRAFT[-_]?\w+|\d+)",
        text,
    ):
        raw_id = m.group(1)
        cid = raw_id.upper().replace("_", "-")
        if cid.isdigit():
            cid = f"DRAFT-{cid}"
        if not cid.startswith("DRAFT"):
            cid = f"DRAFT-{cid}"
        # grab following description line if any
        after = text[m.end() : m.end() + 400]
        desc_m = re.search(
            r"(?im)(?:description|text|claim)\s*[:\-]\s*(.+)",
            after,
        )
        body = (desc_m.group(1).strip() if desc_m else after.strip().splitlines()[0] if after.strip() else "")
        body = body[:500]
        claims.append(_draft(cid, body or f"Scout draft claim {cid}"))

    # Pattern: **DRAFT claim** section content
    sec = re.search(
        r"(?is)(?:DRAFT\s+claim|DRAFT\s+multi-plane\s+claim)[^\n]*\n(.+?)(?=\n\s*(?:Residue|residue|CERTIFY:|\*\(end|\Z))",
        text,
    )
    if sec and not claims:
        block = sec.group(1).strip()
        cid_m = re.search(r"(DRAFT[-_]?\w+)", block, re.I)
        cid = (cid_m.group(1) if cid_m else "DRAFT-SCOUT-001").upper().replace("_", "-")
        # strip list markers
        lines = [ln.strip(" -\t*") for ln in block.splitlines() if ln.strip()]
        body = " ".join(lines)[:500]
        claims.append(_draft(cid, body or "Scout multi-plane draft"))

    # Fallback: any line containing DRAFT and claim-like language
    if not claims:
        for ln in text.splitlines():
            if re.search(r"\bDRAFT\b", ln, re.I) and len(ln) > 20:
                claims.append(_draft("DRAFT-SCOUT-001", ln.strip()[:500]))
                break

    # de-dupe by id
    seen = set()
    out = []
    for c in claims:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
    return out


def _draft(cid: str, text: str) -> dict[str, Any]:
    # multi-plane if text mentions 2+ known plane keywords
    planes = []
    low = text.lower()
    mapping = {
        "tool_magpi": ["magpi", "mag-pi", "tool_magpi"],
        "wits_surface": ["wits", "surface", "rpm"],
        "decoder_rt": ["decoder", "decode", "rt"],
    }
    for pid, keys in mapping.items():
        if any(k in low for k in keys):
            planes.append(pid)
    if len(planes) < 1:
        planes = ["wits_surface"]  # minimal cover — will STOP if incomplete for multi

    tags = ["from_scout", "explore_only"]
    if "decoder" in low or "dead" in low:
        tags.append("allow_dead_plane")

    return {
        "id": cid,
        "text": text,
        "required_planes": planes if len(planes) > 1 else planes,
        "confidence_requested": "DRAFT",
        "relation_summary": (
            " + ".join(planes) if len(planes) > 1 else "scout single-plane draft"
        ),
        "tags": tags,
    }
