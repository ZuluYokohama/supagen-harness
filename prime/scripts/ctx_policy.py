"""
Load-time context policy for this kit (16GB Snapdragon / LMS).

WHY models were loading at 4096/8192
-----------------------------------
1. lm_studio_client / lfm_ops / MCP tools defaulted to context_length=4096.
2. lms_home.derived_policy() read LMS settings.defaultContextLength (UI default)
   and l4_ops_pass *replaced* PRIME_LFM_CTX=128000 with that tiny UI default.
3. Holonomy / gemma benches intentionally used 4096 for RAM survival — that
   leaked into "daily fiber" paths.

Policy (purpose-aware, RAM-aware, model-max-aware)
--------------------------------------------------
  embed nomic     → 2048 (model max)
  embed jina      → 8192 (model max) — enough for retrieve chunks
  micro ≤300M     → up to 32k
  small 1–3B      → 32k default; 128k if free≥5GB and model allows (LFM)
  mid   ~7B       → 16k default; 8k if free<3GB
  large ≥12B      → 4k–8k only on this box
  floor / identity rewrites → 8k minimum (short prompts, but don't starve)

Is 8192 enough?
  Short scout JSON: usually yes.
  dual_enter + KB chunks + multi-role: often NO — prefer 32k for 1–3B fibers.
  Holonomy transform ladder: 8–16k fine if packs are capped.
  LFM long-context measurement: use 128k alone, not co-resident with 7B+.

Never treat LMS chat UI default as the production load target.
"""
from __future__ import annotations

import os
import re
from typing import Any


# Hard floors / ceilings for this machine class
CTX_EMBED_NOMIC = 2048
CTX_EMBED_JINA = 8192
CTX_FLOOR_MIN = 8192          # identity / holonomy floor prompts
CTX_DAILY_SMALL = 32768       # 1–3B daily fiber
CTX_LFM_MAX = 128000
CTX_MID = 16384               # ~7B
CTX_LARGE = 8192              # 12B+ on 16GB
CTX_MICRO = 32768


def _free_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def _params_billions(model: str, params_string: str | None = None) -> float | None:
    s = (params_string or "").strip().upper()
    m = re.match(r"([\d.]+)\s*B", s)
    if m:
        return float(m.group(1))
    m = re.match(r"([\d.]+)\s*M", s)
    if m:
        return float(m.group(1)) / 1000.0
    key = (model or "").lower()
    if "230m" in key or "212m" in key or "423m" in key:
        return 0.23
    if "1.2b" in key or "lfm2.5-1.2b" in key:
        return 1.2
    if "ministral-3" in key or "3b" in key:
        return 3.0
    if "frankenstein" in key or "7.2" in key:
        return 7.2
    if "granite" in key and "tiny" in key:
        return 7.0
    if "12b" in key or "gemma-4-12b" in key:
        return 12.0
    if "27b" in key or "bonsai" in key:
        return 27.0
    if "queen" in key or "8b-a1b" in key or "a1b" in key:
        return 8.0
    if "nomic" in key:
        return 0.1
    if "jina" in key:
        return 0.212
    return None


def catalog_max_for(model: str, base: str = "http://127.0.0.1:1234") -> int | None:
    try:
        from lms_layers import l1_catalog

        cat = l1_catalog(base=base)
        for m in cat.get("models") or []:
            if m.get("key") == model:
                mx = m.get("max_context_length")
                return int(mx) if mx else None
    except Exception:
        return None
    return None


def loaded_context_for(model: str, base: str = "http://127.0.0.1:1234") -> int | None:
    try:
        from lms_layers import l1_catalog

        cat = l1_catalog(base=base)
        for m in cat.get("models") or []:
            if m.get("key") != model:
                continue
            for inst in m.get("loaded_instances") or []:
                cfg = inst.get("config") or {}
                if cfg.get("context_length"):
                    return int(cfg["context_length"])
    except Exception:
        return None
    return None


def resolve_load_context(
    model: str,
    requested: int | None = None,
    *,
    purpose: str = "chat",
    catalog_max: int | None = None,
    params_string: str | None = None,
    free_gb: float | None = None,
    base: str = "http://127.0.0.1:1234",
) -> dict[str, Any]:
    """
    Return target context_length for a load/ensure.

    purpose:
      embed | chat | floor | holonomy | deep | max
    """
    free = free_gb if free_gb is not None else _free_gb()
    mx = catalog_max if catalog_max is not None else catalog_max_for(model, base=base)
    key = (model or "").lower()
    pb = _params_billions(model, params_string)
    purpose = (purpose or "chat").lower()

    # --- embed models ---
    if purpose == "embed" or "nomic" in key or (
        "jina" in key and "embed" in key
    ) or key.endswith("retrieval"):
        if "nomic" in key:
            target = CTX_EMBED_NOMIC
        else:
            target = min(CTX_EMBED_JINA, mx or CTX_EMBED_JINA)
        if mx:
            target = min(target, mx)
        return _pack(target, mx, free, purpose, "embed_fixed", requested)

    # --- start from purpose defaults ---
    if purpose == "floor":
        target = CTX_FLOOR_MIN
    elif purpose == "holonomy":
        target = CTX_MID
    elif purpose == "max":
        target = mx or CTX_LFM_MAX
    elif purpose == "deep":
        target = CTX_MID
    else:
        # chat / daily
        if pb is not None and pb <= 0.3:
            target = CTX_MICRO
        elif pb is not None and pb <= 1.5:
            # LFM class — allow long ctx when RAM allows
            if free is not None and free >= 5.0:
                target = min(CTX_LFM_MAX, mx or CTX_LFM_MAX)
            else:
                target = CTX_DAILY_SMALL
        elif pb is not None and pb <= 4.0:
            # Ministral 3B class — 32k is the daily floor on this kit
            if free is not None and free < 2.5:
                target = CTX_FLOOR_MIN
            else:
                target = CTX_DAILY_SMALL
        elif pb is not None and pb <= 9.0:
            if free is not None and free < 3.0:
                target = CTX_FLOOR_MIN
            else:
                target = CTX_MID
        else:
            # 12B+ 
            target = CTX_LARGE if (free is None or free >= 1.5) else 4096

    # explicit request wins but still capped
    if requested is not None and requested > 0:
        # if caller asked for tiny UI default (≤8192) on a small model for chat,
        # promote to policy minimum so we don't silently starve
        if purpose in ("chat", "max") and requested <= 8192 and (pb is None or pb <= 4.0):
            target = max(requested, CTX_DAILY_SMALL if (pb is None or pb <= 4.0) else requested)
        else:
            target = requested

    # env override
    env_ctx = os.environ.get("PRIME_LOAD_CTX")
    if env_ctx and env_ctx.isdigit() and purpose in ("chat", "max"):
        target = int(env_ctx)

    if mx:
        target = min(int(target), int(mx))
    target = max(512, int(target))

    note = "policy"
    if free is not None and free < 2.0 and target > CTX_FLOOR_MIN:
        target = min(target, CTX_FLOOR_MIN)
        note = "ram_clamp_2gb"
    if free is not None and free < 1.2 and target > 4096:
        target = 4096
        note = "ram_clamp_1.2gb"

    return _pack(target, mx, free, purpose, note, requested)


def _pack(target, mx, free, purpose, note, requested) -> dict[str, Any]:
    return {
        "ok": True,
        "context_length": int(target),
        "catalog_max": int(mx) if mx else None,
        "free_gb": round(free, 2) if free is not None else None,
        "purpose": purpose,
        "note": note,
        "requested": requested,
        "enough_for_daily_pack": int(target) >= 16384,
        "reading": (
            f"ctx={target}"
            + (f" (max {mx})" if mx else "")
            + (f", free≈{free:.1f}GB" if free is not None else "")
            + f", purpose={purpose}"
            + (
                " — 8192 is tight for multi-role dual_enter; prefer ≥32k on 1–3B"
                if int(target) <= 8192 and purpose == "chat"
                else ""
            )
        ),
    }


def should_reload_for_ctx(model: str, desired: int, base: str = "http://127.0.0.1:1234") -> bool:
    """True if model is loaded with substantially smaller context than desired."""
    cur = loaded_context_for(model, base=base)
    if cur is None:
        return False
    # reload if less than 75% of desired and gap > 4k
    return cur < desired * 0.75 and (desired - cur) >= 4096
