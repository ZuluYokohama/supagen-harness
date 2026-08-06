"""
LFM residency + hyperparameter profile — max useful context + load/infer knobs.

Kit: liquid/lfm2.5-1.2b, max_context_length=128000, ~1.25GB weights.
Load: context_length, flash_attention, eval_batch_size, offload_kv (server may set).
Infer (per role): temperature / top_p from hub model.yaml; short NLI uses smaller
effective context via short prompts (not a second instance).

Never leave Granite/Bonsai resident when LFM is the measurement substrate.
"""
from __future__ import annotations

import json
from typing import Any

from lms_layers import (
    DEFAULT_BASE,
    DEFAULT_EMBED,
    DEFAULT_LFM,
    l0_post,
    l1_catalog,
    l1_free_ram_gb,
)

# Max context this model exposes on LMS (verified live)
LFM_MAX_CTX = 128_000
# Hub sampling defaults (model.yaml)
LFM_TEMP = 0.1
LFM_TOP_P = 0.1
LFM_TOP_K = 50
LFM_REPEAT = 1.05

# Load plane
LOAD_FLASH = True
LOAD_EVAL_BATCH = 512  # smaller batch reduces peak RAM vs 2048 on Snapdragon


def ensure_lfm_max(
    *,
    base: str = DEFAULT_BASE,
    context_length: int = LFM_MAX_CTX,
    unload_others: bool = True,
    keep_embed: bool = True,
) -> dict[str, Any]:
    """
    Unload non-LFM LLMs; load one LFM instance at max context + flash.
    Returns residency report.
    """
    actions: list[dict[str, Any]] = []
    cat = l1_catalog(base=base)
    free0 = l1_free_ram_gb()

    for m in cat.get("models") or []:
        key = m.get("key") or ""
        if m.get("type") == "embedding" and keep_embed:
            continue
        if key == DEFAULT_LFM or key.startswith(DEFAULT_LFM):
            # unload duplicate LFM instances only
            insts = m.get("loaded_instances") or []
            for inst in insts[1:]:
                r = l0_post(
                    "/api/v1/models/unload",
                    {"instance_id": inst["id"]},
                    base=base,
                    timeout=120,
                )
                actions.append({"unload": inst["id"], "ok": r.ok})
            continue
        if unload_others and m.get("type") == "llm":
            for inst in m.get("loaded_instances") or []:
                r = l0_post(
                    "/api/v1/models/unload",
                    {"instance_id": inst["id"]},
                    base=base,
                    timeout=120,
                )
                actions.append({"unload": inst["id"], "ok": r.ok})

    # Already have single LFM at target ctx?
    cat = l1_catalog(base=base)
    for m in cat.get("models") or []:
        if m.get("key") != DEFAULT_LFM:
            continue
        insts = m.get("loaded_instances") or []
        if len(insts) == 1:
            cfg = insts[0].get("config") or {}
            if int(cfg.get("context_length") or 0) >= context_length and cfg.get("flash_attention"):
                return {
                    "ok": True,
                    "action": "already_max",
                    "instance_id": insts[0].get("id"),
                    "config": cfg,
                    "free_gb": l1_free_ram_gb(),
                    "free_gb_before": free0,
                    "actions": actions,
                    "profile": profile_dict(),
                }
        # unload remaining LFM to reload with max params
        for inst in insts:
            r = l0_post(
                "/api/v1/models/unload",
                {"instance_id": inst["id"]},
                base=base,
                timeout=120,
            )
            actions.append({"unload": inst["id"], "ok": r.ok})

    body: dict[str, Any] = {
        "model": DEFAULT_LFM,
        "context_length": int(context_length),
        "flash_attention": LOAD_FLASH,
        "eval_batch_size": LOAD_EVAL_BATCH,
    }
    r = l0_post("/api/v1/models/load", body, base=base, timeout=600)
    if not r.ok and LOAD_FLASH:
        body.pop("flash_attention", None)
        r = l0_post("/api/v1/models/load", body, base=base, timeout=600)
        actions.append({"load_retry_no_flash": True, "ok": r.ok})

    cat = l1_catalog(base=base)
    cfg = {}
    iid = None
    for m in cat.get("models") or []:
        if m.get("key") == DEFAULT_LFM and m.get("loaded_instances"):
            iid = m["loaded_instances"][0].get("id")
            cfg = m["loaded_instances"][0].get("config") or {}
            break

    # ensure embed still up
    if keep_embed:
        emb_ok = any(
            m.get("key") == DEFAULT_EMBED and m.get("loaded")
            for m in (cat.get("models") or [])
        )
        if not emb_ok:
            er = l0_post(
                "/api/v1/models/load",
                {"model": DEFAULT_EMBED, "context_length": 2048},
                base=base,
                timeout=120,
            )
            actions.append({"load_embed": er.ok})

    return {
        "ok": bool(r.ok and iid),
        "action": "loaded",
        "instance_id": iid,
        "load_body": body,
        "config": cfg,
        "load_response": r.data if r.ok else None,
        "error": r.error,
        "free_gb": l1_free_ram_gb(),
        "free_gb_before": free0,
        "actions": actions,
        "profile": profile_dict(),
    }


def profile_dict() -> dict[str, Any]:
    return {
        "model": DEFAULT_LFM,
        "load": {
            "context_length": LFM_MAX_CTX,
            "flash_attention": LOAD_FLASH,
            "eval_batch_size": LOAD_EVAL_BATCH,
            "offload_kv_cache_to_gpu": True,  # LMS may set; no-op if no GPU offload
        },
        "infer_defaults": {
            "temperature": LFM_TEMP,
            "top_p": LFM_TOP_P,
            "top_k": LFM_TOP_K,
            "repeat_penalty": LFM_REPEAT,
        },
        "role_context_budget": {
            "NLI": 2048,
            "rewrite": 4096,
            "deep_pack": LFM_MAX_CTX,
        },
        "note": (
            "Max ctx is capacity. Fill is still gated (packs/roles). "
            "ARM64: flash may help; GPU offload often no-op on this kit."
        ),
    }


def chat_sampling(role: str = "default") -> dict[str, float]:
    """Infer sampling by role abstraction."""
    if role.upper() in ("NLI", "VERDICT", "FALSIFY"):
        return {"temperature": 0.0, "top_p": 0.1}
    if role.upper() in ("SCOUT", "GLUE", "RESTRICT", "REWRITE"):
        return {"temperature": LFM_TEMP, "top_p": LFM_TOP_P}
    return {"temperature": LFM_TEMP, "top_p": LFM_TOP_P}


if __name__ == "__main__":
    print(json.dumps(ensure_lfm_max(), indent=2))
