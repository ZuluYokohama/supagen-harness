"""
Aggressive residency: free RAM for policy ctx, promote under-loaded fibers.

Called before dual_enter / ensure so we don't sit forever at UI-ish 4k/8k
while a 7B+ ghost holds the box.
"""
from __future__ import annotations

from typing import Any


# Models that must never be co-resident with daily fiber on 16GB
HEAVY_KEYS = (
    "gemma-4-12b",
    "frankenstein",
    "bonsai",
    "granite-4-h-tiny",
    "queen-opus",
    "ibm/granite",
    "prism-ml/bonsai",
)


def unload_heavies(
    *,
    keep: set[str] | None = None,
    base: str = "http://127.0.0.1:1234",
) -> dict[str, Any]:
    from lms_layers import l0_post, l1_catalog, l1_free_ram_gb

    keep = keep or set()
    cat = l1_catalog(base=base)
    acts: list[dict[str, Any]] = []
    free0 = l1_free_ram_gb()
    for m in cat.get("models") or []:
        key = m.get("key") or ""
        if key in keep:
            continue
        if m.get("type") == "embedding":
            continue  # keep nomic
        heavy = any(h in key.lower() for h in HEAVY_KEYS)
        # also unload duplicate non-target chat models if free is low
        free = l1_free_ram_gb()
        low = free is not None and free < 4.0
        if not heavy and not low:
            continue
        if not heavy and key in keep:
            continue
        for inst in m.get("loaded_instances") or []:
            iid = inst.get("id")
            if not iid:
                continue
            # never unload the keep target
            if key in keep:
                continue
            if heavy or (low and key not in keep):
                r = l0_post(
                    "/api/v1/models/unload",
                    {"instance_id": iid},
                    base=base,
                    timeout=180,
                )
                acts.append({"unload": iid, "key": key, "ok": r.ok, "err": r.error})
    return {
        "ok": True,
        "actions": acts,
        "free_gb_before": free0,
        "free_gb_after": l1_free_ram_gb(),
        "n_unloaded": sum(1 for a in acts if a.get("ok")),
    }


def pick_chat_model(
    preferred: str | None = None,
    *,
    base: str = "http://127.0.0.1:1234",
) -> dict[str, Any]:
    """
    Prefer: explicit preferred → already-loaded non-heavy LLM with largest ctx
    → DEFAULT_LFM → first small catalog model (ministral/lfm).
    Never picks embedders or jina-as-llm.
    """
    from lms_layers import DEFAULT_LFM, l1_catalog

    cat = l1_catalog(base=base)
    models = cat.get("models") or []
    if preferred:
        for m in models:
            if m.get("key") == preferred:
                return {"key": preferred, "reason": "preferred", "loaded": bool(m.get("loaded"))}

    loaded: list[tuple[int, str]] = []
    for m in models:
        key = m.get("key") or ""
        if m.get("type") == "embedding":
            continue
        if "jina" in key.lower() or "embed" in key.lower():
            continue
        if any(h in key.lower() for h in HEAVY_KEYS):
            continue
        if not m.get("loaded"):
            continue
        ctx = 0
        for inst in m.get("loaded_instances") or []:
            cfg = inst.get("config") or {}
            ctx = max(ctx, int(cfg.get("context_length") or 0))
        loaded.append((ctx, key))
    if loaded:
        loaded.sort(reverse=True)
        return {"key": loaded[0][1], "reason": "already_loaded_max_ctx", "loaded_ctx": loaded[0][0]}

    # prefer LFM if in catalog, else ministral, else DEFAULT
    keys = [m.get("key") for m in models]
    for cand in (DEFAULT_LFM, "liquid/lfm2.5-1.2b", "mistralai/ministral-3-3b"):
        if cand in keys:
            return {"key": cand, "reason": "catalog_prefer", "loaded": False}
    return {"key": DEFAULT_LFM, "reason": "default", "loaded": False}


def promote_chat_fiber(
    model: str,
    *,
    purpose: str = "chat",
    base: str = "http://127.0.0.1:1234",
) -> dict[str, Any]:
    """
    Unload heavies → resolve policy ctx → ensure_loaded.
    Never *downgrade* context (128k stays if already higher than policy floor).
    Only reload when starved (cur << desired).
    """
    from ctx_policy import loaded_context_for, resolve_load_context, should_reload_for_ctx
    from lm_studio_client import LMStudio
    from lms_layers import l1_free_ram_gb

    freed = unload_heavies(keep={model}, base=base)
    pol = resolve_load_context(model, purpose=purpose, free_gb=l1_free_ram_gb(), base=base)
    desired = int(pol["context_length"])
    lm = LMStudio(base)
    cur = loaded_context_for(model, base=base)

    # Keep higher-than-policy ctx (e.g. LFM@128k when policy floor is 32k)
    if cur is not None and cur >= desired:
        return {
            "ok": True,
            "model": model,
            "desired_ctx": desired,
            "prior_ctx": cur,
            "loaded_ctx": cur,
            "free_gb": l1_free_ram_gb(),
            "freed": freed,
            "ensure": {"ok": True, "action": "keep_high_ctx", "context_length": cur},
            "ctx_policy": pol,
            "note": "no_downgrade",
        }

    if cur is not None and should_reload_for_ctx(model, desired, base=base):
        cat = lm.list_models_native()
        for m in cat.get("models") or []:
            if m.get("key") == model:
                for inst in m.get("loaded_instances") or []:
                    lm.unload(inst.get("id") or model)
        ens = lm.load(model, context_length=desired, purpose=purpose)
    else:
        ens = lm.ensure_loaded(model, context_length=desired, purpose=purpose)
    return {
        "ok": bool(ens.get("ok")),
        "model": model,
        "desired_ctx": desired,
        "prior_ctx": cur,
        "loaded_ctx": ens.get("context_length") or loaded_context_for(model, base=base),
        "free_gb": l1_free_ram_gb(),
        "freed": freed,
        "ensure": ens,
        "ctx_policy": pol,
    }


def seamless_substrate(
    *,
    chat_model: str | None = None,
    base: str = "http://127.0.0.1:1234",
) -> dict[str, Any]:
    """One shot: jina ensure + pick/promote chat fiber + nomic embed fallback."""
    out: dict[str, Any] = {"ok": True, "jina": None, "fiber": None, "errors": [], "pick": None}
    try:
        from jina_service import ensure_jina

        j = ensure_jina()
        out["jina"] = {
            "ok": j.get("ok"),
            "status": j.get("status"),
            "base": j.get("base"),
            "dim": j.get("dim"),
            "started": j.get("started"),
        }
        if not j.get("ok"):
            out["errors"].append(f"jina:{j.get('error') or j.get('status')}")
    except Exception as e:
        out["jina"] = {"ok": False, "error": str(e)}
        out["errors"].append(f"jina:{e}")

    pick = pick_chat_model(chat_model, base=base)
    out["pick"] = pick
    model = pick["key"]
    try:
        out["fiber"] = promote_chat_fiber(model, base=base)
        if not out["fiber"].get("ok"):
            out["errors"].append(f"fiber:{out['fiber'].get('ensure', {}).get('error')}")
    except Exception as e:
        out["fiber"] = {"ok": False, "error": str(e)}
        out["errors"].append(f"fiber:{e}")

    try:
        from holonomy_capacity_bench import ensure_embed

        ensure_embed()
        out["nomic"] = {"ok": True}
    except Exception as e:
        out["nomic"] = {"ok": False, "error": str(e)}

    out["ok"] = bool((out.get("jina") or {}).get("ok")) and bool(
        (out.get("fiber") or {}).get("ok")
    )
    return out
