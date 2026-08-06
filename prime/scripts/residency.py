"""
Aggressive residency: free RAM for policy ctx, promote under-loaded fibers.

Called before dual_enter / ensure so we don't sit forever at UI-ish 4k/8k
while a 7B+ ghost holds the box.

Fiber modes (PRIME_FIBER_MODE / seamless_substrate fiber_mode)
--------------------------------------------------------------
  scout     daily: small LFM/Ministral; UNLOAD frankenstein + other heavies
  preserve  identity/holonomy: frankenstein ALONE @ policy ctx; unload scouts
"""
from __future__ import annotations

from typing import Any


# Models that must never be co-resident with daily SCOUT fiber on 16GB
HEAVY_KEYS = (
    "gemma-4-12b",
    "frankenstein",
    "bonsai",
    "granite-4-h-tiny",
    "queen-opus",
    "ibm/granite",
    "prism-ml/bonsai",
)

PRESERVE_KEYS = (
    "frankenstein",
    "frankenstein-2.0",
    "thedrummer/frankenstein",
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
    fiber_mode: str = "scout",
) -> dict[str, Any]:
    """
    scout: explicit preferred → already-loaded non-heavy LLM max ctx
           → DEFAULT_LFM → ministral. Never embedders / jina-as-llm / frankenstein.
    preserve: frankenstein (or PRIME_PRESERVE_MODEL) only.
    """
    import os

    from lms_layers import DEFAULT_LFM, l1_catalog

    cat = l1_catalog(base=base)
    models = cat.get("models") or []
    keys = [m.get("key") for m in models]

    if fiber_mode == "preserve":
        # frankenstein alone
        env_p = os.environ.get("PRIME_PRESERVE_MODEL")
        if preferred and any(p in preferred.lower() for p in PRESERVE_KEYS):
            return {"key": preferred, "reason": "preserve_preferred", "loaded": preferred in keys}
        if env_p:
            return {"key": env_p, "reason": "preserve_env", "loaded": env_p in keys}
        for m in models:
            key = m.get("key") or ""
            if any(p in key.lower() for p in PRESERVE_KEYS):
                return {
                    "key": key,
                    "reason": "preserve_catalog",
                    "loaded": bool(m.get("loaded")),
                }
        return {
            "key": preferred or env_p or "frankenstein-2.0-i1",
            "reason": "preserve_fallback_name",
            "loaded": False,
            "warning": "frankenstein key not in catalog — set PRIME_PRESERVE_MODEL",
        }

    if preferred and not any(p in preferred.lower() for p in PRESERVE_KEYS):
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
    fiber_mode: str | None = None,
) -> dict[str, Any]:
    """
    One shot: jina ensure + pick/promote chat fiber by mode.

    scout:    unload heavies (incl. frankenstein), load small fiber
    preserve: unload everything else, load frankenstein alone
    nomic ensure is optional fallback only — Job1 is jina.

    fiber_mode=None → PRIME_FIBER_MODE env → scout
    """
    import os

    mode = (fiber_mode if fiber_mode is not None else None) or os.environ.get(
        "PRIME_FIBER_MODE"
    ) or "scout"
    mode = str(mode).lower().strip()
    if mode not in ("scout", "preserve"):
        mode = "scout"

    out: dict[str, Any] = {
        "ok": True,
        "fiber_mode": mode,
        "jina": None,
        "fiber": None,
        "errors": [],
        "pick": None,
    }
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

    pick = pick_chat_model(chat_model, base=base, fiber_mode=mode)
    out["pick"] = pick
    model = pick["key"]

    # Mode-specific unload before promote
    try:
        if mode == "preserve":
            # keep only frankenstein; unload scouts + other heavies
            freed = unload_heavies(keep={model}, base=base)
            # also unload non-heavy chat that is not the preserve model
            from lms_layers import l0_post, l1_catalog

            extra_acts: list[dict[str, Any]] = []
            for m in (l1_catalog(base=base).get("models") or []):
                key = m.get("key") or ""
                if key == model or m.get("type") == "embedding":
                    continue
                if "jina" in key.lower():
                    continue
                for inst in m.get("loaded_instances") or []:
                    iid = inst.get("id")
                    if iid:
                        r = l0_post(
                            "/api/v1/models/unload",
                            {"instance_id": iid},
                            base=base,
                            timeout=180,
                        )
                        extra_acts.append(
                            {
                                "unload": iid,
                                "key": key,
                                "ok": bool(getattr(r, "ok", r) if not isinstance(r, dict) else r.get("ok")),
                                "err": (
                                    getattr(r, "error", "")
                                    if not isinstance(r, dict)
                                    else (r.get("error") or "")
                                ),
                            }
                        )
            freed = dict(freed or {})
            acts = list(freed.get("actions") or []) + extra_acts
            freed["actions"] = acts
            freed["n_unloaded"] = sum(1 for a in acts if a.get("ok"))
            out["preserve_freed"] = freed
            if any(not a.get("ok") for a in extra_acts):
                out["errors"].append("preserve_unload_partial")
        else:
            # scout: frankenstein is HEAVY — unload_heavies inside promote
            pass
    except Exception as e:
        out["errors"].append(f"mode_unload:{e}")

    try:
        purpose = "chat" if mode == "scout" else "preserve"
        out["fiber"] = promote_chat_fiber(model, purpose=purpose, base=base)
        if not out["fiber"].get("ok"):
            out["errors"].append(f"fiber:{out['fiber'].get('ensure', {}).get('error')}")
    except Exception as e:
        out["fiber"] = {"ok": False, "error": str(e)}
        out["errors"].append(f"fiber:{e}")

    # nomic only as degraded aboutness fallback — do not thrash if jina ok
    if not (out.get("jina") or {}).get("ok"):
        try:
            from holonomy_capacity_bench import ensure_embed

            ensure_embed()
            out["nomic"] = {"ok": True, "role": "fallback_only"}
        except Exception as e:
            out["nomic"] = {"ok": False, "error": str(e)}
    else:
        out["nomic"] = {"ok": True, "skipped": True, "reason": "jina_primary"}

    out["ok"] = bool((out.get("jina") or {}).get("ok")) and bool(
        (out.get("fiber") or {}).get("ok")
    )
    out["frankenstein_note"] = (
        "PRESERVE: frankenstein required alone"
        if mode == "preserve"
        else "SCOUT: frankenstein must NOT be loaded (HEAVY); instruments are off-LMS"
    )
    return out
