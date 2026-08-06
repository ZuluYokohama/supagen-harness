"""
LM Studio as projection algebra — not a chatbot.

What LMS actually is on this kit (native v1 + OpenAI compat):
  GET  /api/v1/models              catalog + loaded_instances
  POST /api/v1/models/load        resident fiber (model stalk)
  POST /api/v1/models/unload      free RAM
  POST /api/v1/chat               native chat (input/system_prompt; response_id)
  GET  /v1/models                 openai list
  POST /v1/chat/completions       openai chat
  POST /v1/embeddings             geometric projection (nomic 768-d)
  POST /v1/responses              Open Responses / stateful tools surface

Conceptual map (Prime topology):
  user enter     → base event in possibility space
  each LLM       → fiber / dimensional projection of that event
  embedding      → metric on language stalks (glue better than bag-of-words)
  load/unload    → which fibers are *resident* under 16GB RAM
  ensemble       → H0 of multi-fiber outputs (shared claims) vs residual disagreement
  never OPEN     on LMS alone — measure only

Hyper-optimized path: see ``lms_layers`` (L0–L7 gated stack).
This module remains the compatibility façade for existing imports.
"""
from __future__ import annotations

import json
from typing import Any

# Prefer layered stack for all new work
from lms_layers import (  # noqa: F401
    DEFAULT_BASE,
    DEFAULT_EMBED,
    DEFAULT_LFM,
    StateFiber,
    cosine,
    l0_health,
    l1_catalog,
    l1_ensure_substrate,
    l1_plan,
    l2_chat,
    l2_embed,
    l2_responses,
    layered_enter,
    layer_matrix,
)


class LMStudio:
    """Thin façade over L0–L2. Prefer ``lms_layers`` for gated multi-role work."""

    def __init__(self, base: str = DEFAULT_BASE):
        self.base = base.rstrip("/")

    def list_models_native(self) -> dict[str, Any]:
        cat = l1_catalog(base=self.base)
        if not cat.get("ok"):
            return {"ok": False, "error": cat.get("error"), "base": self.base}
        return {"ok": True, "models": cat.get("models") or [], "base": self.base, "duplicates": cat.get("duplicates")}

    def load(
        self,
        model: str,
        context_length: int | None = None,
        purpose: str = "chat",
    ) -> dict[str, Any]:
        from lms_layers import l0_post
        from ctx_policy import resolve_load_context

        pol = resolve_load_context(model, context_length, purpose=purpose, base=self.base)
        ctx = int(pol["context_length"])
        body: dict[str, Any] = {"model": model, "context_length": ctx}
        r = l0_post("/api/v1/models/load", body, base=self.base, timeout=600)
        if not r.ok:
            # retry without ctx, then with half ctx
            r = l0_post("/api/v1/models/load", {"model": model}, base=self.base, timeout=600)
            if r.ok:
                return {
                    "ok": True,
                    "action": "load",
                    **r.data,
                    "note": "loaded without context_length",
                    "ctx_policy": pol,
                }
            half = max(4096, ctx // 2)
            r = l0_post(
                "/api/v1/models/load",
                {"model": model, "context_length": half},
                base=self.base,
                timeout=600,
            )
            if r.ok:
                return {
                    "ok": True,
                    "action": "load",
                    **r.data,
                    "context_length": half,
                    "ctx_policy": {**pol, "note": "half_retry"},
                }
        if r.ok:
            return {
                "ok": True,
                "action": "load",
                **r.data,
                "context_length": ctx,
                "ctx_policy": pol,
            }
        return {"ok": False, "action": "load", "error": r.error, "ctx_policy": pol}

    def unload(self, instance_id: str) -> dict[str, Any]:
        from lms_layers import l0_post

        r = l0_post(
            "/api/v1/models/unload",
            {"instance_id": instance_id},
            base=self.base,
            timeout=120,
        )
        if r.ok:
            return {"ok": True, "action": "unload", **r.data}
        return {"ok": False, "action": "unload", "error": r.error}

    def ensure_loaded(
        self,
        model: str,
        context_length: int | None = None,
        purpose: str = "chat",
    ) -> dict[str, Any]:
        from ctx_policy import resolve_load_context, should_reload_for_ctx
        from lms_layers import l0_post

        pol = resolve_load_context(model, context_length, purpose=purpose, base=self.base)
        ctx = int(pol["context_length"])

        # Hyper path: consolidate substrate, no instance spam
        if model in (DEFAULT_LFM, "liquid/lfm2.5-1.2b") or model == DEFAULT_EMBED:
            return l1_ensure_substrate(
                model=model if model != DEFAULT_EMBED else DEFAULT_LFM,
                embed=DEFAULT_EMBED,
                context_length=ctx,
                base=self.base,
            )
        cat = self.list_models_native()
        for m in cat.get("models") or []:
            if m.get("key") == model and m.get("loaded"):
                if should_reload_for_ctx(model, ctx, base=self.base):
                    for inst in m.get("loaded_instances") or []:
                        l0_post(
                            "/api/v1/models/unload",
                            {"instance_id": inst.get("id")},
                            base=self.base,
                            timeout=120,
                        )
                    return self.load(model, context_length=ctx, purpose=purpose)
                return {
                    "ok": True,
                    "action": "already_loaded",
                    "model": model,
                    "context_length": (m.get("loaded_instances") or [{}])[0]
                    .get("config", {})
                    .get("context_length"),
                    "ctx_policy": pol,
                }
        return self.load(model, context_length=ctx, purpose=purpose)

    def chat_native(
        self,
        prompt: str,
        model: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 256,
        previous_response_id: str | None = None,
        store: bool = True,
        context_length: int | None = None,
        integrations: list[Any] | None = None,
    ) -> dict[str, Any]:
        r = l2_chat(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            store=store,
            context_length=context_length,
            integrations=integrations,
            base=self.base,
        )
        # shape expected by older callers
        return {
            "ok": r.get("ok"),
            "mode": "lm_native_chat",
            "model": model,
            "content": r.get("content") or "",
            "response_id": r.get("response_id"),
            "stats": r.get("stats"),
            "model_instance_id": r.get("model_instance_id"),
            "cost": r.get("cost"),
            "tool_calls": r.get("tool_calls"),
            "error": r.get("error"),
            "gate": r.get("gate"),
        }

    def chat_openai(
        self,
        prompt: str,
        model: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        from lms_layers import l2_chat_openai_fallback

        r = l2_chat_openai_fallback(
            prompt, model=model, system=system, temperature=temperature, max_tokens=max_tokens, base=self.base
        )
        return {
            "ok": r.get("ok"),
            "mode": "lm_openai_chat",
            "model": model,
            "content": r.get("content") or "",
            "usage": r.get("usage"),
            "error": r.get("error"),
        }

    def responses(
        self,
        prompt: str,
        model: str = DEFAULT_LFM,
        instructions: str = "",
        max_tokens: int = 160,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        return l2_responses(
            prompt,
            model=model,
            instructions=instructions,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            base=self.base,
        )

    def embed(
        self,
        text: str,
        model: str | None = None,
        task: str = "search_document",
        dims: int | None = None,
    ) -> dict[str, Any]:
        """Job 1 aboutness — jina default (auto-ensure); nomic fallback. Not agreement."""
        r = l2_embed(text, model=model, base=self.base, task=task, dims=dims)
        return {
            "ok": r.get("ok"),
            "mode": "lm_embed",
            "job": "retrieval_aboutness",
            "not_agreement": True,
            "model": r.get("model") or model,
            "family": r.get("family"),
            "task": r.get("task") or task,
            "dim": r.get("dim"),
            "embedding": r.get("embedding") or [],
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
            "warning": r.get("warning"),
            "jina_service": r.get("jina_service"),
        }


LAW_SYSTEM = (
    "You are a local projection fiber for the Prime stack. "
    "Design law: restrict → measure → audit → OPEN|STOP. Residue never forced. "
    "Language is a projection from both sides. Be terse. Flag uncertainty. "
    "Output: (1) 3 bullets of what the human is asking, (2) one risk, (3) OPEN|STOP|NEED_INFO."
)


def enter_projection(
    prompt: str,
    base: str = DEFAULT_BASE,
    models: list[str] | None = None,
    embed: bool = True,
    max_tokens: int = 180,
    parallel: bool = True,
    mode: str = "lfm_ops",
) -> dict[str, Any]:
    """
    One enter → projection.

    Default mode ``lfm_ops`` / ``layered``: L0–L7 gated LFM ops (JSON roles).
    mode ``multi_model``: legacy multi-LLM fibers if explicitly requested.
    mode ``legacy_roles``: free-text roles without JSON gate (lfm_ops.legacy).
    """
    if mode in ("lfm_ops", "lfm", "orthogonal", "default", "layered", "dual", None, ""):
        # Canonical path: dual_enter (aboutness + NLI + cert face)
        from dual_enter import dual_enter

        r = dual_enter(
            prompt,
            base=base,
            model=(models or [DEFAULT_LFM])[0],
            embed=embed,
            retrieve_kb=True,
        )
        return r

    if mode in ("legacy_roles", "free_text_roles"):
        from lfm_ops import lfm_role_pass

        model = (models or [DEFAULT_LFM])[0]
        r = lfm_role_pass(
            prompt,
            base=base,
            model=model,
            embed=embed,
            max_tokens=max_tokens,
            use_layers=False,
        )
        r["prompt_preview"] = prompt[:240]
        r["enter_mode"] = "legacy_roles"
        return r

    # ---- multi_model path (explicit only) ----
    lm = LMStudio(base)
    cat = lm.list_models_native()
    if not cat.get("ok"):
        return {"ok": False, "error": "cannot list models", "detail": cat}

    llms = [m for m in cat["models"] if m.get("type") == "llm"]
    loaded = [m["key"] for m in llms if m.get("loaded")]
    preferred = models or [DEFAULT_LFM]
    fibers = [m for m in preferred if m in loaded]
    if not fibers:
        fibers = loaded[:1] if loaded else []
    if not fibers:
        for m in preferred:
            ens = lm.ensure_loaded(m, purpose="chat")
            if ens.get("ok") or ens.get("action") == "already_loaded":
                fibers = [m]
                break
    if not fibers:
        return {"ok": False, "error": "no LLM fibers available", "catalog": cat["models"]}

    results: dict[str, Any] = {}
    for m in fibers:
        results[m] = lm.chat_native(
            prompt, model=m, system=LAW_SYSTEM, temperature=0.15, max_tokens=max_tokens
        )

    votes = {"OPEN": 0, "STOP": 0, "NEED_INFO": 0, "OTHER": 0}
    for m, r in results.items():
        text = (r.get("content") or "").upper()
        if "NEED_INFO" in text:
            votes["NEED_INFO"] += 1
        elif "STOP" in text:
            votes["STOP"] += 1
        elif "OPEN" in text:
            votes["OPEN"] += 1
        else:
            votes["OTHER"] += 1

    emb_report: dict[str, Any] = {}
    if embed:
        he = lm.embed(prompt)
        if he.get("ok"):
            sims = {}
            for m, r in results.items():
                if not r.get("ok"):
                    continue
                ee = lm.embed(r.get("content") or "")
                if ee.get("ok"):
                    sims[m] = round(cosine(he["embedding"], ee["embedding"]), 4)
            emb_report["cosine_to_human"] = sims
            if sims:
                emb_report["mean_cosine"] = round(sum(sims.values()) / len(sims), 4)

    dominant = max(votes, key=votes.get)  # type: ignore[arg-type]
    agreement = votes[dominant] / max(sum(votes.values()), 1)
    return {
        "ok": True,
        "mode": "enter_projection_multi",
        "enter_mode": "multi_model",
        "prompt_preview": prompt[:240],
        "fibers": fibers,
        "outputs": {
            m: {
                "ok": r.get("ok"),
                "content": (r.get("content") or "")[:600],
                "stats": r.get("stats"),
                "response_id": r.get("response_id"),
                "error": r.get("error"),
            }
            for m, r in results.items()
        },
        "votes": votes,
        "dominant_vote": dominant,
        "vote_agreement": round(agreement, 4),
        "embeddings": emb_report,
        "not_open_authority": True,
        "note": "Multi-model path is optional. Prefer layered lfm_ops.",
    }


def resource_aware_roster(base: str = DEFAULT_BASE) -> dict[str, Any]:
    """Which fibers to keep resident on ~16GB Snapdragon."""
    lm = LMStudio(base)
    cat = lm.list_models_native()
    loaded = [m for m in cat.get("models") or [] if m.get("loaded")]
    return {
        "ok": True,
        "loaded": loaded,
        "duplicates": cat.get("duplicates"),
        "recommendation": {
            "always_on": [DEFAULT_LFM, DEFAULT_EMBED],
            "swap_in": [],
            "deep_only": list(__import__("lms_layers", fromlist=["DEEP_ONLY"]).DEEP_ONLY),
            "policy": (
                "DEFAULT: one LFM instance + nomic embed only. "
                "Consolidate duplicate instances (L1). "
                "Orthogonal JSON roles replace multi-model fanout. "
                "Unload Granite/Bonsai unless explicitly needed."
            ),
            "layers": "see lms_layers.layer_matrix()",
        },
        "catalog": cat.get("models"),
    }
