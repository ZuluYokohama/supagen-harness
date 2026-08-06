"""
LMS layered abstraction — hyper-optimized gates for maximal capability.

LM Studio is not a chatbot. It is a local inference + residency controller.
Prime wraps it as a **gated stack** so every activity has a layer, a cost,
and a stop condition. OPEN authority never lives in LMS.

Layers (bottom → top):

  L0 TRANSPORT     HTTP, auth, timeouts, error taxonomy
  L1 RESIDENCY     catalog, load/unload, instance select, ctx budget, RAM gate
  L2 INFERENCE     chat_native, responses, embeddings, openai fallback
  L3 STATE FIBER   response_id chain, role memory, store discipline
  L4 STRUCTURE OPS JSON-gated SCOUT/FALSIFY/GLUE/VERDICT (schema validators)
  L5 METRIC        embed cosine, glue scores, tok/s + TTFT as cost measures
  L6 POLICY GATE   design law: never OPEN from LMS alone; FATAL→STOP; NEED_INFO
  L7 ORCHESTRATION enter_projection / deep_loop / dimensional handoff

Design law: restrict → measure → audit → OPEN|STOP. Residue never forced.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Constants / kit defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE = os.environ.get("LM_STUDIO_BASE", "http://127.0.0.1:1234").rstrip("/")
DEFAULT_LFM = os.environ.get("PRIME_LFM", "liquid/lfm2.5-1.2b")
DEFAULT_EMBED = os.environ.get("PRIME_EMBED", "text-embedding-nomic-embed-text-v1.5")
# Prefer max LFM context when env unset (kit exposes 128k); packs still gate fill
DEFAULT_CTX = int(os.environ.get("PRIME_LFM_CTX", "128000"))
DEFAULT_API_TOKEN = os.environ.get("LM_API_TOKEN", "")


def _home_policy() -> dict[str, Any]:
    """Lazy local ~/.lmstudio policy (ctx, sampling, chain caps, CPU-only flag)."""
    try:
        from lms_home import derived_policy

        return derived_policy()
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "default_context_length": DEFAULT_CTX,
            "pack_budget_chars": 3500,
            "chain_max_input_chars": 3000,
            "disable_chain_on_large_input": True,
            "cpu_only_engine": True,
            "sampling_lfm": {"temperature": 0.1, "top_p": 0.1},
        }

# Snapdragon-class headroom: never thrash OS + Grok + LMS
ALWAYS_ON = (DEFAULT_LFM, DEFAULT_EMBED)
DEEP_ONLY = ("prism-ml/bonsai-27b", "ibm/granite-4-h-tiny")
# Prefer single instance per key; unload extras
MAX_INSTANCES_PER_KEY = 1


# ---------------------------------------------------------------------------
# L0 — TRANSPORT
# ---------------------------------------------------------------------------

@dataclass
class TransportResult:
    ok: bool
    status: int = 200
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    layer: str = "L0_TRANSPORT"


def l0_get(path: str, base: str = DEFAULT_BASE, timeout: float = 15) -> TransportResult:
    url = f"{base.rstrip('/')}{path}"
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, method="GET")
        if DEFAULT_API_TOKEN:
            req.add_header("Authorization", f"Bearer {DEFAULT_API_TOKEN}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return TransportResult(
                ok=True, status=resp.status, data=data,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        return TransportResult(
            ok=False, status=e.code, error=f"HTTP {e.code}: {err[:600]}",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return TransportResult(
            ok=False, status=0, error=str(e)[:600],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def l0_post(
    path: str,
    body: dict[str, Any],
    base: str = DEFAULT_BASE,
    timeout: float = 180,
) -> TransportResult:
    url = f"{base.rstrip('/')}{path}"
    t0 = time.perf_counter()
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    if DEFAULT_API_TOKEN:
        req.add_header("Authorization", f"Bearer {DEFAULT_API_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return TransportResult(
                ok=True, status=resp.status, data=payload,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        return TransportResult(
            ok=False, status=e.code, error=f"HTTP {e.code}: {err[:800]}",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return TransportResult(
            ok=False, status=0, error=str(e)[:800],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def l0_health(base: str = DEFAULT_BASE) -> dict[str, Any]:
    r = l0_get("/api/v1/models", base=base, timeout=5)
    return {
        "ok": r.ok,
        "layer": "L0_TRANSPORT",
        "base": base,
        "latency_ms": round(r.latency_ms, 1),
        "error": r.error or None,
        "gate": "PASS" if r.ok else "STOP",
    }


# ---------------------------------------------------------------------------
# L1 — RESIDENCY (load/unload is the real control surface)
# ---------------------------------------------------------------------------

@dataclass
class ModelInstance:
    key: str
    instance_id: str
    type: str
    context_length: int | None
    size_gb: float
    tool_use: bool | None
    vision: bool | None
    reasoning: Any
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResidencyPlan:
    keep: list[str]
    unload: list[str]  # instance_ids
    load: list[tuple[str, int]]  # (model, ctx)
    note: str
    free_gb_estimate: float | None = None


def l1_catalog(base: str = DEFAULT_BASE) -> dict[str, Any]:
    r = l0_get("/api/v1/models", base=base, timeout=15)
    if not r.ok:
        return {"ok": False, "layer": "L1_RESIDENCY", "error": r.error, "gate": "STOP"}
    models = []
    loaded: list[ModelInstance] = []
    for m in r.data.get("models") or []:
        key = m.get("key") or ""
        caps = m.get("capabilities") or {}
        size_gb = round((m.get("size_bytes") or 0) / 1e9, 2)
        instances = m.get("loaded_instances") or []
        entry = {
            "key": key,
            "type": m.get("type"),
            "loaded": bool(instances),
            "n_instances": len(instances),
            "loaded_instances": instances,
            "size_gb": size_gb,
            "max_context_length": m.get("max_context_length"),
            "tool_use": caps.get("trained_for_tool_use"),
            "vision": caps.get("vision"),
            "reasoning": caps.get("reasoning"),
            "quantization": (m.get("quantization") or {}).get("name"),
            "params": m.get("params_string"),
        }
        models.append(entry)
        for inst in instances:
            cfg = inst.get("config") or {}
            loaded.append(
                ModelInstance(
                    key=key,
                    instance_id=inst.get("id") or key,
                    type=m.get("type") or "llm",
                    context_length=cfg.get("context_length"),
                    size_gb=size_gb,
                    tool_use=caps.get("trained_for_tool_use"),
                    vision=caps.get("vision"),
                    reasoning=caps.get("reasoning"),
                    config=cfg,
                )
            )
    # duplicate detection — waste of RAM
    by_key: dict[str, list[str]] = {}
    for inst in loaded:
        by_key.setdefault(inst.key, []).append(inst.instance_id)
    duplicates = {k: v for k, v in by_key.items() if len(v) > MAX_INSTANCES_PER_KEY}

    return {
        "ok": True,
        "layer": "L1_RESIDENCY",
        "gate": "PASS",
        "models": models,
        "loaded": [asdict(x) for x in loaded],
        "duplicates": duplicates,
        "latency_ms": round(r.latency_ms, 1),
    }


def l1_free_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def l1_plan(
    want_llm: str = DEFAULT_LFM,
    want_embed: str = DEFAULT_EMBED,
    context_length: int = DEFAULT_CTX,
    allow_deep: bool = False,
    base: str = DEFAULT_BASE,
) -> dict[str, Any]:
    """Compute residency plan: unload thrash, keep substrate, optional deep."""
    cat = l1_catalog(base=base)
    if not cat.get("ok"):
        return {**cat, "gate": "STOP"}

    free = l1_free_ram_gb()
    unload: list[str] = []
    load: list[tuple[str, int]] = []
    keep: list[str] = []

    loaded_keys = {x["key"] for x in cat.get("loaded") or []}
    # unload duplicates (keep first instance only)
    for key, ids in (cat.get("duplicates") or {}).items():
        for extra_id in ids[1:]:
            unload.append(extra_id)

    # unload deep models unless allowed
    for inst in cat.get("loaded") or []:
        k = inst["key"]
        if k in DEEP_ONLY and not allow_deep:
            unload.append(inst["instance_id"])
        elif k in (want_llm, want_embed) or k in ALWAYS_ON:
            keep.append(inst["instance_id"])

    if want_llm not in loaded_keys:
        # RAM gate before load
        if free is not None and free < 1.5:
            return {
                "ok": False,
                "layer": "L1_RESIDENCY",
                "gate": "STOP",
                "error": f"free RAM {free:.1f}GB < 1.5GB headroom; cannot load {want_llm}",
                "plan": asdict(
                    ResidencyPlan(keep=keep, unload=unload, load=[], note="RAM gate STOP")
                ),
            }
        load.append((want_llm, context_length))
    if want_embed not in loaded_keys:
        load.append((want_embed, 2048))

    note = (
        f"substrate={want_llm}+{want_embed}; "
        f"ctx={context_length}; free≈{free}; "
        f"deep={'allowed' if allow_deep else 'blocked'}"
    )
    plan = ResidencyPlan(keep=list(set(keep)), unload=list(set(unload)), load=load, note=note, free_gb_estimate=free)
    return {
        "ok": True,
        "layer": "L1_RESIDENCY",
        "gate": "PASS",
        "plan": asdict(plan),
        "catalog_summary": {
            "n_models": len(cat.get("models") or []),
            "n_loaded": len(cat.get("loaded") or []),
            "duplicates": cat.get("duplicates"),
        },
    }


def l1_apply(plan: dict[str, Any] | None = None, base: str = DEFAULT_BASE, **kwargs) -> dict[str, Any]:
    """Execute residency plan. Idempotent ensure substrate."""
    if plan is None:
        pr = l1_plan(base=base, **kwargs)
        if not pr.get("ok"):
            return pr
        plan = pr["plan"]
    else:
        # accept raw plan dict
        if "plan" in plan:
            plan = plan["plan"]

    actions: list[dict[str, Any]] = []
    for iid in plan.get("unload") or []:
        r = l0_post("/api/v1/models/unload", {"instance_id": iid}, base=base, timeout=120)
        actions.append({"action": "unload", "instance_id": iid, "ok": r.ok, "error": r.error or None, "data": r.data if r.ok else None})

    for model, ctx in plan.get("load") or []:
        body: dict[str, Any] = {"model": model}
        if ctx:
            body["context_length"] = int(ctx)
        r = l0_post("/api/v1/models/load", body, base=base, timeout=300)
        if not r.ok and "context_length" in body:
            r = l0_post("/api/v1/models/load", {"model": model}, base=base, timeout=300)
        actions.append({
            "action": "load",
            "model": model,
            "context_length": ctx,
            "ok": r.ok,
            "error": r.error or None,
            "data": r.data if r.ok else None,
            "latency_ms": round(r.latency_ms, 1),
        })

    # verify
    cat = l1_catalog(base=base)
    keys = {x["key"] for x in cat.get("loaded") or []}
    substrate_ok = DEFAULT_LFM in keys or any(
        a.get("action") == "load" and a.get("ok") for a in actions
    )
    # if LFM already was loaded, ok
    want = kwargs.get("want_llm", DEFAULT_LFM)
    substrate_ok = want in keys or DEFAULT_LFM in keys

    return {
        "ok": substrate_ok or not (plan.get("load")),
        "layer": "L1_RESIDENCY",
        "gate": "PASS" if substrate_ok or not plan.get("load") else "NEED_INFO",
        "actions": actions,
        "loaded_keys": sorted(keys),
        "duplicates_remaining": cat.get("duplicates"),
        "plan": plan,
    }


def l1_ensure_substrate(
    model: str = DEFAULT_LFM,
    embed: str = DEFAULT_EMBED,
    context_length: int = DEFAULT_CTX,
    base: str = DEFAULT_BASE,
    consolidate: bool = True,
) -> dict[str, Any]:
    """
    Hyper-optimized residency: one LFM instance + embed, unload thrash.
    Prefer max-context LFM profile (lfm_profile.ensure_lfm_max) when model is LFM.
    """
    if model in (DEFAULT_LFM, "liquid/lfm2.5-1.2b") and consolidate:
        try:
            from lfm_profile import LFM_MAX_CTX, ensure_lfm_max

            ctx = context_length if context_length and context_length >= 8192 else LFM_MAX_CTX
            return ensure_lfm_max(
                base=base,
                context_length=ctx,
                unload_others=True,
                keep_embed=True,
            )
        except Exception as e:
            pass  # fall through to plan path
    plan_r = l1_plan(
        want_llm=model,
        want_embed=embed,
        context_length=context_length,
        allow_deep=False,
        base=base,
    )
    if not plan_r.get("ok"):
        return plan_r
    if not consolidate:
        # only load missing, don't unload
        plan_r["plan"]["unload"] = []
    return l1_apply(plan_r["plan"], base=base, want_llm=model)


# ---------------------------------------------------------------------------
# L2 — INFERENCE
# ---------------------------------------------------------------------------

def _parse_native_output(raw: dict[str, Any]) -> dict[str, Any]:
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    reasoning_parts: list[str] = []
    invalid: list[dict] = []
    for block in raw.get("output") or []:
        if not isinstance(block, dict):
            content_parts.append(str(block))
            continue
        t = block.get("type")
        if t == "message":
            content_parts.append(str(block.get("content") or ""))
        elif t == "tool_call":
            tool_calls.append(block)
        elif t == "reasoning":
            reasoning_parts.append(str(block.get("content") or ""))
        elif t == "invalid_tool_call":
            invalid.append(block)
        else:
            # responses-style nested
            if block.get("type") == "message" or block.get("role") == "assistant":
                for c in block.get("content") or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        content_parts.append(str(c.get("text") or c.get("content") or ""))
                    elif isinstance(c, str):
                        content_parts.append(c)
    return {
        "content": "".join(content_parts).strip(),
        "tool_calls": tool_calls,
        "reasoning": "\n".join(reasoning_parts).strip(),
        "invalid_tool_calls": invalid,
    }


def l2_chat(
    prompt: str,
    model: str = DEFAULT_LFM,
    system: str = "",
    temperature: float = 0.15,
    max_tokens: int = 160,
    context_length: int | None = None,
    store: bool = True,
    previous_response_id: str | None = None,
    integrations: list[Any] | None = None,
    reasoning: str | None = None,
    base: str = DEFAULT_BASE,
    timeout: float = 180,
) -> dict[str, Any]:
    """
    Native POST /api/v1/chat — primary inference fiber.
    Always store=True for stateful fiber (L3) unless explicitly disabled.
    """
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
    }
    if system:
        body["system_prompt"] = system
    if context_length:
        body["context_length"] = int(context_length)
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    if integrations:
        body["integrations"] = integrations
    if reasoning:
        body["reasoning"] = reasoning

    r = l0_post("/api/v1/chat", body, base=base, timeout=timeout)
    if not r.ok:
        return {
            "ok": False,
            "layer": "L2_INFERENCE",
            "mode": "native_chat",
            "gate": "STOP",
            "model": model,
            "error": r.error,
            "latency_ms": round(r.latency_ms, 1),
        }
    parsed = _parse_native_output(r.data)
    stats = r.data.get("stats") or {}
    return {
        "ok": True,
        "layer": "L2_INFERENCE",
        "mode": "native_chat",
        "gate": "PASS",
        "model": model,
        "content": parsed["content"],
        "reasoning": parsed["reasoning"],
        "tool_calls": parsed["tool_calls"],
        "invalid_tool_calls": parsed["invalid_tool_calls"],
        "response_id": r.data.get("response_id"),
        "model_instance_id": r.data.get("model_instance_id"),
        "stats": stats,
        "cost": {
            "input_tokens": stats.get("input_tokens"),
            "output_tokens": stats.get("total_output_tokens"),
            "tok_s": stats.get("tokens_per_second"),
            "ttft_s": stats.get("time_to_first_token_seconds"),
            "load_s": stats.get("model_load_time_seconds"),
        },
        "latency_ms": round(r.latency_ms, 1),
    }


def l2_responses(
    prompt: str,
    model: str = DEFAULT_LFM,
    instructions: str = "",
    max_tokens: int = 160,
    temperature: float = 0.15,
    store: bool = True,
    previous_response_id: str | None = None,
    tools: list[Any] | None = None,
    base: str = DEFAULT_BASE,
    timeout: float = 180,
) -> dict[str, Any]:
    """OpenAI-compat POST /v1/responses — rich stateful + tools surface."""
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "store": store,
    }
    if instructions:
        body["instructions"] = instructions
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    if tools:
        body["tools"] = tools
    r = l0_post("/v1/responses", body, base=base, timeout=timeout)
    if not r.ok:
        return {
            "ok": False,
            "layer": "L2_INFERENCE",
            "mode": "responses",
            "gate": "STOP",
            "error": r.error,
            "latency_ms": round(r.latency_ms, 1),
        }
    content_parts: list[str] = []
    for block in r.data.get("output") or []:
        if not isinstance(block, dict):
            continue
        for c in block.get("content") or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                content_parts.append(str(c.get("text") or ""))
    return {
        "ok": True,
        "layer": "L2_INFERENCE",
        "mode": "responses",
        "gate": "PASS",
        "model": model,
        "content": "".join(content_parts).strip(),
        "response_id": r.data.get("id"),
        "usage": r.data.get("usage"),
        "status": r.data.get("status"),
        "latency_ms": round(r.latency_ms, 1),
        "raw_keys": sorted(r.data.keys()),
    }


def l2_embed(
    text: str,
    model: str | None = None,
    base: str | None = None,
    timeout: float = 60,
    task: str = "search_document",
    dims: int | None = None,
    mean: list[float] | None = None,
) -> dict[str, Any]:
    """
    Job 1 ABOUTNESS embed (jina default via nomic_metric; nomic fallback).
    Not agreement/glue. Family-correct prefixes applied inside nomic_metric.
    """
    try:
        from nomic_metric import default_embed_model, embed as nomic_embed

        r = nomic_embed(
            text,
            task=task if task in (
                "search_query", "search_document", "clustering", "classification", "none"
            ) else "search_document",
            model=model or default_embed_model(),
            base=base,
            dims=dims,
            mean=mean,
            timeout=timeout,
        )
        if r.get("ok"):
            return {
                "ok": True,
                "layer": "L2_INFERENCE",
                "mode": "embed",
                "job": "retrieval_aboutness",
                "not_agreement": True,
                "gate": "PASS",
                "model": r.get("model") or model,
                "family": r.get("family"),
                "task": r.get("task"),
                "dim": r.get("dim"),
                "embedding": r.get("embedding") or [],
                "latency_ms": r.get("latency_ms"),
                "warning": r.get("warning"),
                "jina_service": r.get("jina_service"),
            }
        return {
            "ok": False,
            "layer": "L2_INFERENCE",
            "mode": "embed",
            "gate": "STOP",
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
            "family": r.get("family"),
            "jina_service": r.get("jina_service"),
        }
    except Exception as e:
        # fallback raw (off-distribution) — marked
        from nomic_metric import NOMIC_MODEL

        r = l0_post(
            "/v1/embeddings",
            {"model": model or NOMIC_MODEL, "input": text},
            base=base or DEFAULT_BASE,
            timeout=timeout,
        )
        if not r.ok:
            return {
                "ok": False,
                "layer": "L2_INFERENCE",
                "mode": "embed",
                "gate": "STOP",
                "error": r.error or str(e),
                "latency_ms": round(r.latency_ms, 1),
            }
        vec = (r.data.get("data") or [{}])[0].get("embedding") or []
        return {
            "ok": True,
            "layer": "L2_INFERENCE",
            "mode": "embed",
            "gate": "PASS",
            "model": model,
            "dim": len(vec),
            "embedding": vec,
            "latency_ms": round(r.latency_ms, 1),
            "warning": "raw embed without nomic prefix (fallback)",
            "not_agreement": True,
        }


def l2_chat_openai_fallback(
    prompt: str,
    model: str = DEFAULT_LFM,
    system: str = "",
    temperature: float = 0.15,
    max_tokens: int = 160,
    base: str = DEFAULT_BASE,
) -> dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = l0_post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        base=base,
        timeout=180,
    )
    if not r.ok:
        return {"ok": False, "layer": "L2_INFERENCE", "mode": "openai_chat", "gate": "STOP", "error": r.error}
    msg = (r.data.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return {
        "ok": True,
        "layer": "L2_INFERENCE",
        "mode": "openai_chat",
        "gate": "PASS",
        "model": model,
        "content": content,
        "usage": r.data.get("usage"),
    }


# ---------------------------------------------------------------------------
# L3 — STATE FIBER (response_id memory)
# ---------------------------------------------------------------------------

class StateFiber:
    """Persistent fiber trajectory via LMS store + previous_response_id."""

    def __init__(self, model: str = DEFAULT_LFM, base: str = DEFAULT_BASE):
        self.model = model
        self.base = base
        self.response_id: str | None = None
        self.turns: list[dict[str, Any]] = []
        self.role_ids: dict[str, str] = {}

    def reset(self) -> None:
        self.response_id = None
        self.turns.clear()
        self.role_ids.clear()

    def chat(
        self,
        prompt: str,
        system: str = "",
        role: str | None = None,
        chain: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        prev = self.response_id if chain and self.response_id else None
        r = l2_chat(
            prompt,
            model=self.model,
            system=system,
            previous_response_id=prev,
            store=True,
            base=self.base,
            **kwargs,
        )
        if r.get("ok") and r.get("response_id"):
            self.response_id = r["response_id"]
            if role:
                self.role_ids[role] = r["response_id"]
        self.turns.append({
            "role": role,
            "ok": r.get("ok"),
            "response_id": r.get("response_id"),
            "content_preview": (r.get("content") or "")[:200],
            "cost": r.get("cost"),
        })
        r["fiber_turns"] = len(self.turns)
        r["layer"] = "L3_STATE_FIBER"
        return r

    def snapshot(self) -> dict[str, Any]:
        return {
            "layer": "L3_STATE_FIBER",
            "model": self.model,
            "response_id": self.response_id,
            "role_ids": dict(self.role_ids),
            "n_turns": len(self.turns),
            "turns": self.turns[-8:],
        }


# ---------------------------------------------------------------------------
# L4 — STRUCTURE OPS (JSON-gated roles)
# ---------------------------------------------------------------------------

ROLE_SCHEMAS: dict[str, dict[str, Any]] = {
    "SCOUT": {
        "system": (
            "ROLE=SCOUT. Map human intent only. "
            "Output STRICT JSON only, no markdown. "
            "Fields what/domain/success must be concrete phrases from THIS intent "
            "(min 8 chars each). Never use ellipsis or placeholders. "
            'Schema: {"what":"<concrete>","domain":"<concrete>","success":"<concrete>"}'
        ),
        "required": ("what", "domain", "success"),
        "temp": 0.1,
        "max_tokens": 120,
    },
    "FALSIFY": {
        "system": (
            "ROLE=FALSIFY. Attack the plan with concrete failure modes. "
            "Output STRICT JSON only. attacks must be 3 specific risks for THIS intent "
            "(not generic). Never use ellipsis. "
            'Schema: {"attacks":["<specific>","<specific>","<specific>"],'
            '"fatal":false,"note":"<one line>"}'
        ),
        "required": ("attacks", "fatal"),
        "temp": 0.1,
        "max_tokens": 140,
    },
    "GLUE": {
        "system": (
            "ROLE=GLUE. Name interface terms that must hold on BOTH human and domain sides. "
            "shared MUST include domain-specific terms from the intent "
            "(e.g. model names, APIs, file concepts) — NOT only the words "
            "open/stop/measure/audit/residue (those are law-core; repeating them alone is FAIL). "
            "missing = concrete gaps. risk = one concrete risk sentence. "
            "Never use ellipsis or '...'. "
            'Schema: {"shared":["<domain term>",...],"missing":["<gap>",...],"risk":"<sentence>"}'
        ),
        "required": ("shared", "missing", "risk"),
        "temp": 0.1,
        "max_tokens": 140,
    },
    "VERDICT": {
        "system": (
            "ROLE=VERDICT. Law: restrict→measure→audit→OPEN|STOP. Residue never forced. "
            "You have NO authority to OPEN production alone. "
            "reason must be specific to THIS intent (min 12 chars), never filler "
            "like 'task is ongoing' or 'monitoring required'. "
            'Schema: {"verdict":"OPEN_CANDIDATE|STOP|NEED_INFO","reason":"<specific>"}'
        ),
        "required": ("verdict", "reason"),
        "temp": 0.05,
        "max_tokens": 100,
    },
}


def l4_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        from metric_text import parse_json_loose

        obj = parse_json_loose(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    t = text.strip()
    # strip code fences if model ignored instructions
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    # try full parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # salvage first {...} with light repair (trailing junk brackets common on tiny models)
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    blob = m.group(0)
    for candidate in (blob, re.sub(r",\s*}", "}", blob), re.sub(r"\]\s*}", "}", blob), re.sub(r",\s*]", "]", blob)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def l4_validate(role: str, obj: dict[str, Any] | None) -> dict[str, Any]:
    schema = ROLE_SCHEMAS.get(role) or {}
    req = schema.get("required") or ()
    if obj is None:
        return {"ok": False, "gate": "STOP", "error": "unparseable_json", "role": role}
    missing = [k for k in req if k not in obj]
    if missing:
        return {"ok": False, "gate": "NEED_INFO", "error": f"missing keys {missing}", "role": role, "partial": obj}
    # normalize verdict
    if role == "VERDICT":
        v = str(obj.get("verdict", "")).upper().replace(" ", "_")
        if v not in ("OPEN_CANDIDATE", "STOP", "NEED_INFO", "OPEN"):
            if "NEED" in v:
                v = "NEED_INFO"
            elif "STOP" in v:
                v = "STOP"
            elif "OPEN" in v:
                v = "OPEN_CANDIDATE"
            else:
                v = "NEED_INFO"
        if v == "OPEN":
            v = "OPEN_CANDIDATE"
        obj = {**obj, "verdict": v}
    # contentful payload check (rejects ellipsis / law-core-only GLUE / filler reasons)
    from metric_text import validate_role_payload

    content = validate_role_payload(role, obj)
    if not content.get("ok"):
        return {
            "ok": False,
            "gate": "NEED_INFO",
            "error": "placeholder_or_filler:" + ",".join(content.get("errors") or []),
            "role": role,
            "partial": obj,
            "content_check": content,
        }
    return {
        "ok": True,
        "gate": "PASS",
        "role": role,
        "data": obj,
        "payload_text": content.get("payload_text"),
        "content_warnings": content.get("warnings") or [],
    }


def l4_role(
    role: str,
    prompt: str,
    fiber: StateFiber | None = None,
    base: str = DEFAULT_BASE,
    model: str = DEFAULT_LFM,
    chain: bool = True,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Run one structured role with JSON gate + optional retry."""
    schema = ROLE_SCHEMAS.get(role)
    if not schema:
        return {"ok": False, "layer": "L4_STRUCTURE", "gate": "STOP", "error": f"unknown role {role}"}

    fiber = fiber or StateFiber(model=model, base=base)
    last: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        user = prompt
        # Retry must not chain bad fiber state — tiny models copy prior wrong schema
        use_chain = bool(chain and role != "SCOUT" and attempt == 0)
        if attempt > 0:
            user = (
                f"ROLE={role}. Emit STRICT JSON with keys exactly: "
                f"{list(schema.get('required') or [])}. No markdown.\n\n"
                f"TASK:\n{prompt}"
            )
            use_chain = False
        r = fiber.chat(
            user,
            system=schema["system"],
            role=role,
            chain=use_chain,
            temperature=schema["temp"],
            max_tokens=schema["max_tokens"],
        )
        obj = l4_parse_json(r.get("content") or "")
        val = l4_validate(role, obj)
        last = {
            "ok": val.get("ok") and r.get("ok"),
            "layer": "L4_STRUCTURE",
            "role": role,
            "gate": val.get("gate") if r.get("ok") else "STOP",
            "raw": (r.get("content") or "")[:600],
            "parsed": val.get("data") or val.get("partial"),
            "validate_error": val.get("error"),
            "response_id": r.get("response_id"),
            "cost": r.get("cost"),
            "stats": r.get("stats"),
            "attempt": attempt + 1,
            "inference_ok": r.get("ok"),
            "error": r.get("error"),
        }
        if last["ok"]:
            return last
    return last


def l4_ops_pass(
    prompt: str,
    roles: list[str] | None = None,
    base: str = DEFAULT_BASE,
    model: str = DEFAULT_LFM,
    ensure_residency: bool = True,
    context_length: int = DEFAULT_CTX,
    embed: bool = True,
) -> dict[str, Any]:
    """
    Full orthogonal role algebra with layered gates.
    L0.5 local home → L0 health → L1 ensure → L3 fiber → L4 roles → L5 metric → L6 policy.
    """
    gates: list[dict[str, Any]] = []
    home_pol = _home_policy()
    # Base from local LMS; context from RAM/model policy — NOT LMS UI default (4096/8192)
    if base == DEFAULT_BASE and home_pol.get("base_url"):
        base = str(home_pol["base_url"]).rstrip("/")
    try:
        from ctx_policy import resolve_load_context

        # Only re-resolve when caller left the module default (or tiny UI-ish values)
        if context_length in (DEFAULT_CTX, 4096, 8192) or context_length is None:
            pol = resolve_load_context(model, purpose="chat", base=base)
            context_length = int(pol["context_length"])
            gates.append({
                "step": "L0.5_ctx_policy",
                "ok": True,
                "gate": "PASS",
                "context_length": context_length,
                "reading": pol.get("reading"),
                "enough_for_daily_pack": pol.get("enough_for_daily_pack"),
            })
    except Exception as e:
        gates.append({
            "step": "L0.5_ctx_policy",
            "ok": False,
            "gate": "NEED_INFO",
            "error": str(e),
            "context_length": context_length,
        })

    from metric_text import strip_prompt_chrome

    # Scheduler/system-reminder chrome must not become "intent" (log: search_query: <system-reminder>)
    prompt_clean = strip_prompt_chrome(prompt)
    if not prompt_clean.strip():
        prompt_clean = (prompt or "").strip()

    chain_max = int(home_pol.get("chain_max_input_chars") or 3000)
    pack_budget = int(home_pol.get("pack_budget_chars") or 3500)
    # Cap prompt to avoid Context size exceeded (log-proven on this kit)
    prompt_capped = (
        prompt_clean if len(prompt_clean) <= pack_budget
        else (prompt_clean[: pack_budget - 20] + "\n…[capped]")
    )
    allow_fiber_chain = len(prompt_capped) <= chain_max

    gates.append({
        "step": "L0.5_local_home",
        "ok": bool(home_pol.get("ok")),
        "gate": home_pol.get("log_gate") or "PASS",
        "cpu_only_engine": home_pol.get("cpu_only_engine"),
        "default_ctx": context_length,
        "pack_budget_chars": pack_budget,
        "chain_max_input_chars": chain_max,
        "fiber_chain": allow_fiber_chain,
        "prompt_chars": len(prompt),
        "prompt_clean_chars": len(prompt_clean),
        "prompt_capped_chars": len(prompt_capped),
        "chrome_stripped": len(prompt_clean) < len(prompt or ""),
        "log_signals": [g.get("signal") for g in (home_pol.get("log_signals") or [])],
    })

    # L0
    h = l0_health(base=base)
    gates.append({"step": "L0_health", **{k: h[k] for k in ("ok", "gate", "latency_ms", "error") if k in h}})
    if not h.get("ok"):
        return {
            "ok": False,
            "layer": "L0_TRANSPORT",
            "gate": "STOP",
            "gates": gates,
            "error": "LMS unreachable",
            "home_policy": {k: home_pol.get(k) for k in (
                "home", "cpu_only_engine", "default_context_length", "recommendations"
            ) if k in home_pol},
            "not_open_authority": True,
        }

    # L1
    if ensure_residency:
        res = l1_ensure_substrate(model=model, context_length=context_length, base=base)
        gates.append({
            "step": "L1_residency",
            "ok": res.get("ok"),
            "gate": res.get("gate"),
            "loaded_keys": res.get("loaded_keys"),
            "duplicates_remaining": res.get("duplicates_remaining"),
            "n_actions": len(res.get("actions") or []),
        })
        if res.get("gate") == "STOP":
            return {
                "ok": False,
                "layer": "L1_RESIDENCY",
                "gate": "STOP",
                "gates": gates,
                "error": res.get("error") or "residency STOP",
                "residency": res,
                "not_open_authority": True,
            }

    fiber = StateFiber(model=model, base=base)
    use_roles = roles or list(ROLE_SCHEMAS.keys())
    outputs: dict[str, Any] = {}
    chain_context: list[str] = []

    for role in use_roles:
        user = prompt_capped
        if role == "VERDICT" and chain_context:
            # keep VERDICT payload tight — prior JSON only, not full pack again
            user = (
                f"INTENT_SUMMARY:\n{prompt_capped[:800]}\n\nPRIOR_ROLES_JSON:\n"
                + "\n".join(chain_context)
                + "\n\nEmit VERDICT JSON."
            )
        # Large packs: disable previous_response_id chain (context overflow in logs)
        do_chain = allow_fiber_chain and (role != "SCOUT")
        r = l4_role(
            role,
            user,
            fiber=fiber,
            base=base,
            model=model,
            chain=do_chain,
            max_retries=1,
        )
        outputs[role] = r
        gates.append({
            "step": f"L4_{role}",
            "ok": r.get("ok"),
            "gate": r.get("gate"),
            "attempt": r.get("attempt"),
            "cost": r.get("cost"),
        })
        if r.get("parsed"):
            chain_context.append(f"{role}: {json.dumps(r['parsed'], ensure_ascii=False)[:400]}")
        # hard stop if FALSIFY unparseable after retries? continue to verdict with NEED_INFO bias

    # L5 metric
    metric = l5_metric(prompt, outputs, base=base, do_embed=embed)
    gates.append({"step": "L5_metric", "ok": metric.get("ok"), "gate": metric.get("gate"), "mean_cosine": metric.get("mean_cosine")})

    # L6 policy
    policy = l6_policy(outputs, metric)
    gates.append({"step": "L6_policy", "ok": True, "gate": policy.get("gate"), "verdict": policy.get("verdict")})

    return {
        "ok": True,
        "mode": "lms_layered_ops",
        "layer": "L7_ORCHESTRATION",
        "thesis": (
            "Hyper-optimized LMS stack: residency → native chat fiber → JSON roles → "
            "embed glue → policy gate. Structure multiplies small-model capability."
        ),
        "model": model,
        "roles": use_roles,
        "outputs": {
            k: {
                "ok": v.get("ok"),
                "gate": v.get("gate"),
                "parsed": v.get("parsed"),
                "raw": v.get("raw"),
                "response_id": v.get("response_id"),
                "cost": v.get("cost"),
                "validate_error": v.get("validate_error"),
            }
            for k, v in outputs.items()
        },
        "verdict": policy.get("verdict"),
        "fatal_flag": policy.get("fatal_flag"),
        "policy_reason": policy.get("reason"),
        "embeddings": metric.get("embeddings"),
        "mean_cosine": metric.get("mean_cosine"),  # aboutness diagnostic only
        "aboutness": metric.get("aboutness"),
        "agreement": metric.get("agreement"),  # Job 2 NLI glue
        "fiber": fiber.snapshot(),
        "gates": gates,
        "gate": policy.get("gate"),
        "regime": policy.get("regime"),
        "not_open_authority": True,
        "sheaf_tag": "lms_layered_role_restriction",
        "note": "MEASURE only. Production OPEN requires domain measures + Prime audit.",
        "residency_loaded": next(
            (g.get("loaded_keys") for g in gates if g.get("step") == "L1_residency"), None
        ),
        "home_policy": {
            "home": home_pol.get("home"),
            "cpu_only_engine": home_pol.get("cpu_only_engine"),
            "default_context_length": context_length,
            "pack_budget_chars": pack_budget,
            "fiber_chain": allow_fiber_chain,
            "sampling_lfm": home_pol.get("sampling_lfm"),
            "recommendations": (home_pol.get("recommendations") or [])[:6],
        },
    }


# ---------------------------------------------------------------------------
# L5 — METRIC
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def l5_metric(
    prompt: str,
    outputs: dict[str, Any],
    base: str = DEFAULT_BASE,
    do_embed: bool = True,
) -> dict[str, Any]:
    """
    L5 is now dual:
      - aboutness: jina (nomic fallback) cosine (diagnostic only — NOT agreement)
      - agreement: NLI glue on VERDICT/SCOUT text vs human (Job 2)

    Cosine must never promote OPEN. NLI contradiction → STOP bias.
    """
    if not do_embed:
        return {
            "ok": True,
            "layer": "L5_METRIC",
            "gate": "PASS",
            "embeddings": {},
            "mean_cosine": None,
            "aboutness": None,
            "agreement": None,
            "skipped": True,
        }

    # --- Job 1 diagnostic: aboutness ---
    he = l2_embed(prompt, base=base, task="search_query")
    if not he.get("ok"):
        return {
            "ok": False,
            "layer": "L5_METRIC",
            "gate": "NEED_INFO",
            "error": he.get("error"),
            "embeddings": {},
        }

    from metric_text import strip_envelope

    sims: dict[str, float] = {}
    role_texts: dict[str, str] = {}
    role_payloads: dict[str, str] = {}
    for role, o in outputs.items():
        # ABOUTNESS: strip JSON envelope — never embed braces/keys
        if o.get("parsed") is not None:
            text = strip_envelope(o["parsed"])
            hyp_for_nli = strip_envelope(o["parsed"]) or json.dumps(o["parsed"], ensure_ascii=False)
        else:
            raw = o.get("raw") or o.get("content") or ""
            text = strip_envelope(raw)
            hyp_for_nli = text or raw
        if not text.strip():
            # empty after strip = envelope-only / placeholder — skip aboutness
            role_texts[role] = hyp_for_nli
            role_payloads[role] = ""
            continue
        role_texts[role] = hyp_for_nli
        role_payloads[role] = text
        ee = l2_embed(text[:2000], base=base, task="search_document")
        if ee.get("ok"):
            sims[role] = round(cosine(he["embedding"], ee["embedding"]), 4)

    mean = round(sum(sims.values()) / max(len(sims), 1), 4) if sims else None

    # --- Job 2: agreement NLI on stripped payload (not JSON envelope) ---
    agreement: dict[str, Any] | None = None
    try:
        from entailment_glue import glue_agreement

        hyp = (
            role_payloads.get("VERDICT")
            or role_texts.get("VERDICT")
            or role_payloads.get("SCOUT")
            or role_texts.get("SCOUT")
            or ""
        )
        # strip human prompt of instruction chrome if short enough leave as is
        premise = strip_envelope(prompt) if prompt.strip().startswith("{") else prompt
        if hyp and len(hyp.strip()) >= 8:
            # DeBERTa/ORT first (prefer=auto); LFM only as fallback inside glue_agreement
            agreement = glue_agreement(premise[:1800], hyp[:800], prefer="auto", base=base)
        else:
            agreement = {
                "ok": False,
                "label": "unknown",
                "gate": "NEED_INFO",
                "error": "empty_payload_after_envelope_strip",
                "agrees": False,
            }
    except Exception as e:
        agreement = {"ok": False, "error": str(e), "label": "unknown", "gate": "NEED_INFO", "agrees": False}

    # Gate from AGREEMENT only; aboutness never promotes
    gate = "PASS"
    if agreement and agreement.get("label") == "contradiction":
        gate = "STOP"
    elif agreement and not agreement.get("agrees") and agreement.get("label") in ("neutral", "unknown", None):
        gate = "NEED_INFO"
    elif agreement and agreement.get("agrees"):
        gate = "PASS"
    # aboutness floor is diagnostic only
    aboutness_warn = mean is not None and mean < 0.25

    return {
        "ok": True,
        "layer": "L5_METRIC",
        "gate": gate,
        "dim": he.get("dim"),
        "embeddings": {
            "job": "retrieval_aboutness",
            "not_agreement": True,
            "envelope_stripped": True,
            "cosine_role_to_human": sims,
            "mean_cosine": mean,
            "payload_previews": {k: (v or "")[:120] for k, v in role_payloads.items()},
            "warning": "aboutness only — never agreement; envelope stripped before embed",
        },
        "mean_cosine": mean,  # legacy key; aboutness diagnostic only
        "aboutness": {
            "mean_cosine": mean,
            "per_role": sims,
            "not_agreement": True,
            "envelope_stripped": True,
        },
        "agreement": agreement,
        "thesis": (
            "Nomic = aboutness chart (stripped payloads + prefixes). "
            "NLI = agreement. Certificate = logogram. "
            "Lift is earned only if it makes something newly decidable."
        ),
    }


# ---------------------------------------------------------------------------
# L6 — POLICY GATE (design law)
# ---------------------------------------------------------------------------

def l6_policy(outputs: dict[str, Any], metric: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Collapse role outputs + metric into candidate verdict.
    NEVER elevates LMS alone to production OPEN.
    """
    fatal = False
    fals = outputs.get("FALSIFY") or {}
    if fals.get("parsed") and fals["parsed"].get("fatal") is True:
        fatal = True
    if fals.get("raw") and re.search(r'"fatal"\s*:\s*true', fals.get("raw") or "", re.I):
        fatal = True

    verd = outputs.get("VERDICT") or {}
    verdict = "NEED_INFO"
    reason = ""
    if verd.get("parsed"):
        verdict = str(verd["parsed"].get("verdict") or "NEED_INFO")
        reason = str(verd["parsed"].get("reason") or "")
    elif not verd.get("ok"):
        verdict = "NEED_INFO"
        reason = "VERDICT parse failed"

    if fatal and verdict == "OPEN_CANDIDATE":
        verdict = "STOP"
        reason = (reason + " | FATAL from FALSIFY demotes OPEN_CANDIDATE").strip(" |")

    # role health
    role_oks = [bool((outputs.get(r) or {}).get("ok")) for r in ("SCOUT", "FALSIFY", "GLUE", "VERDICT") if r in outputs]
    if role_oks and not any(role_oks):
        verdict = "STOP"
        reason = "all roles failed"
        regime = "ops_failed"
        gate = "STOP"
    elif role_oks and not all(role_oks):
        if verdict == "OPEN_CANDIDATE":
            verdict = "NEED_INFO"
            reason = (reason + " | partial role failure").strip(" |")
        regime = "partial_ops"
        gate = "NEED_INFO" if verdict == "NEED_INFO" else ("STOP" if verdict == "STOP" else "PASS")
    else:
        regime = "structured_ops"
        gate = "PASS" if verdict in ("OPEN_CANDIDATE", "STOP", "NEED_INFO") else "NEED_INFO"

    # agreement (NLI) owns glue — not nomic cosine aboutness
    agree = (metric or {}).get("agreement") or {}
    if agree.get("label") == "contradiction":
        if verdict == "OPEN_CANDIDATE":
            verdict = "STOP"
            reason = (reason + " | NLI contradiction vs human").strip(" |")
            gate = "STOP"
            fatal = fatal or False
    elif agree.get("ok") and not agree.get("agrees") and verdict == "OPEN_CANDIDATE":
        # neutral/unknown: demote candidate, do not force STOP
        verdict = "NEED_INFO"
        reason = (reason + f" | NLI {agree.get('label')} not entailment").strip(" |")
        gate = "NEED_INFO"
    # aboutness never promotes OPEN; optional diagnostic demote only if extreme
    if metric and metric.get("mean_cosine") is not None and metric["mean_cosine"] < 0.10:
        if verdict == "OPEN_CANDIDATE":
            reason = (reason + " | extreme low aboutness (diagnostic)").strip(" |")

    # Map to gate language used by orchestration
    if verdict == "STOP" or fatal:
        gate = "STOP"
    elif verdict == "NEED_INFO":
        gate = "NEED_INFO"
    elif verdict == "OPEN_CANDIDATE":
        gate = "PASS"  # pass as *candidate measure*, not production OPEN

    return {
        "layer": "L6_POLICY",
        "verdict": verdict,
        "fatal_flag": fatal,
        "reason": reason,
        "regime": regime,
        "gate": gate,
        "not_open_authority": True,
        "law": "restrict→measure→audit→OPEN|STOP",
    }


# ---------------------------------------------------------------------------
# L7 helpers — surface for MCP / deep_loop
# ---------------------------------------------------------------------------

def layered_enter(prompt: str, **kwargs) -> dict[str, Any]:
    """Public entry: full layered pass. Alias for l4_ops_pass with orchestration tag."""
    return l4_ops_pass(prompt, **kwargs)


def layer_matrix() -> dict[str, Any]:
    """Document the gate matrix for operators / docs."""
    pol = _home_policy()
    return {
        "layers": [
            {"id": "L0.5", "name": "LOCAL_HOME", "activity": "~/.lmstudio settings+logs+model.yaml", "gate_on_fail": "NEED_INFO"},
            {"id": "L0", "name": "TRANSPORT", "activity": "HTTP GET/POST", "gate_on_fail": "STOP"},
            {"id": "L1", "name": "RESIDENCY", "activity": "catalog/load/unload/dedupe", "gate_on_fail": "STOP"},
            {"id": "L2", "name": "INFERENCE", "activity": "/api/v1/chat|/v1/responses|/v1/embeddings", "gate_on_fail": "STOP"},
            {"id": "L3", "name": "STATE_FIBER", "activity": "previous_response_id chain (off if pack large)", "gate_on_fail": "NEED_INFO"},
            {"id": "L4", "name": "STRUCTURE_OPS", "activity": "JSON roles SCOUT→…→VERDICT", "gate_on_fail": "NEED_INFO|retry"},
            {"id": "L5", "name": "METRIC", "activity": "nomic cosine glue", "gate_on_fail": "NEED_INFO soft"},
            {"id": "L6", "name": "POLICY", "activity": "FATAL demote, never production OPEN", "gate_on_fail": "STOP"},
            {"id": "L7", "name": "ORCHESTRATION", "activity": "enter/deep/dimensional handoff", "gate_on_fail": "STOP"},
        ],
        "lms_surfaces": {
            "native": ["/api/v1/models", "/api/v1/models/load", "/api/v1/models/unload", "/api/v1/chat"],
            "openai": ["/v1/models", "/v1/chat/completions", "/v1/embeddings", "/v1/responses"],
            "local_home": str(pol.get("home") or "~/.lmstudio"),
            "server_logs": "server-logs/YYYY-MM/*.log",
            "state_field": "previous_response_id (verified live; disabled on large packs)",
            "store_default": True,
            "mcp_integrations": "ephemeral_mcp | plugin mcp/<label> on /api/v1/chat",
            "cpu_only_engine": pol.get("cpu_only_engine"),
        },
        "substrate": {
            "llm": DEFAULT_LFM,
            "embed": DEFAULT_EMBED,
            "ctx": pol.get("default_context_length") or DEFAULT_CTX,
            "sampling": pol.get("sampling_lfm"),
            "pack_budget_chars": pol.get("pack_budget_chars"),
        },
        "law": "restrict→measure→audit→OPEN|STOP; residue never forced; LMS is measure only",
    }


if __name__ == "__main__":
    import sys

    cmd = (sys.argv[1] if len(sys.argv) > 1 else "matrix").lower()
    if cmd == "matrix":
        print(json.dumps(layer_matrix(), indent=2))
    elif cmd == "health":
        print(json.dumps(l0_health(), indent=2))
    elif cmd == "catalog":
        print(json.dumps(l1_catalog(), indent=2)[:4000])
    elif cmd == "ensure":
        print(json.dumps(l1_ensure_substrate(), indent=2))
    elif cmd in ("home", "local", "policy"):
        from lms_home import snapshot, derived_policy, scan_server_log

        if cmd == "policy":
            print(json.dumps(derived_policy(), indent=2))
        elif len(sys.argv) > 2 and sys.argv[2] == "logs":
            print(json.dumps(scan_server_log(), indent=2))
        else:
            print(json.dumps(snapshot(), indent=2)[:14000])
    elif cmd == "ops":
        p = " ".join(sys.argv[2:]) or "Verify Prime LMS layered gates work on Snapdragon kit."
        print(json.dumps(layered_enter(p, embed=True), indent=2)[:8000])
    else:
        print("usage: lms_layers.py [matrix|health|catalog|ensure|home|policy|ops <prompt>]")
