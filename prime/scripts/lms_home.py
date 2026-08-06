"""
LMS local home observer — `~/.lmstudio` + server-logs as gate inputs.

Not another API client. This is the **filesystem / runtime truth** layer:

  ~/.lmstudio/settings.json              policy (ctx default, JIT TTL, guardrails)
  ~/.lmstudio/.internal/http-server-config.json
  ~/.lmstudio/.internal/model-data.json  lastLoaded, artifact keys
  ~/.lmstudio/.internal/backend-preferences-v1.json
  ~/.lmstudio/hub/models/*/model.yaml    minMemory, sampling, tool_use
  ~/.lmstudio/mcp.json                   LMS-side MCP plugins
  ~/.lmstudio/server-logs/YYYY-MM/*.log  ERROR/WARN/context overflow/GPU offload
  ~/.lmstudio/extensions/backends/*      engine version / arch

Feeds L0.5 LOCAL into lms_layers so gates match *this machine*, not generic docs.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _home() -> Path:
    env = os.environ.get("LM_STUDIO_HOME") or os.environ.get("LMS_HOME")
    if env:
        return Path(env)
    # Windows default + portable
    candidates = [
        Path.home() / ".lmstudio",
        Path(os.environ.get("USERPROFILE", "")) / ".lmstudio",
        Path(r"C:\Users\Deving-1\.lmstudio"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return Path.home() / ".lmstudio"


HOME = _home()


def _read_json(path: Path) -> Any | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path, max_chars: int = 0) -> str | None:
    try:
        if not path.is_file():
            return None
        t = path.read_text(encoding="utf-8", errors="replace")
        if max_chars and len(t) > max_chars:
            return t[-max_chars:]
        return t
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Settings / server / backends
# ---------------------------------------------------------------------------

def read_settings() -> dict[str, Any]:
    raw = _read_json(HOME / "settings.json") or {}
    dev = raw.get("developer") or {}
    guard = raw.get("modelLoadingGuardrails") or {}
    dctx = raw.get("defaultContextLength") or {}
    return {
        "ok": bool(raw),
        "path": str(HOME / "settings.json"),
        "downloads_folder": raw.get("downloadsFolder"),
        "developer_mode": raw.get("developerMode"),
        "default_context_length": dctx.get("value") if isinstance(dctx, dict) else dctx,
        "model_loading_guardrails": {
            "mode": guard.get("mode"),
            "threshold_bytes": guard.get("customThresholdBytes"),
            "threshold_gb": round((guard.get("customThresholdBytes") or 0) / 1e9, 2),
            "always_allow_load_anyway": guard.get("alwaysAllowLoadAnyway"),
        },
        "jit": {
            "unload_previous_on_load": dev.get("unloadPreviousJITModelOnLoad"),
            "ttl_enabled": (dev.get("jitModelTTL") or {}).get("enabled"),
            "ttl_seconds": (dev.get("jitModelTTL") or {}).get("ttlSeconds"),
        },
        "chat_unload_previous_on_select": (raw.get("chat") or {}).get("unloadPreviousModelOnSelect"),
        "separate_reasoning_in_api": dev.get("separateReasoningContentInAPI"),
        "enable_local_service": raw.get("enableLocalService"),
        "enable_engine_protocol": raw.get("enableEngineProtocolRuntime"),
        "show_resource_widget": dev.get("showResourceConsumptionWidget"),
        "raw_keys": sorted(raw.keys()) if raw else [],
    }


def read_http_server() -> dict[str, Any]:
    raw = _read_json(HOME / ".internal" / "http-server-config.json") or {}
    return {
        "ok": bool(raw),
        "path": str(HOME / ".internal" / "http-server-config.json"),
        "port": raw.get("port", 1234),
        "bind": raw.get("networkInterface", "127.0.0.1"),
        "auto_start": raw.get("autoStartOnLaunch"),
        "jit_model_loading": raw.get("justInTimeModelLoading"),
        "cors": raw.get("cors"),
        "verbose": raw.get("verbose"),
        "file_logging_mode": raw.get("fileLoggingMode"),
        "log_sensitive_data": raw.get("logSensitiveData"),
        "log_incoming_tokens": raw.get("logIncomingTokens"),
        "base_url": f"http://{raw.get('networkInterface') or '127.0.0.1'}:{raw.get('port') or 1234}",
    }


def read_mcp_json() -> dict[str, Any]:
    raw = _read_json(HOME / "mcp.json") or {}
    servers = raw.get("mcpServers") or {}
    return {
        "ok": True,
        "path": str(HOME / "mcp.json"),
        "n_servers": len(servers),
        "labels": list(servers.keys()),
        "empty": len(servers) == 0,
        "note": "Empty mcp.json → no LMS plugin MCPs; ephemeral_mcp still works via API",
    }


def read_backend() -> dict[str, Any]:
    prefs = _read_json(HOME / ".internal" / "backend-preferences-v1.json") or []
    backends_dir = HOME / "extensions" / "backends"
    installed = []
    if backends_dir.is_dir():
        for d in sorted(backends_dir.iterdir()):
            if not d.is_dir() or d.name == "vendor":
                continue
            man = _read_json(d / "backend-manifest.json") or {}
            installed.append({
                "dir": d.name,
                "name": man.get("name"),
                "version": man.get("version"),
                "engine": man.get("engine"),
                "arch": (man.get("cpu") or {}).get("architecture"),
                "formats": man.get("supported_model_formats"),
                "min_lmstudio": man.get("minimum_lmstudio_version"),
            })
    return {
        "ok": True,
        "preferred": prefs,
        "installed": installed,
        "active_note": (
            "ARM64 llama.cpp on this kit: logs show "
            "'compiled without support for GPU offload' — CPU-only engine path."
        ),
    }


def _parse_model_yaml(path: Path) -> dict[str, Any]:
    text = _read_text(path) or ""
    out: dict[str, Any] = {"path": str(path)}
    m = re.search(r"(?m)^model:\s*(\S+)", text)
    if m:
        out["model"] = m.group(1)
    m = re.search(r"minMemoryUsageBytes:\s*(\d+)", text)
    if m:
        out["min_memory_bytes"] = int(m.group(1))
        out["min_memory_gb"] = round(int(m.group(1)) / 1e9, 2)
    m = re.search(r"contextLengths:\s*\n\s*-\s*(\d+)", text)
    if m:
        out["context_length_max"] = int(m.group(1))
    for key, field in (
        ("temperature", r"llm\.prediction\.temperature\n\s+value:\s*([\d.]+)"),
        ("top_k", r"llm\.prediction\.topKSampling\n\s+value:\s*(\d+)"),
    ):
        m = re.search(field, text)
        if m:
            out[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    m = re.search(
        r"llm\.prediction\.topPSampling\n\s+value:\n\s+checked:\s*true\n\s+value:\s*([\d.]+)",
        text,
    )
    if m:
        out["top_p"] = float(m.group(1))
    m = re.search(r"trainedForToolUse:\s*(true|false)", text)
    if m:
        out["trained_for_tool_use"] = m.group(1) == "true"
    m = re.search(r"reasoning:\s*(true|false)", text)
    if m:
        out["reasoning"] = m.group(1) == "true"
    m = re.search(r"vision:\s*(true|false)", text)
    if m:
        out["vision"] = m.group(1) == "true"
    return out


def read_hub_models() -> dict[str, Any]:
    hub = HOME / "hub" / "models"
    models = []
    if hub.is_dir():
        for yml in hub.rglob("model.yaml"):
            models.append(_parse_model_yaml(yml))
    return {"ok": True, "n": len(models), "models": models}


def read_model_data() -> dict[str, Any]:
    """lastLoaded timestamps + artifact keys from LMS internal index."""
    raw = _read_json(HOME / ".internal" / "model-data.json")
    if not raw:
        return {"ok": False, "error": "no model-data.json"}
    # shape: {"json": [[key, meta], ...], "meta": ...}
    entries = []
    try:
        pairs = raw.get("json") or []
        for key, meta in pairs:
            if not isinstance(meta, dict):
                continue
            ts = meta.get("lastLoadedTimestamp")
            entries.append({
                "key": key,
                "last_loaded_ms": ts,
                "transitive": meta.get("transitive"),
                "source": meta.get("source"),
            })
    except Exception as e:
        return {"ok": False, "error": str(e)}
    # sort by last loaded
    with_ts = [e for e in entries if e.get("last_loaded_ms")]
    with_ts.sort(key=lambda e: e["last_loaded_ms"] or 0, reverse=True)
    return {
        "ok": True,
        "n": len(entries),
        "recently_loaded": with_ts[:12],
        "hub_keys": [e["key"] for e in entries if "/" in str(e["key"]) and not str(e["key"]).endswith(".gguf")][:20],
    }


def read_plugins() -> dict[str, Any]:
    root = HOME / "extensions" / "plugins"
    plugs = []
    if root.is_dir():
        for man in root.rglob("manifest.json"):
            m = _read_json(man) or {}
            plugs.append({
                "path": str(man.parent.relative_to(root)) if man.parent != root else man.parent.name,
                "name": m.get("name") or man.parent.name,
                "type": m.get("type"),
                "revision": m.get("revision"),
            })
    return {"ok": True, "plugins": plugs}


# ---------------------------------------------------------------------------
# Server logs
# ---------------------------------------------------------------------------

def _latest_log() -> Path | None:
    logs = HOME / "server-logs"
    if not logs.is_dir():
        return None
    files = sorted(logs.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def scan_server_log(
    path: Path | None = None,
    max_chars: int = 2_500_000,
    tail_errors: int = 8,
) -> dict[str, Any]:
    path = path or _latest_log()
    if not path or not path.is_file():
        return {"ok": False, "error": "no server log found", "home": str(HOME)}

    # read tail for large files
    size = path.stat().st_size
    if size > max_chars:
        with open(path, "rb") as f:
            f.seek(-max_chars, 2)
            raw = f.read().decode("utf-8", errors="replace")
        partial = True
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        partial = False

    lines = raw.splitlines()
    n_error = sum(1 for ln in lines if "[ERROR]" in ln)
    n_warn = sum(1 for ln in lines if "[WARN]" in ln)
    n_api = sum(1 for ln in lines if "api/v1" in ln)
    n_embed = sum(1 for ln in lines if "embeddings" in ln or "/v1/embeddings" in ln)
    gpu_offload = raw.count("without support for GPU offload")
    context_exceeded = raw.count("Context size has been exceeded")
    unrecognized = raw.count("Unrecognized key")
    load_times = [float(x) for x in re.findall(r'"model_load_time_seconds"\s*:\s*([\d.]+)', raw)]

    # extract error messages
    err_msgs: list[str] = []
    kinds: Counter[str] = Counter()
    for m in re.finditer(
        r'\[ERROR\].*?"message"\s*:\s*"([^"]{10,200})"',
        raw,
        re.DOTALL,
    ):
        msg = m.group(1).replace("\\n", " ")[:200]
        err_msgs.append(msg)
        if "Context size" in msg:
            kinds["context_exceeded"] += 1
        elif "Unrecognized key" in msg:
            kinds["unrecognized_key"] += 1
        elif "previous_response" in msg:
            kinds["bad_prev_field"] += 1
        else:
            kinds["other"] += 1

    # recent unique
    recent = []
    seen = set()
    for msg in reversed(err_msgs):
        if msg in seen:
            continue
        seen.add(msg)
        recent.append(msg)
        if len(recent) >= tail_errors:
            break

    gates = []
    if context_exceeded > 0:
        gates.append({
            "signal": "context_exceeded",
            "count": context_exceeded,
            "gate": "NEED_INFO",
            "action": "cap pack size; disable response_id chain on large inputs; keep ctx≤4096",
        })
    if gpu_offload > 0:
        gates.append({
            "signal": "no_gpu_offload",
            "count": gpu_offload,
            "gate": "PASS",
            "action": "treat LMS as CPU-only on this ARM64 backend; do not plan GPU RAM savings",
        })
    if n_error > 50:
        gates.append({
            "signal": "error_flood",
            "count": n_error,
            "gate": "NEED_INFO",
            "action": "inspect server log; likely context thrash or bad request fields",
        })

    return {
        "ok": True,
        "path": str(path),
        "size_bytes": size,
        "partial_tail": partial,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "counts": {
            "ERROR_lines": n_error,
            "WARN_lines": n_warn,
            "api_v1_mentions": n_api,
            "embeddings_mentions": n_embed,
            "gpu_offload_warnings": gpu_offload,
            "context_exceeded": context_exceeded,
            "unrecognized_key": unrecognized,
        },
        "error_kinds": dict(kinds),
        "recent_errors": recent,
        "model_load_times_s": {
            "n": len(load_times),
            "max": max(load_times) if load_times else None,
            "mean": round(sum(load_times) / len(load_times), 2) if load_times else None,
        },
        "gates": gates,
        "gate": "NEED_INFO" if any(g["gate"] == "NEED_INFO" for g in gates) else "PASS",
    }


def read_prediction_history_summary(limit: int = 20) -> dict[str, Any]:
    idx = _read_json(HOME / ".internal" / "api-prediction-history" / "index.json")
    if not idx:
        return {"ok": False, "error": "no prediction history index"}
    mapping = (idx.get("json") or {}).get("mapping") or {}
    ids = list(mapping.keys())
    return {
        "ok": True,
        "n_indexed": len(ids),
        "sample_ids": ids[-limit:],
        "packs": sorted(set(mapping.values())),
        "note": "response_id tails map into pack files under api-prediction-history/packs/",
    }


# ---------------------------------------------------------------------------
# Policy derived from home (align Prime defaults)
# ---------------------------------------------------------------------------

def derived_policy() -> dict[str, Any]:
    """Hyper-optimized defaults from *this* LMS install, not generic docs."""
    settings = read_settings()
    http = read_http_server()
    hub = read_hub_models()
    logs = scan_server_log()
    backend = read_backend()
    mcp = read_mcp_json()

    lfm = next((m for m in hub.get("models") or [] if m.get("model") == "liquid/lfm2.5-1.2b"), {})
    bonsai = next((m for m in hub.get("models") or [] if m.get("model") == "prism-ml/bonsai-27b"), {})
    granite = next((m for m in hub.get("models") or [] if m.get("model") == "ibm/granite-4-h-tiny"), {})

    # LMS UI default is often 4096/8192 — that is NOT the production load target.
    # Production ctx comes from ctx_policy (RAM + model size). UI default only
    # informs pack budgets.
    ui_default_ctx = int(settings.get("default_context_length") or 4096)
    try:
        from ctx_policy import CTX_DAILY_SMALL, resolve_load_context

        daily = resolve_load_context("liquid/lfm2.5-1.2b", purpose="chat")
        default_ctx = int(daily.get("context_length") or CTX_DAILY_SMALL)
    except Exception:
        default_ctx = max(ui_default_ctx, 32768)
    # if logs show context thrash, prefer smaller effective budget for packs
    pack_budget_chars = 2800 if (logs.get("counts") or {}).get("context_exceeded", 0) > 0 else 4000
    chain_max_input_chars = 2200 if (logs.get("counts") or {}).get("context_exceeded", 0) > 0 else 3500

    return {
        "ok": True,
        "layer": "L0.5_LOCAL_HOME",
        "home": str(HOME),
        "base_url": http.get("base_url") or "http://127.0.0.1:1234",
        "port": http.get("port") or 1234,
        "default_context_length": default_ctx,
        "lms_ui_default_context_length": ui_default_ctx,
        "sampling_lfm": {
            "temperature": lfm.get("temperature", 0.1),
            "top_p": lfm.get("top_p", 0.1),
            "top_k": lfm.get("top_k", 50),
        },
        "min_memory_gb": {
            "lfm": lfm.get("min_memory_gb", 0.95),
            "bonsai": bonsai.get("min_memory_gb", 3.8),
            "granite": granite.get("min_memory_gb"),
        },
        "guardrails_gb": (settings.get("model_loading_guardrails") or {}).get("threshold_gb"),
        "cpu_only_engine": (logs.get("counts") or {}).get("gpu_offload_warnings", 0) > 0,
        "jit_loading": http.get("jit_model_loading"),
        "pack_budget_chars": pack_budget_chars,
        "chain_max_input_chars": chain_max_input_chars,
        "disable_chain_on_large_input": True,
        "mcp_plugins_configured": not mcp.get("empty"),
        "log_gate": logs.get("gate"),
        "log_signals": logs.get("gates") or [],
        "backend_preferred": backend.get("preferred"),
        "recommendations": [
            "Keep LFM + nomic only; Bonsai min ~3.8GB + guardrail 4GB = thrash risk",
            "Cap deep_loop packs under pack_budget_chars; logs show Context size exceeded",
            "Do not chain previous_response_id when input > chain_max_input_chars",
            "GPU offload unsupported on llama.cpp-win-arm64 — CPU residency is truth",
            "LMS mcp.json empty — use API ephemeral_mcp or Prime MCP, not LMS plugins",
            f"Do NOT load at LMS UI default ({ui_default_ctx}); use ctx_policy daily={default_ctx}",
            "8192 is tight for multi-role dual_enter — prefer ≥32k on 1–3B fibers",
            f"LFM hub sampling: temp={lfm.get('temperature', 0.1)} top_p={lfm.get('top_p', 0.1)}",
            "Jina aboutness is side-server :8765 (jina_service.ensure_jina), not LMS embeddings",
        ],
    }


def snapshot(include_log_scan: bool = True) -> dict[str, Any]:
    """Full local LMS truth for operators / MCP."""
    out: dict[str, Any] = {
        "ok": True,
        "layer": "L0.5_LOCAL_HOME",
        "home": str(HOME),
        "settings": read_settings(),
        "http_server": read_http_server(),
        "mcp": read_mcp_json(),
        "backend": read_backend(),
        "hub_models": read_hub_models(),
        "model_data": read_model_data(),
        "plugins": read_plugins(),
        "prediction_history": read_prediction_history_summary(),
        "policy": derived_policy(),
    }
    if include_log_scan:
        out["server_log"] = scan_server_log()
    # overall gate
    log_gate = (out.get("server_log") or {}).get("gate") or "PASS"
    out["gate"] = log_gate
    out["thesis"] = (
        "LMS capability on this kit is gated by local home: ARM64 CPU engine, "
        "RAM-aware ctx_policy (not UI 4096/8192), high load guardrails, empty "
        "mcp.json, and log-proven overflow risk when packs ignore char budgets."
    )
    return out


if __name__ == "__main__":
    import sys

    cmd = (sys.argv[1] if len(sys.argv) > 1 else "snapshot").lower()
    if cmd == "policy":
        print(json.dumps(derived_policy(), indent=2))
    elif cmd == "logs":
        print(json.dumps(scan_server_log(), indent=2))
    elif cmd == "settings":
        print(json.dumps(read_settings(), indent=2))
    else:
        # snapshot can be large — trim embeddings-style noise
        s = snapshot()
        print(json.dumps(s, indent=2)[:12000])
