"""Seamless substrate bring-up: jina aboutness + LMS fiber + nomic fallback."""
from __future__ import annotations

from typing import Any

from .paths import ensure_sys_path


def ensure_all(
    *,
    chat_model: str | None = None,
    purpose: str = "chat",
    jina: bool = True,
    lms: bool = True,
    fiber_mode: str | None = None,
) -> dict[str, Any]:
    """
    One call → Job1 jina up, unload heavies, promote chat fiber to policy ctx.

    fiber_mode: scout | preserve | None (→ PRIME_FIBER_MODE / scout)
    Never uses LMS UI default 4096 as the intentional target.
    """
    import os

    paths = ensure_sys_path()
    mode = (fiber_mode or os.environ.get("PRIME_FIBER_MODE") or "scout").lower().strip()
    if mode not in ("scout", "preserve"):
        mode = "scout"
    if mode == "preserve" and purpose == "chat":
        purpose = "preserve"

    out: dict[str, Any] = {
        "ok": True,
        "paths": paths,
        "fiber_mode": mode,
        "jina": None,
        "lms": None,
        "ctx_policy": None,
        "errors": [],
    }

    # Prefer residency.seamless_substrate when both requested
    if jina and lms:
        try:
            from residency import seamless_substrate
            from lms_layers import DEFAULT_LFM

            r = seamless_substrate(
                chat_model=chat_model or DEFAULT_LFM,
                fiber_mode=mode,
            )
            out["jina"] = r.get("jina")
            out["lms"] = r.get("fiber")
            out["ctx_policy"] = (r.get("fiber") or {}).get("ctx_policy")
            out["nomic"] = r.get("nomic")
            out["errors"] = list(r.get("errors") or [])
            out["fiber_mode"] = r.get("fiber_mode") or mode
            out["ok"] = bool(r.get("ok"))
            return out
        except Exception as e:
            out["errors"].append(f"seamless:{e}")

    if jina:
        try:
            from jina_service import ensure_jina, jina_status

            ej = ensure_jina()
            out["jina"] = {
                "ok": ej.get("ok"),
                "status": ej.get("status"),
                "base": ej.get("base"),
                "dim": ej.get("dim"),
                "started": ej.get("started"),
                "hyperparams": ej.get("hyperparams"),
                "status_snap": jina_status().get("probe"),
            }
            if not ej.get("ok"):
                out["errors"].append(f"jina: {ej.get('error') or ej.get('status')}")
        except Exception as e:
            out["jina"] = {"ok": False, "error": str(e)}
            out["errors"].append(f"jina: {e}")

    if lms:
        try:
            from residency import promote_chat_fiber
            from lms_layers import DEFAULT_LFM, l0_health, l1_free_ram_gb

            model = chat_model or DEFAULT_LFM
            health = l0_health()
            if not health.get("ok"):
                out["lms"] = {
                    "ok": False,
                    "error": "LMS unreachable on :1234 — start LM Studio local server",
                    "health": health,
                }
                out["errors"].append("lms: unreachable")
            else:
                fiber = promote_chat_fiber(model, purpose=purpose)
                out["ctx_policy"] = fiber.get("ctx_policy")
                out["lms"] = fiber
                if not fiber.get("ok"):
                    out["errors"].append(
                        f"lms load: {(fiber.get('ensure') or {}).get('error')}"
                    )
                out["lms"]["free_gb"] = l1_free_ram_gb()
        except Exception as e:
            out["lms"] = {"ok": False, "error": str(e)}
            out["errors"].append(f"lms: {e}")

    if lms and jina:
        out["ok"] = bool((out.get("jina") or {}).get("ok")) and bool(
            (out.get("lms") or {}).get("ok")
        )
    elif jina:
        out["ok"] = bool((out.get("jina") or {}).get("ok"))
    elif lms:
        out["ok"] = bool((out.get("lms") or {}).get("ok"))
    return out
