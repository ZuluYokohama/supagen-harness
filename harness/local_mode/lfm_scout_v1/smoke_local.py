#!/usr/bin/env python3
"""Smoke-test LM Studio for lfm_scout_v1 (fast 1.2B scout).

  python harness/local_mode/lfm_scout_v1/smoke_local.py

Load LFM2.5-1.2B-Instruct in LM Studio; server :1234.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESET = json.loads((HERE / "preset.json").read_text(encoding="utf-8"))
SYSTEM = (HERE / PRESET["system_prompt_file"]).read_text(encoding="utf-8")
BASE = PRESET["lm_studio"]["api_base"].rstrip("/")
HINTS = [h.lower() for h in PRESET["model"].get("lms_id_hints", ["lfm"])]


def get(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_model(ids: list[str]) -> str | None:
    for i in ids:
        low = i.lower()
        if any(h in low for h in HINTS):
            return i
    return ids[0] if ids else None


def post_chat(model: str, user: str, timeout: float = 120.0, max_tokens: int = 256):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": PRESET["sampling_regimes"]["scout_default"]["temperature"],
        "top_p": PRESET["sampling_regimes"]["scout_default"]["top_p"],
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    report = {"preset_id": PRESET["preset_id"], "api_base": BASE, "checks": []}
    ok = True

    def check(name: str, passed: bool, detail="") -> None:
        nonlocal ok
        report["checks"].append({"check": name, "pass": passed, "detail": detail})
        if not passed:
            ok = False

    gguf = Path(PRESET["model"]["gguf_path"])
    check("gguf_on_disk", gguf.is_file(), str(gguf))

    try:
        models = get(f"{BASE}/models")
        ids = [m.get("id", "") for m in models.get("data", [])] if isinstance(models, dict) else []
        check("lm_studio_server_up", True, f"models={ids[:8]}")
        report["models"] = ids
        model_id = pick_model(ids)
        check("model_id_resolved", bool(model_id), str(model_id))
        if model_id:
            low = model_id.lower()
            check(
                "model_looks_like_lfm",
                any(h in low for h in HINTS),
                "load LFM in LMS if this fails (may have picked another loaded model)",
            )
    except Exception as e:
        check(
            "lm_studio_server_up",
            False,
            f"{type(e).__name__}: {e}. Start LM Studio with LFM loaded on :1234",
        )
        model_id = None

    if model_id:
        try:
            user = (
                "SMOKE ONLY. Under 80 words. "
                "Planes: tool_dump present, acq_emz present, vision absent. "
                "List planes + one DRAFT claim id. End with: CERTIFY: NOT AUTHORIZED"
            )
            out = post_chat(model_id, user, timeout=120.0, max_tokens=300)
            msg = out.get("choices", [{}])[0].get("message", {}) or {}
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            text = content or reasoning
            report["sample_reply_head"] = (text or "")[:500]
            check("chat_completions", bool(text), f"chars={len(text or '')}")
            low = (text or "").lower()
            bad = any(
                p in low
                for p in (
                    "certified open",
                    "i certify this is open",
                    "production ready to ship",
                )
            )
            check("no_false_certify_language", not bad, "scout stayed explore-ish")
        except Exception as e:
            check("chat_completions", False, f"{type(e).__name__}: {e}")

    report["ok"] = ok
    print(json.dumps(report, indent=2))
    if ok:
        print("LFM SCOUT SMOKE OK")
        return 0
    print("LFM SCOUT SMOKE INCOMPLETE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
