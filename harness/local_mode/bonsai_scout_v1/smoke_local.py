#!/usr/bin/env python3
"""Smoke-test LM Studio local server for bonsai_scout_v1.

Usage:
  python harness/local_mode/bonsai_scout_v1/smoke_local.py

Requires LM Studio: load Bonsai-27B-Q1_0.gguf, start local server (default :1234).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESET = json.loads((HERE / "preset.json").read_text(encoding="utf-8"))
SYSTEM = (HERE / PRESET["system_prompt_file"]).read_text(encoding="utf-8")
BASE = PRESET["lm_studio"]["api_base"].rstrip("/")


def get(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_chat(model: str, user: str, timeout: float = 600.0, max_tokens: int = 256):
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

    mmproj_part = gguf.parent / "downloading_mmproj-Bonsai-27B-BF16.gguf.part"
    mmproj_ready = list(gguf.parent.glob("mmproj*.gguf"))
    report["vision_plane"] = {
        "mmproj_ready": bool(mmproj_ready),
        "still_downloading_part": mmproj_part.is_file(),
        "files": [p.name for p in mmproj_ready],
    }

    try:
        models = get(f"{BASE}/models")
        ids = []
        if isinstance(models, dict) and "data" in models:
            ids = [m.get("id", "") for m in models["data"]]
        check("lm_studio_server_up", True, f"models={ids[:8]}")
        report["models"] = ids
        # pick a model id that looks like bonsai if present
        model_id = next((i for i in ids if "onsai" in i.lower() or "bonsai" in i.lower()), None)
        if not model_id and ids:
            model_id = ids[0]
        check("model_id_resolved", bool(model_id), str(model_id))
    except Exception as e:
        check(
            "lm_studio_server_up",
            False,
            f"{type(e).__name__}: {e}. Start LM Studio server with Bonsai loaded on :1234",
        )
        model_id = None

    if model_id:
        try:
            user = (
                "SMOKE ONLY. Reply in under 120 words. "
                "Planes: tool_dump present, acq_emz present, vision absent. "
                "List planes + one DRAFT claim id. End with: CERTIFY: NOT AUTHORIZED"
            )
            # First token on 1-bit 27B can be slow; allow long wall clock.
            out = post_chat(model_id, user, timeout=600.0, max_tokens=200)
            msg = out.get("choices", [{}])[0].get("message", {}) or {}
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            text = content or reasoning
            report["sample_reply_head"] = (text or "")[:500]
            report["used_reasoning_field"] = bool(reasoning) and not bool(content)
            check("chat_completions", bool(text), f"chars={len(text or '')}")
            low = (text or "").lower()
            # soft: scout should not claim certified open
            bad = any(
                p in low
                for p in (
                    "certified open",
                    "status: open",
                    "i certify",
                    "production ready to ship",
                )
            )
            check("no_false_certify_language", not bad, "scout stayed explore-ish")
        except Exception as e:
            check("chat_completions", False, f"{type(e).__name__}: {e}")

    report["ok"] = ok
    print(json.dumps(report, indent=2))
    if ok:
        print("LOCAL SCOUT SMOKE OK")
        return 0
    print("LOCAL SCOUT SMOKE INCOMPLETE — see checks (server may be off)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
