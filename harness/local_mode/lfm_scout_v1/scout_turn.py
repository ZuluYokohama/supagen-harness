#!/usr/bin/env python3
"""LFM2.5 fast scout turn → runs/; optional external certify demo.

  python harness/local_mode/lfm_scout_v1/scout_turn.py
  python harness/local_mode/lfm_scout_v1/scout_turn.py --certify-demo

Load liquid/lfm2.5-1.2b (or LFM GGUF) in LM Studio; server :1234.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRESET = json.loads((HERE / "preset.json").read_text(encoding="utf-8"))
SYSTEM = (HERE / PRESET["system_prompt_file"]).read_text(encoding="utf-8")
BASE = PRESET["lm_studio"]["api_base"].rstrip("/")
HINTS = [h.lower() for h in PRESET["model"].get("lms_id_hints", ["lfm"])]
OUT_DIR = HERE / "runs"


def get_models():
    req = urllib.request.Request(f"{BASE}/models", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("id", "") for m in data.get("data", [])]


def pick_model(ids: list[str]) -> str | None:
    for i in ids:
        low = i.lower()
        if any(h in low for h in HINTS):
            return i
    return ids[0] if ids else None


def message_text(msg: dict) -> str:
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if content and reasoning:
        return f"{content}\n\n<!-- reasoning -->\n{reasoning}"
    return content or reasoning


def chat(model: str, user: str, system: str, timeout: float = 180.0, max_tokens: int = 1024):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": PRESET["sampling_regimes"]["scout_default"]["temperature"],
        "top_p": PRESET["sampling_regimes"]["scout_default"]["top_p"],
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out.get("choices", [{}])[0].get("message", {}) or {}
    return message_text(msg), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify-demo", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Seamless substrate: prefer supagen/prime ensure (ctx_policy + jina) when available
    try:
        prime_scripts = ROOT / "prime" / "scripts"
        if prime_scripts.is_dir() and str(prime_scripts) not in sys.path:
            sys.path.insert(0, str(prime_scripts))
        from lm_studio_client import LMStudio  # type: ignore

        want = None
        for h in HINTS:
            # prefer liquid LFM key if present in catalog after ensure
            pass
        lm = LMStudio(BASE if BASE.endswith("1234") else "http://127.0.0.1:1234")
        ens = lm.ensure_loaded("liquid/lfm2.5-1.2b", purpose="chat")
        print(f"ensure_loaded: ok={ens.get('ok')} action={ens.get('action')} "
              f"ctx={ens.get('context_length') or (ens.get('ctx_policy') or {}).get('context_length')}")
    except Exception as e:
        print(f"prime ensure skipped: {e}")

    try:
        ids = get_models()
    except Exception as e:
        print(f"LMS not reachable at {BASE}: {e}", file=sys.stderr)
        return 1

    model = pick_model(ids)
    if not model:
        print("No models loaded — run: supagen ensure", file=sys.stderr)
        return 1

    user = """You are scouting a multi-plane job pack (EXPLORE only).

Cover (condition:states):
- tool_magpi: LIVE, 50 events @ ~415 ft (rih_magpi_onset.json)
- wits_surface: HOLD, RPM=0 SPP~0 (magpi_burst_zoom.json)
- decoder_rt: DEAD in burst; S0029 re-acquire later (decoder_raw_probe.json)
- vision: absent

Write:
1) Planes observed
2) Expected process vs deviation (relations)
3) One DRAFT multi-plane claim (id + text) — never say OPEN or certified
4) Residue
5) Final line exactly: CERTIFY: NOT AUTHORIZED — external gate required
"""
    print(f"model={model}")
    try:
        reply, raw = chat(model, user, SYSTEM, timeout=180.0, max_tokens=900)
    except Exception as e:
        print(f"chat failed: {e}", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"scout_{ts}.md"
    raw_path = OUT_DIR / f"scout_{ts}.raw.json"
    raw_path.write_text(json.dumps(raw, indent=2)[:200000], encoding="utf-8")
    out_path.write_text(
        f"# LFM scout turn {ts}\n\nmodel: `{model}`\n\n## Reply\n\n{reply}\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"reply_chars={len(reply)}")
    print("--- scout reply head ---")
    print(reply[:1500] if reply else "(empty)")
    print("--- end head ---")

    if args.certify_demo:
        print("\n=== external certify (not the model) ===")
        return subprocess.call(
            [sys.executable, str(ROOT / "harness/certify/v1/certify.py"), "--demo", "filmore"],
            cwd=str(ROOT),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
