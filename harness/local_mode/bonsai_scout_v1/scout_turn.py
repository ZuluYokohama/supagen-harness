#!/usr/bin/env python3
"""One local scout turn → write draft notes; optional fixed bundle → certify.

Usage (LM Studio server must be up with Bonsai loaded):
  python harness/local_mode/bonsai_scout_v1/scout_turn.py
  python harness/local_mode/bonsai_scout_v1/scout_turn.py --certify-demo
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
OUT_DIR = HERE / "runs"


def get_models():
    req = urllib.request.Request(f"{BASE}/models", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("id", "") for m in data.get("data", [])]


def message_text(msg: dict) -> str:
    """Bonsai/thinking models often fill reasoning_content before content."""
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if content and reasoning:
        return f"{content}\n\n<!-- reasoning -->\n{reasoning}"
    return content or reasoning


def chat(
    model: str,
    user: str,
    system: str | None = None,
    timeout: float = 900.0,
    max_tokens: int = 1024,
) -> tuple[str, dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
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
    ap.add_argument(
        "--certify-demo",
        action="store_true",
        help="After scout, run external certify --demo filmore",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Short system prompt + smaller max_tokens (better on slow 1-bit)",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ids = get_models()
    except Exception as e:
        print(f"LMS server not reachable at {BASE}: {e}", file=sys.stderr)
        return 1

    model = next((i for i in ids if "onsai" in i.lower() or "bonsai" in i.lower()), None)
    if not model:
        model = ids[0] if ids else None
    if not model:
        print("No models loaded in LM Studio", file=sys.stderr)
        return 1

    if args.fast:
        system = (
            "You are LOCAL SCOUT. Explore only. Never certify or OPEN. "
            "Structure: planes | relations/deviation | DRAFT claim | residue | "
            "end with CERTIFY: NOT AUTHORIZED — external gate required"
        )
        user = (
            "Planes: tool_magpi LIVE (50 @415ft), wits_surface HOLD (RPM0), "
            "decoder_rt DEAD in burst, vision absent. "
            "Write the 5 sections briefly."
        )
        max_tokens = 768
    else:
        system = SYSTEM
        user = """You are scouting a multi-plane job pack (EXPLORE only).

Given this cover (treat as observed condition:states):
- plane tool_magpi: LIVE, evidence rih_magpi_onset.json (50 events @ ~415 ft)
- plane wits_surface: HOLD, evidence magpi_burst_zoom.json (RPM=0, SPP~0)
- plane decoder_rt: DEAD in burst window, evidence decoder_raw_probe.json (0 sessions; S0029 later)
- vision: absent

Tasks:
1) Planes observed
2) Expected process vs deviation (relations)
3) One strong DRAFT multi-plane claim (id + text) — do NOT say OPEN or certified
4) Residue
5) Final line exactly: CERTIFY: NOT AUTHORIZED — external gate required
"""
        max_tokens = 1536

    print(f"model={model} fast={args.fast}")
    print("scout thinking (may take several minutes on 1-bit 27B)...")
    try:
        reply, raw = chat(model, user, system=system, max_tokens=max_tokens, timeout=900.0)
    except Exception as e:
        print(f"chat failed: {e}", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"scout_{ts}.md"
    raw_path = OUT_DIR / f"scout_{ts}.raw.json"
    raw_path.write_text(json.dumps(raw, indent=2)[:200000], encoding="utf-8")
    out_path.write_text(
        f"# Scout turn {ts}\n\nmodel: `{model}`\n\n## Reply\n\n{reply}\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"wrote {raw_path}")
    print(f"reply_chars={len(reply)}")
    print("--- scout reply head ---")
    print(reply[:1500] if reply else "(empty — check raw.json / enable more max_tokens)")
    print("--- end head ---")

    if args.certify_demo:
        print("\n=== external certify (not the model) ===")
        rc = subprocess.call(
            [sys.executable, str(ROOT / "harness/certify/v1/certify.py"), "--demo", "filmore"],
            cwd=str(ROOT),
        )
        return rc if rc != 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
