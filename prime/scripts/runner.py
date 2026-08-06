#!/usr/bin/env python3
"""
Prime silent runner — human:Grok only surface.

Called by UserPromptSubmit hook (or manually):
  echo '{"prompt":"..."}' | python runner.py
  python runner.py --prompt "..."

Writes:
  - state/last_exchange.json  (machine)
  - state/last_exchange.md    (for Grok context injection)

Stdout JSON for hooks:
  {"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"..."}}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stable state under prime package (or PRIME_STATE_DIR)
DEFAULT_STATE = ROOT.parent / "state" / "runner"
STATE_DIR = Path(os.environ.get("PRIME_STATE_DIR", str(DEFAULT_STATE)))
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _read_stdin_event() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"prompt": raw.strip()}


def _extract_prompt(event: dict, cli_prompt: str) -> str:
    if cli_prompt:
        return cli_prompt.strip()
    for key in ("prompt", "userPrompt", "user_prompt", "text", "message"):
        v = event.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # nested
    for nest in ("input", "data", "payload"):
        d = event.get(nest)
        if isinstance(d, dict):
            for key in ("prompt", "userPrompt", "text"):
                v = d.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""


def _skip_prompt(prompt: str) -> bool:
    p = prompt.strip().lower()
    if not p:
        return True
    # pure slash builtins that don't need LFM
    if p in ("/status", "/prime status", "status", "/help", "/hooks", "/plugins"):
        return True
    if p.startswith("/prime status") or p.startswith("/prime halt"):
        return True
    return False


def run_exchange(prompt: str, cwd: str) -> dict:
    os.environ["PRIME_STATE_DIR"] = str(STATE_DIR)
    try:
        from workspace import workspace_root

        ws = cwd or str(workspace_root())
    except Exception:
        ws = cwd or str(Path.cwd())
    import importlib
    import session_store
    import mcp_server

    importlib.reload(session_store)
    importlib.reload(mcp_server)
    m = mcp_server
    m.STORE = session_store.SessionStore(STATE_DIR)

    if not m.STORE.s.get("session_id"):
        m.tool_meta_loop(prompt[:500] if prompt else "runner session")
        m.tool_restrict(
            goal=prompt[:800] if prompt else "development under design law",
            non_goals="force-OPEN; skip exchange; plugin shopping",
            success="exchange card; honest OPEN|STOP",
            constraints="human:grok only UX; LFM+embed runner; residue never forced",
        )
    else:
        # refresh intent each enter
        m.STORE.s["intent"] = prompt
        m.STORE.save()

    # Lightweight exchange: LFM ops + projection; domain measures optional (env)
    heavy = os.environ.get("PRIME_RUNNER_HEAVY", "0") == "1"
    card = m.tool_exchange(
        prompt=prompt,
        include_domain_measures=heavy,
        domains="code,rplc,eref,field",
    )
    return card if isinstance(card, dict) else {"ok": False, "error": "bad card"}


def format_context(card: dict, prompt: str, elapsed: float) -> str:
    lfm = card.get("lfm") or {}
    proj = card.get("projection") or {}
    best = proj.get("best_domain") or {}
    roles = lfm.get("roles") or {}
    face = card.get("cert_face") or {}
    agree = lfm.get("agreement") or {}
    op = card.get("operator_summary") or {}
    lines = [
        "## PRIME RUNNER CARD — dual enter (MEASURE only)",
        f"elapsed_s={elapsed:.1f} ok={card.get('ok')} mode={card.get('mode')}",
        f"cert_face=**{face.get('face') or op.get('face')}** closed={face.get('closed')}",
        f"NLI agreement: {agree.get('label')} conf={agree.get('confidence')} "
        f"agrees={agree.get('agrees')} (Job 2 — owns glue)",
        f"aboutness cos={lfm.get('mean_cosine') or op.get('aboutness_pair_cos')} "
        f"family={op.get('aboutness_family') or 'jina'} (Job1 diagnostic — never OPEN)",
        f"lfm_verdict={lfm.get('verdict')} fatal={lfm.get('fatal_flag')} "
        f"kb_hits={op.get('kb_hits') or lfm.get('retrieval_hits')} "
        f"kb_top={((op.get('kb_top') or {}).get('title') or '')[:50]!r} "
        f"fiber={op.get('fiber_model')}@{op.get('fiber_ctx')}",
        f"projection: interface_glue={proj.get('any_glue_ok')} frustrated={proj.get('all_frustrated')} "
        f"best={best.get('domain')}/{best.get('regime')} mean_align={proj.get('mean_align')}",
        f"domain_measures={card.get('domain_measures')}",
        "",
        "### LFM roles (local fiber)",
    ]
    for role in ("SCOUT", "FALSIFY", "GLUE", "VERDICT"):
        text = (roles.get(role) or "").replace("\n", " ").strip()
        if text:
            lines.append(f"- **{role}**: {text[:280]}")
    if agree.get("reason"):
        lines.extend(["", f"### NLI reason\n{(agree.get('reason') or '')[:300]}"])
    lines.extend(
        [
            "",
            "### Operator law (mandatory)",
            "- Dual metric: jina=aboutness (Job1), NLI=agreement (Job2), cert_face=candidate logogram.",
            "- Never OPEN from cosine or LFM alone. Cosine never promotes OPEN.",
            "- If cert_face=STOP or fatal=true or all_frustrated=true → no CODE.",
            "- Production OPEN only via audit + domain measures.",
            "- Residue never forced.",
            f"- User enter preview: {prompt[:200]!r}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Prime silent runner")
    ap.add_argument("--prompt", default="", help="User prompt (else stdin JSON)")
    ap.add_argument("--cwd", default="", help="Workspace cwd")
    ap.add_argument("--hook", action="store_true", help="Emit UserPromptSubmit additionalContext JSON")
    args = ap.parse_args(argv)

    event = _read_stdin_event() if not args.prompt else {}
    prompt = _extract_prompt(event, args.prompt)
    cwd = args.cwd or event.get("cwd") or event.get("workspaceRoot") or os.getcwd()

    if _skip_prompt(prompt):
        if args.hook or event:
            # empty allow — no context injection
            print(json.dumps({"decision": "allow"}))
        return 0

    t0 = time.time()
    try:
        card = run_exchange(prompt, str(cwd))
        err = None
    except Exception as e:
        card = {"ok": False, "error": str(e), "mode": "runner_fail"}
        err = str(e)
    elapsed = time.time() - t0

    # persist
    (STATE_DIR / "last_exchange.json").write_text(
        json.dumps({"prompt": prompt, "elapsed": elapsed, "card": card}, indent=2, default=str),
        encoding="utf-8",
    )
    md = format_context(card, prompt, elapsed) if card.get("ok") else (
        f"## PRIME RUNNER CARD\nFAIL ok=false error={err or card.get('error')}\n"
        "Continue without local LFM if LMS down; still obey OPEN|STOP law."
    )
    (STATE_DIR / "last_exchange.md").write_text(md, encoding="utf-8")

    if args.hook or event.get("hookEventName") or "sessionId" in event:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": md,
            }
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(md)

    return 0 if card.get("ok") else 0  # fail-open for hooks


if __name__ == "__main__":
    raise SystemExit(main())
