"""End-to-end smoke for supagen (offline always; live if LMS up)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .paths import ensure_sys_path, harness_root, prime_scripts, workspace_root


@dataclass
class Step:
    name: str
    ok: bool
    detail: str = ""
    ms: float = 0.0
    required: bool = True


@dataclass
class Report:
    ok: bool = True
    steps: list[Step] = field(default_factory=list)
    mode: str = "offline"
    paths: dict[str, str] = field(default_factory=dict)

    def add(self, step: Step) -> None:
        self.steps.append(step)
        if step.required and not step.ok:
            self.ok = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "paths": self.paths,
            "steps": [
                {
                    "name": s.name,
                    "ok": s.ok,
                    "detail": s.detail,
                    "ms": round(s.ms, 1),
                    "required": s.required,
                }
                for s in self.steps
            ],
            "n_ok": sum(1 for s in self.steps if s.ok),
            "n_fail": sum(1 for s in self.steps if not s.ok and s.required),
        }


def _run(name: str, fn: Callable[[], tuple[bool, str]], required: bool = True) -> Step:
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    return Step(name=name, ok=ok, detail=detail[:500], ms=(time.perf_counter() - t0) * 1000, required=required)


def _lms_up(base: str = "http://127.0.0.1:1234") -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def run_e2e(*, live: bool | None = None, chat_model: str | None = None) -> Report:
    """
    live=None → auto (run live steps if LMS up)
    live=True → require LMS
    live=False → offline only
    """
    paths = ensure_sys_path()
    rep = Report(paths=paths)
    lms = _lms_up()
    if live is None:
        live = lms
    rep.mode = "live" if live else "offline"

    # --- offline imports / unit ---
    def t_imports():
        from ctx_policy import resolve_load_context  # noqa: F401
        from jina_service import apply_jina_prefix, probe_jina  # noqa: F401
        from nomic_metric import aboutness, apply_prefix, resolve_family  # noqa: F401
        from dual_enter import dual_enter  # noqa: F401
        from metric_text import strip_envelope  # noqa: F401

        fam = resolve_family()
        return True, f"family_default={fam}"

    rep.add(_run("imports_prime", t_imports))

    def t_ctx_policy():
        from ctx_policy import resolve_load_context

        a = resolve_load_context("liquid/lfm2.5-1.2b", free_gb=8.0)
        b = resolve_load_context("mistralai/ministral-3-3b", free_gb=8.0)
        c = resolve_load_context("frankenstein-2.0-i1", free_gb=8.0)
        ok = (
            a["context_length"] >= 32000
            and b["context_length"] >= 16000
            and c["context_length"] >= 8000
        )
        return ok, (
            f"lfm@{a['context_length']} ministral@{b['context_length']} "
            f"frank@{c['context_length']} (free=8 sim)"
        )

    rep.add(_run("ctx_policy_tiers", t_ctx_policy))

    def t_prefixes():
        from nomic_metric import apply_prefix

        q = apply_prefix("hello", "search_query", family="jina")
        d = apply_prefix("world", "search_document", family="jina")
        ok = q.startswith("Query:") and d.startswith("Document:")
        return ok, f"q={q!r} d={d!r}"

    rep.add(_run("jina_prefixes", t_prefixes))

    def t_metric_strip():
        from metric_text import strip_envelope

        s = strip_envelope({"verdict": "OPEN", "reason": "E_ref opens after audit"})
        return "E_ref" in s and "OPEN" not in s, s[:80]

    rep.add(_run("strip_envelope", t_metric_strip))

    # --- harness offline ---
    def t_harness_offline():
        import subprocess

        hr = harness_root()
        smoke = hr / "smoke_offline.py"
        if not smoke.is_file():
            return False, f"missing {smoke}"
        p = subprocess.run(
            [sys.executable, str(smoke)],
            cwd=str(hr),
            capture_output=True,
            text=True,
            timeout=300,
        )
        tail = (p.stdout or "")[-400:] + (p.stderr or "")[-200:]
        return p.returncode == 0, f"exit={p.returncode} {tail[:300]}"

    rep.add(_run("harness_smoke_offline", t_harness_offline))

    def t_harness_pipeline_pack():
        import subprocess

        hr = harness_root()
        pipe = hr / "pipeline" / "v1" / "pipeline.py"
        if not pipe.is_file():
            return False, f"missing {pipe}"
        p = subprocess.run(
            [sys.executable, str(pipe), "--pack", "filmore_magpi"],
            cwd=str(hr),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return p.returncode == 0, f"exit={p.returncode}"

    rep.add(_run("harness_pipeline_filmore", t_harness_pipeline_pack))

    # --- live ---
    if not live:
        rep.add(
            Step(
                name="live_skipped",
                ok=True,
                detail="LMS not required (offline mode)",
                required=False,
            )
        )
        return rep

    if not lms:
        rep.add(
            Step(
                name="lms_required",
                ok=False,
                detail="live=True but LMS :1234 down",
                required=True,
            )
        )
        return rep

    def t_jina_ensure():
        from jina_service import ensure_jina

        r = ensure_jina()
        return bool(r.get("ok")), json.dumps(
            {k: r.get(k) for k in ("status", "base", "dim", "started", "error") if k in r or r.get(k) is not None}
        )

    rep.add(_run("jina_ensure", t_jina_ensure))

    def t_aboutness_floor():
        from nomic_metric import aboutness

        r = aboutness(
            "E_ref sheaf certificate opens after lambda1 check",
            "Carbonara uses guanciale, egg, pecorino, and black pepper",
        )
        cos = r.get("cosine")
        ok = bool(r.get("ok")) and cos is not None and float(cos) < 0.35
        return ok, f"family={r.get('family')} cos={cos} svc={r.get('jina_service')}"

    rep.add(_run("aboutness_pasta_floor", t_aboutness_floor))

    def t_aboutness_ceiling():
        from nomic_metric import aboutness

        r = aboutness(
            "E_ref meets production readiness criteria under measured audit",
            "Under measured audit, E_ref satisfies criteria for production readiness",
        )
        cos = r.get("cosine")
        ok = bool(r.get("ok")) and cos is not None and float(cos) > 0.70
        return ok, f"cos={cos}"

    rep.add(_run("aboutness_paraphrase_ceiling", t_aboutness_ceiling))

    def t_ensure_chat():
        from supagen.ensure import ensure_all

        r = ensure_all(chat_model=chat_model, jina=True, lms=True)
        lms = r.get("lms") or {}
        ctx = (
            lms.get("loaded_ctx")
            or lms.get("context_length")
            or (lms.get("ctx_policy") or {}).get("context_length")
        )
        return bool(r.get("ok")), json.dumps(
            {
                "jina": (r.get("jina") or {}).get("status"),
                "lms_ctx": ctx,
                "lms_ok": lms.get("ok"),
                "errors": r.get("errors"),
            }
        )

    rep.add(_run("ensure_substrate", t_ensure_chat))

    def t_dual_enter():
        from dual_enter import dual_enter

        r = dual_enter(
            "Smoke: can we measure aboutness without promoting OPEN from cosine?",
            retrieve_kb=False,
        )
        # dual_enter should never force OPEN from cosine alone
        face = r.get("cert_face") or r.get("face") or {}
        verdict = r.get("verdict") or face.get("verdict") or ""
        ok = r.get("ok") is not False
        # soft: structure present
        has = "agreement" in r or "aboutness" in r or "cert_face" in r or "face" in r or "roles" in r
        return ok and has, f"verdict={verdict} keys={list(r.keys())[:12]}"

    rep.add(_run("dual_enter", t_dual_enter, required=True))

    def t_fence_sanitize():
        from metric_text import parse_json_loose, strip_code_fences

        raw = '```json\n{"ok": true, "echo": "ping"}\n```'
        t = strip_code_fences(raw)
        obj = parse_json_loose(raw)
        return obj == {"ok": True, "echo": "ping"} and "```" not in t, t

    rep.add(_run("json_fence_sanitize", t_fence_sanitize))

    def t_token_budget():
        from metric_text import estimate_tokens, pack_to_token_budget

        big = "word " * 5000
        packed = pack_to_token_budget(big, max_tokens=200)
        return estimate_tokens(packed) <= 250, f"tokens≈{estimate_tokens(packed)}"

    rep.add(_run("token_budget", t_token_budget))

    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Supagen E2E smoke")
    ap.add_argument("--live", action="store_true", help="require LMS live steps")
    ap.add_argument("--offline", action="store_true", help="offline only")
    ap.add_argument("--model", default=None, help="chat model key for ensure")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    live: bool | None
    if a.offline:
        live = False
    elif a.live:
        live = True
    else:
        live = None
    rep = run_e2e(live=live, chat_model=a.model)
    d = rep.to_dict()
    if a.json:
        print(json.dumps(d, indent=2))
    else:
        print(f"supagen e2e mode={d['mode']} ok={d['ok']}")
        for s in d["steps"]:
            mark = "PASS" if s["ok"] else "FAIL"
            req = "" if s["required"] else " (optional)"
            print(f"  [{mark}]{req} {s['name']}  {s['ms']}ms  {s['detail'][:120]}")
        print(f"paths: {d['paths']}")
    # write report
    out = workspace_root() / "supagen" / "state" / "e2e_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
