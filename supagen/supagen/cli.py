#!/usr/bin/env python3
"""
supagen — super agent harness CLI

  supagen smoke          offline unit + harness packs
  supagen e2e            offline + live (if LMS up)
  supagen e2e --live     require LMS
  supagen ensure         jina + LMS fiber (ctx_policy)
  supagen status         jina + LMS + ctx snapshot
  supagen enter "..."    dual_enter (aboutness + NLI face)
  supagen aboutness      A/B/C null (jina default)
  supagen harness ...    proxy to multiplane pipeline/smoke
  supagen doctor         diagnose jina/LMS/ctx failures
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    from .bootstrap import main as boot

    return boot()


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .e2e import main as e2e_main

    return e2e_main(["--offline"])


def _cmd_e2e(args: argparse.Namespace) -> int:
    from .e2e import main as e2e_main

    argv = []
    if args.live:
        argv.append("--live")
    if args.offline:
        argv.append("--offline")
    if args.model:
        argv.extend(["--model", args.model])
    if args.json:
        argv.append("--json")
    return e2e_main(argv)


def _cmd_ensure(args: argparse.Namespace) -> int:
    from .ensure import ensure_all

    r = ensure_all(
        chat_model=args.model,
        purpose=args.purpose,
        jina=not args.no_jina,
        lms=not args.no_lms,
        fiber_mode=getattr(args, "mode", None),
    )
    print(json.dumps(r, indent=2, default=str))
    return 0 if r.get("ok") else 1


def _cmd_status(args: argparse.Namespace) -> int:
    from .paths import ensure_sys_path, workspace_root

    paths = ensure_sys_path()
    out: dict = {"paths": paths, "version": __import__("supagen").__version__}
    try:
        from jina_service import jina_status

        out["jina"] = jina_status()
    except Exception as e:
        out["jina"] = {"error": str(e)}
    try:
        from lms_layers import l0_health, l1_catalog, l1_free_ram_gb
        from ctx_policy import resolve_load_context

        out["lms"] = {
            "health": l0_health(),
            "free_gb": l1_free_ram_gb(),
            "loaded": [
                {
                    "key": m.get("key"),
                    "ctx": (m.get("loaded_instances") or [{}])[0]
                    .get("config", {})
                    .get("context_length")
                    if m.get("loaded_instances")
                    else None,
                    "max": m.get("max_context_length"),
                }
                for m in (l1_catalog().get("models") or [])
                if m.get("loaded")
            ],
            "policy_lfm": resolve_load_context("liquid/lfm2.5-1.2b"),
            "policy_ministral": resolve_load_context("mistralai/ministral-3-3b"),
        }
    except Exception as e:
        out["lms"] = {"error": str(e)}
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_enter(args: argparse.Namespace) -> int:
    from .paths import ensure_sys_path
    from .ensure import ensure_all

    ensure_sys_path()
    ensure_all(chat_model=args.model, jina=True, lms=True)
    from dual_enter import dual_enter

    r = dual_enter(args.prompt, retrieve_kb=not args.no_kb)
    print(json.dumps(r, indent=2, default=str)[:12000])
    return 0 if r.get("ok") is not False else 1


def _cmd_aboutness(args: argparse.Namespace) -> int:
    from .paths import ensure_sys_path, prime_scripts

    ensure_sys_path()
    from jina_service import ensure_jina

    ensure_jina()
    script = prime_scripts() / "null_aboutness.py"
    argv = [sys.executable, str(script)]
    if args.family:
        argv.extend(["--family", args.family])
    return subprocess.call(argv)


def _cmd_harness(args: argparse.Namespace) -> int:
    from .paths import harness_root

    hr = harness_root()
    rest = list(args.rest or [])
    if not rest or rest[0] in ("smoke", "offline"):
        return subprocess.call([sys.executable, str(hr / "smoke_offline.py")], cwd=str(hr))
    if rest[0] == "pipeline":
        return subprocess.call(
            [sys.executable, str(hr / "pipeline" / "v1" / "pipeline.py"), *rest[1:]],
            cwd=str(hr),
        )
    if rest[0] == "certify":
        return subprocess.call(
            [sys.executable, str(hr / "certify" / "v1" / "certify.py"), *rest[1:]],
            cwd=str(hr),
        )
    if rest[0] == "ingest":
        return subprocess.call(
            [sys.executable, str(hr / "ingest" / "v1" / "ingest.py"), *rest[1:]],
            cwd=str(hr),
        )
    print("usage: supagen harness [smoke|pipeline|certify|ingest] ...", file=sys.stderr)
    return 2


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose seamless stack failures with fixes."""
    from .paths import ensure_sys_path

    ensure_sys_path()
    issues: list[str] = []
    fixes: list[str] = []
    snap: dict = {}

    # import path health (buddy .pth)
    try:
        import importlib.util

        spec = importlib.util.find_spec("nomic_metric")
        snap["import_nomic_metric"] = bool(spec)
        if not spec:
            issues.append("nomic_metric not importable — bootstrap .pth missing")
            fixes.append("python -m supagen bootstrap")
    except Exception as e:
        issues.append(f"import probe: {e}")
        fixes.append("python -m supagen bootstrap")

    try:
        from jina_service import ensure_jina, jina_status, BASE, CTX

        st = jina_status()
        snap["jina_probe"] = st.get("probe")
        if not (st.get("probe") or {}).get("ok"):
            ej = ensure_jina()
            snap["jina_ensure"] = ej
            if not ej.get("ok"):
                issues.append(f"jina down: {ej.get('error') or ej.get('status')}")
                fixes.append("python -m supagen ensure --no-lms")
                fixes.append("Check prime/state/jina_embed.log ; set PRIME_JINA_GGUF=")
            else:
                fixes.append(f"jina started on {BASE} ctx={CTX}")
        else:
            snap["jina"] = "ok"
    except Exception as e:
        issues.append(f"jina module: {e}")
        fixes.append("pip install -e ./supagen && python -m supagen bootstrap")

    # KB family mismatch (nomic-era index vs live jina)
    try:
        from pathlib import Path
        import json as _json
        from nomic_metric import resolve_family
        from kb_index import default_out

        kb_path = default_out()
        if kb_path.is_file():
            kb = _json.loads(kb_path.read_text(encoding="utf-8"))
            live = resolve_family()
            stored = kb.get("embed_family")
            snap["kb"] = {
                "path": str(kb_path),
                "n": kb.get("n_chunks"),
                "stored_family": stored,
                "live_family": live,
            }
            if stored is None or stored != live:
                issues.append(
                    f"KB embed_family={stored!r} != live={live!r} "
                    f"(n={kb.get('n_chunks')}) — cosine retrieve is wrong until reindex"
                )
                fixes.append("python -m supagen reindex-kb")
                fixes.append(
                    "or next retrieve auto-reembeds if n_chunks<=120"
                )
    except Exception as e:
        snap["kb_error"] = str(e)

    try:
        from lms_layers import l0_health, l1_free_ram_gb
        from ctx_policy import resolve_load_context, loaded_context_for

        h = l0_health()
        snap["lms_health"] = h.get("ok")
        snap["free_gb"] = l1_free_ram_gb()
        if not h.get("ok"):
            issues.append("LM Studio :1234 unreachable")
            fixes.append("Start LM Studio → Developer → Local server")
        for key in ("liquid/lfm2.5-1.2b", "mistralai/ministral-3-3b"):
            pol = resolve_load_context(key)
            cur = loaded_context_for(key)
            snap[f"policy_{key}"] = pol.get("context_length")
            snap[f"loaded_ctx_{key}"] = cur
            if cur is not None and cur < int(pol["context_length"]) * 0.75:
                issues.append(
                    f"{key} loaded_ctx={cur} << policy={pol['context_length']}"
                )
                fixes.append(
                    f'supagen ensure --model "{key}"  # reloads to policy ctx when RAM allows'
                )
    except Exception as e:
        issues.append(f"lms: {e}")

    out = {"ok": len(issues) == 0, "issues": issues, "fixes": fixes, "snap": snap}
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    # Path bootstrap before any subcommand imports prime modules
    try:
        from .paths import ensure_sys_path

        ensure_sys_path()
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        prog="supagen",
        description="Super agent harness: Prime dual-metric + multiplane OPEN|STOP",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "bootstrap",
        help="write site-packages .pth so imports work without PYTHONPATH",
    )
    p.set_defaults(func=_cmd_bootstrap)

    p = sub.add_parser("smoke", help="offline smoke")
    p.set_defaults(func=_cmd_smoke)

    p = sub.add_parser("e2e", help="end-to-end (auto live if LMS up)")
    p.add_argument("--live", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_e2e)

    p = sub.add_parser("ensure", help="jina + LMS substrate (seamless)")
    p.add_argument("--model", default=None, help="chat model key")
    p.add_argument(
        "--mode",
        choices=("scout", "preserve"),
        default=None,
        help=(
            "SCOUT=small fiber; PRESERVE=frankenstein alone. "
            "When omitted: PRIME_FIBER_MODE env if set, else scout"
        ),
    )
    p.add_argument("--purpose", default="chat")
    p.add_argument("--no-jina", action="store_true")
    p.add_argument("--no-lms", action="store_true")
    p.set_defaults(func=_cmd_ensure)

    p = sub.add_parser("status", help="jina + LMS + ctx snapshot")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("enter", help="dual_enter prompt")
    p.add_argument("prompt")
    p.add_argument("--model", default=None)
    p.add_argument("--no-kb", action="store_true")
    p.set_defaults(func=_cmd_enter)

    p = sub.add_parser("aboutness", help="A/B/C null aboutness")
    p.add_argument("--family", choices=("jina", "nomic", "both"), default=None)
    p.set_defaults(func=_cmd_aboutness)

    p = sub.add_parser("harness", help="multiplane harness proxy")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_harness)

    p = sub.add_parser("doctor", help="diagnose + fix hints")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("contract", help="hard regression gates (buddy CI)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(
        func=lambda a: __import__("supagen.contract", fromlist=["main"]).main(
            (["--offline"] if a.offline else []) + (["--json"] if a.json else [])
        )
    )

    p = sub.add_parser(
        "serve",
        help="watchdog: keep jina aboutness up (loop; Ctrl+C stop)",
    )
    p.add_argument("--interval", type=float, default=15.0)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("reindex-kb", help="rebuild KB embeds with current Job1 family")
    p.add_argument("--path", default="", help="source file/dir (optional)")
    p.set_defaults(func=_cmd_reindex_kb)

    p = sub.add_parser("query", help="KB aboutness retrieve (jina + hybrid rerank)")
    p.add_argument("q", help="query string")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser(
        "verify",
        help="full package verify: offline contract+smoke + optional live",
    )
    p.add_argument("--live", action="store_true")
    p.set_defaults(func=_cmd_verify)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


def _cmd_serve(args: argparse.Namespace) -> int:
    import time

    from .paths import ensure_sys_path

    ensure_sys_path()
    from jina_service import ensure_jina, probe_jina

    print(f"jina watchdog interval={args.interval}s  Ctrl+C to stop", flush=True)
    while True:
        r = ensure_jina()
        p = probe_jina(timeout=2.0)
        print(
            f"[{time.strftime('%H:%M:%S')}] ensure={r.get('status')} "
            f"probe_ok={p.get('ok')} dim={p.get('dim')} "
            f"err={p.get('error') or ''}",
            flush=True,
        )
        try:
            time.sleep(max(3.0, float(args.interval)))
        except KeyboardInterrupt:
            print("stopped", flush=True)
            return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Buddy acceptance: offline always; live if --live or LMS up."""
    import subprocess

    steps = [
        [sys.executable, "-m", "supagen", "contract", "--offline"],
        [sys.executable, "-m", "supagen", "smoke"],
        [sys.executable, "-m", "supagen", "harness", "smoke"],
    ]
    if args.live:
        steps.append([sys.executable, "-m", "supagen", "contract"])
        steps.append([sys.executable, "-m", "supagen", "e2e", "--live"])
    rc = 0
    for cmd in steps:
        print(">", " ".join(cmd), flush=True)
        r = subprocess.call(cmd)
        if r != 0:
            rc = r
            print(f"FAIL exit={r}", flush=True)
        else:
            print("OK", flush=True)
    print(f"verify done rc={rc}", flush=True)
    return rc


def _cmd_query(args: argparse.Namespace) -> int:
    from .paths import ensure_sys_path

    ensure_sys_path()
    from jina_service import ensure_jina
    from dimensional_parse import load_index, retrieve, pack_for_lfm
    from kb_index import default_out

    ensure_jina()
    path = default_out()
    if not path.is_file():
        print(json.dumps({"ok": False, "error": f"no KB at {path}; run reindex-kb"}))
        return 1
    idx = load_index(path)
    hits = retrieve(idx, args.q, k=args.k)
    slim = [
        {
            "score": h.get("score"),
            "cos": h.get("score_cos"),
            "lex": h.get("score_lex"),
            "bm25": h.get("score_bm25"),
            "title": (h.get("title") or "")[:100],
            "id": h.get("id"),
            "tags": h.get("tags"),
            "method": h.get("method"),
        }
        for h in hits
    ]
    print(
        json.dumps(
            {
                "ok": True,
                "q": args.q,
                "family": idx.get("embed_family"),
                "n_chunks": idx.get("n_chunks"),
                "hits": slim,
                "pack_preview": pack_for_lfm(args.q, hits)[:800],
            },
            indent=2,
        )
    )
    return 0


def _cmd_reindex_kb(args: argparse.Namespace) -> int:
    from .paths import ensure_sys_path, workspace_root

    ensure_sys_path()
    from jina_service import ensure_jina

    ensure_jina()
    try:
        from kb_index import build_default_index, default_out
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"kb_index: {e}"}))
        return 1
    # Prefer rebuild helper if present
    try:
        if args.path:
            from pathlib import Path
            from dimensional_parse import build_index
            from pathlib import Path as P

            src = P(args.path)
            text = src.read_text(encoding="utf-8", errors="replace") if src.is_file() else ""
            if not text:
                print(json.dumps({"ok": False, "error": "empty path"}))
                return 1
            idx = build_index(text, embed=True)
            out = default_out()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(idx), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "path": str(out),
                        "n": idx.get("n_chunks"),
                        "family": idx.get("embed_family"),
                        "dim": idx.get("dim"),
                    },
                    indent=2,
                )
            )
            return 0
        r = build_default_index()
        print(json.dumps(r if isinstance(r, dict) else {"ok": True, "result": str(r)}, indent=2, default=str)[:4000])
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
