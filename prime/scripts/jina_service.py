"""
Always-on jina aboutness server (Job 1).

LMS lists jina-embeddings-v5 as type=llm — /v1/embeddings cannot serve it.
This module owns a dedicated llama-server --embedding on PRIME_JINA_BASE
(:8765 by default) with fixed hyperparams and auto-restart.

"System prompt" for embeddings = task prefixes (not chat system):
  search_query     → Query: 
  search_document  → Document: 
  clustering/class → Document: 

Public API
----------
  ensure_jina()     → {ok, base, dim, started, ...}  always call before embed
  probe_jina()      → health
  jina_status()     → operator snapshot
  stop_jina()       → optional teardown

Hyperparams (env-overridable)
-----------------------------
  PRIME_JINA_HOST / PORT / BASE
  PRIME_JINA_CTX          default 8192 (model max)
  PRIME_JINA_PARALLEL     default 1
  PRIME_JINA_UBATCH       default 512
  PRIME_JINA_AUTO_START   default 1
  PRIME_JINA_GGUF / PRIME_LLAMA_SERVER
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_LAST_ENSURE: dict[str, Any] = {}

HOST = os.environ.get("PRIME_JINA_HOST", "127.0.0.1")
PORT = int(os.environ.get("PRIME_JINA_PORT", "8765"))
BASE = os.environ.get("PRIME_JINA_BASE", f"http://{HOST}:{PORT}").rstrip("/")
CTX = int(os.environ.get("PRIME_JINA_CTX", "8192"))
PARALLEL = int(os.environ.get("PRIME_JINA_PARALLEL", "1"))
UBATCH = int(os.environ.get("PRIME_JINA_UBATCH", "512"))
AUTO_START = os.environ.get("PRIME_JINA_AUTO_START", "1").strip() not in (
    "0",
    "false",
    "no",
)
READY_TIMEOUT = float(os.environ.get("PRIME_JINA_READY_TIMEOUT", "90"))

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
PID_FILE = STATE_DIR / "jina_embed.pid"
LOG_FILE = STATE_DIR / "jina_embed.log"
META_FILE = STATE_DIR / "jina_embed.meta.json"

# Prefix table — the retrieval "system" channel
PREFIX = {
    "search_query": "Query: ",
    "search_document": "Document: ",
    "clustering": "Document: ",
    "classification": "Document: ",
    "none": "",
}


def _gguf_candidates() -> list[Path]:
    """Prefer v5-small if present (Tier-B upgrade); else nano."""
    out: list[Path] = []
    if os.environ.get("PRIME_JINA_GGUF"):
        out.append(Path(os.environ["PRIME_JINA_GGUF"]))
    base = Path(
        r"C:\LM_STUDIO_MODELS\00.LLM HF MODELS4 CODING-RESEARCH-TESTING-USE-RESEARCH-TESTING-USE-1JUN26"
        r"\jinaai"
    )
    # Tier-B: small before nano (prefer Q4_K_M then higher quality quants)
    small_dir = base / "jina-embeddings-v5-text-small-retrieval"
    for name in (
        "v5-small-retrieval-Q4_K_M.gguf",
        "v5-small-retrieval-Q5_K_M.gguf",
        "v5-small-retrieval-Q8_0.gguf",
        "v5-small-retrieval-F16.gguf",
    ):
        out.append(small_dir / name)
    out.extend(sorted(small_dir.glob("*.gguf")))
    out.append(
        base
        / "jina-embeddings-v5-text-nano-retrieval"
        / "v5-nano-retrieval-F16.gguf"
    )
    # LMS downloadsFolder variants
    try:
        from lms_home import read_settings

        dl = (read_settings() or {}).get("downloads_folder")
        if dl:
            dlp = Path(dl) / "jinaai"
            for name in (
                "jina-embeddings-v5-text-small-retrieval",
                "jina-embeddings-v5-text-nano-retrieval",
            ):
                out.extend(sorted((dlp / name).glob("*.gguf")))
    except Exception:
        pass
    return out


def _server_candidates() -> list[Path]:
    out: list[Path] = []
    if os.environ.get("PRIME_LLAMA_SERVER"):
        out.append(Path(os.environ["PRIME_LLAMA_SERVER"]))
    home = Path.home() / ".lmstudio" / "extensions" / "backends"
    if home.is_dir():
        # newest llama.cpp win arm/x64 first
        for p in sorted(home.glob("llama.cpp-*/llama-server.exe"), reverse=True):
            out.append(p)
    return out


def _first_file(paths: list[Path]) -> Path | None:
    for p in paths:
        if p and p.is_file():
            return p
    return None


def apply_jina_prefix(text: str, task: str = "search_document") -> str:
    t = (text or "").strip()
    if not t:
        return t
    for p in PREFIX.values():
        if p and t.startswith(p):
            return t
    return PREFIX.get(task, PREFIX["search_document"]) + t


def probe_jina(base: str | None = None, timeout: float = 3.0) -> dict[str, Any]:
    b = (base or BASE).rstrip("/")
    url = b + "/v1/embeddings"
    body = json.dumps(
        {"model": "jina", "input": apply_jina_prefix("probe", "search_query")}
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        emb = (data.get("data") or [{}])[0].get("embedding") or []
        return {
            "ok": bool(emb),
            "base": b,
            "dim": len(emb),
            "model_field": data.get("model"),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "prefixes": PREFIX,
        }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:300]
        return {
            "ok": False,
            "base": b,
            "error": f"HTTP {e.code}: {err}",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        return {
            "ok": False,
            "base": b,
            "error": f"{type(e).__name__}: {e}",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        # Windows: tasklist fallback light
        if sys.platform == "win32":
            try:
                r = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return str(pid) in (r.stdout or "")
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def _write_meta(meta: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _build_cmd(gguf: Path, srv: Path) -> list[str]:
    # Hyperparams: embedding-only, full jina ctx, bounded batch for ARM stability
    cmd = [
        str(srv),
        "-m",
        str(gguf),
        "--embedding",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "-c",
        str(CTX),
        "-np",
        str(PARALLEL),
        "-ub",
        str(UBATCH),
        "-b",
        str(max(UBATCH, 512)),
        # jina-embeddings-v5: last-token pooling (mean collapses paraphrase ceiling)
        "--pooling",
        os.environ.get("PRIME_JINA_POOLING", "last"),
    ]
    # optional flash-attn if backend supports
    if os.environ.get("PRIME_JINA_FLASH", "0") in ("1", "true", "yes"):
        cmd += ["--flash-attn", "on"]
    return cmd


def _start_detached(gguf: Path, srv: Path) -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(gguf, srv)
    log_f = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
    log_f.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_f.write(" ".join(cmd) + "\n")
    log_f.flush()

    kwargs: dict[str, Any] = {
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    meta = {
        "pid": proc.pid,
        "cmd": cmd,
        "gguf": str(gguf),
        "server": str(srv),
        "base": BASE,
        "ctx": CTX,
        "ubatch": UBATCH,
        "parallel": PARALLEL,
        "started_at": time.time(),
        "log": str(LOG_FILE),
        "hyperparams": {
            "embedding": True,
            "context_length": CTX,
            "pooling": os.environ.get("PRIME_JINA_POOLING", "last"),
            "task_prefixes": PREFIX,
            "note": "Prefixes are the retrieval system channel; no chat system_prompt. Pooling=last for v5.",
        },
    }
    _write_meta(meta)
    return meta


def ensure_jina(
    *,
    force_restart: bool = False,
    timeout: float | None = None,
    auto_start: bool | None = None,
) -> dict[str, Any]:
    """
    Seamless ensure: probe → start if needed → wait ready.
    Safe to call on every embed path.

    Watchdog: if PID file says alive but probe fails → force restart.
    """
    global _LAST_ENSURE
    t_lim = timeout if timeout is not None else READY_TIMEOUT
    do_auto = AUTO_START if auto_start is None else auto_start

    with _LOCK:
        if not force_restart:
            p = probe_jina(timeout=2.5)
            if p.get("ok"):
                # Reconcile with live meta (actual GGUF/pooling that is running)
                live_meta: dict[str, Any] = {}
                if META_FILE.is_file():
                    try:
                        live_meta = json.loads(META_FILE.read_text(encoding="utf-8"))
                    except Exception:
                        live_meta = {}
                # Fail closed / force re-bind when config is not verified
                if not live_meta or not live_meta.get("gguf"):
                    force_restart = True
                else:
                    hyp = live_meta.get("hyperparams") or {}
                    want_pool = os.environ.get("PRIME_JINA_POOLING", "last")
                    live_pool = hyp.get("pooling")
                    # Desired GGUF: explicit env, else first candidate on disk
                    want_gguf = (os.environ.get("PRIME_JINA_GGUF") or "").strip()
                    if not want_gguf:
                        try:
                            cand = _first_file(_gguf_candidates())
                            want_gguf = str(cand) if cand else ""
                        except Exception:
                            want_gguf = ""
                    live_gguf = str(live_meta.get("gguf") or "")
                    config_mismatch = False
                    # Unverified pooling in meta → fail closed to restart
                    if not live_pool:
                        config_mismatch = True
                    elif live_pool != want_pool:
                        config_mismatch = True
                    if want_gguf and live_gguf:
                        # Identity: same resolved file (or same size+mtime fallback),
                        # never basename-only or substring path match.
                        try:
                            wp = Path(want_gguf).expanduser().resolve()
                            lp = Path(live_gguf).expanduser().resolve()
                            same = False
                            if wp.is_file() and lp.is_file():
                                try:
                                    same = wp.samefile(lp)
                                except OSError:
                                    same = False
                                if not same:
                                    same = (
                                        wp.stat().st_size == lp.stat().st_size
                                        and abs(wp.stat().st_mtime - lp.stat().st_mtime)
                                        < 1.0
                                        and wp.name.lower() == lp.name.lower()
                                    )
                            else:
                                same = str(wp).lower() == str(lp).lower()
                            if not same:
                                config_mismatch = True
                        except Exception:
                            config_mismatch = True
                    if config_mismatch:
                        force_restart = True
                    else:
                        out = {
                            "ok": True,
                            "started": False,
                            "status": "already_running",
                            "base": BASE,
                            "dim": p.get("dim"),
                            "latency_ms": p.get("latency_ms"),
                            "gguf": live_gguf,
                            "config_verified": True,
                            "model_field": p.get("model_field"),
                            "hyperparams": {
                                "context_length": live_meta.get("ctx")
                                or hyp.get("context_length")
                                or CTX,
                                "pooling": live_pool,
                                "prefixes": PREFIX,
                            },
                            "seamless": True,
                        }
                        _LAST_ENSURE = out
                        return out
            # zombie: pid alive, port dead → force restart
            if PID_FILE.is_file():
                try:
                    zpid = int(PID_FILE.read_text().strip())
                    if _pid_alive(zpid):
                        force_restart = True
                except Exception:
                    pass

        if not do_auto:
            out = {
                "ok": False,
                "started": False,
                "status": "down_auto_start_disabled",
                "base": BASE,
                "error": "jina embed server down; set PRIME_JINA_AUTO_START=1",
                "seamless": False,
            }
            _LAST_ENSURE = out
            return out

        # kill stale pid if force
        if force_restart and PID_FILE.is_file():
            try:
                old = int(PID_FILE.read_text().strip())
                if _pid_alive(old):
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/PID", str(old), "/F"],
                            capture_output=True,
                            timeout=15,
                        )
                    else:
                        os.kill(old, 9)
            except Exception:
                pass

        gguf = _first_file(_gguf_candidates())
        srv = _first_file(_server_candidates())
        if not gguf:
            out = {
                "ok": False,
                "status": "gguf_missing",
                "error": "jina GGUF not found; set PRIME_JINA_GGUF",
                "candidates": [str(p) for p in _gguf_candidates()],
                "seamless": False,
            }
            _LAST_ENSURE = out
            return out
        if not srv:
            out = {
                "ok": False,
                "status": "llama_server_missing",
                "error": "llama-server.exe not found; set PRIME_LLAMA_SERVER",
                "seamless": False,
            }
            _LAST_ENSURE = out
            return out

        # if pid alive but probe failed, still try start (port conflict handled below)
        try:
            meta = _start_detached(gguf, srv)
        except Exception as e:
            out = {
                "ok": False,
                "status": "start_failed",
                "error": str(e),
                "seamless": False,
            }
            _LAST_ENSURE = out
            return out

        deadline = time.time() + t_lim
        last_probe: dict[str, Any] = {}
        while time.time() < deadline:
            time.sleep(0.75)
            last_probe = probe_jina(timeout=2.0)
            if last_probe.get("ok"):
                out = {
                    "ok": True,
                    "started": True,
                    "status": "started_ready",
                    "base": BASE,
                    "dim": last_probe.get("dim"),
                    "pid": meta.get("pid"),
                    "latency_ms": last_probe.get("latency_ms"),
                    "cmd": meta.get("cmd"),
                    "log": str(LOG_FILE),
                    "hyperparams": meta.get("hyperparams"),
                    "seamless": True,
                }
                _LAST_ENSURE = out
                return out

        out = {
            "ok": False,
            "started": True,
            "status": "started_not_ready",
            "base": BASE,
            "pid": meta.get("pid"),
            "error": last_probe.get("error") or f"not ready within {t_lim}s",
            "log": str(LOG_FILE),
            "cmd": meta.get("cmd"),
            "seamless": False,
            "hint": "Check state/jina_embed.log; port conflict or OOM",
        }
        _LAST_ENSURE = out
        return out


def jina_status() -> dict[str, Any]:
    p = probe_jina()
    meta = {}
    if META_FILE.is_file():
        try:
            meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    pid = None
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text().strip())
        except Exception:
            pid = None
    return {
        "probe": p,
        "base": BASE,
        "pid": pid,
        "pid_alive": _pid_alive(pid) if pid else False,
        "meta": meta,
        "last_ensure": _LAST_ENSURE,
        "auto_start": AUTO_START,
        "ctx": CTX,
        "prefixes": PREFIX,
        "lms_note": (
            "jina is type=llm in LMS — never use LMS /v1/embeddings for jina"
        ),
    }


def stop_jina() -> dict[str, Any]:
    if not PID_FILE.is_file():
        return {"ok": True, "status": "no_pid_file"}
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=15
            )
        else:
            os.kill(pid, 9)
        return {"ok": True, "status": "killed", "pid": pid}
    except Exception as e:
        return {"ok": False, "error": str(e), "pid": pid}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Jina aboutness service")
    ap.add_argument("cmd", nargs="?", default="ensure", choices=("ensure", "status", "stop", "probe"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "ensure":
        print(json.dumps(ensure_jina(force_restart=a.force), indent=2))
    elif a.cmd == "status":
        print(json.dumps(jina_status(), indent=2))
    elif a.cmd == "stop":
        print(json.dumps(stop_jina(), indent=2))
    else:
        print(json.dumps(probe_jina(), indent=2))
