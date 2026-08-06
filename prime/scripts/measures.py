"""Measure backends for Prime — instruments under design law (never force OPEN)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 180) -> dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": (p.stdout or "")[-8000:],
            "stderr": (p.stderr or "")[-4000:],
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "cmd": cmd}
    except Exception as e:
        return {"ok": False, "error": str(e), "cmd": cmd}


def measure_smoke(workspace: str) -> dict[str, Any]:
    ws = Path(workspace)
    smoke = ws / "tests" / "test_smoke.py"
    if smoke.exists():
        r = _run([sys.executable, str(smoke)], cwd=str(ws), timeout=300)
        return {"mode": "smoke", "target": str(smoke), **r}
    # generic import check
    py = list(ws.glob("*.py"))[:5]
    return {
        "mode": "smoke",
        "ok": True,
        "note": "no tests/test_smoke.py; shallow listing only",
        "py_files": [p.name for p in py],
    }


def measure_rplc(workspace: str, domain: str = "frb", n: int = 64) -> dict[str, Any]:
    ws = Path(workspace)
    candidates = [ws / "rplc_sheaf.py"]
    try:
        from workspace import rplc_root

        candidates.append(rplc_root() / "rplc_sheaf.py")
    except Exception:
        pass
    script = next((c for c in candidates if c.exists()), None)
    if not script:
        return {"mode": "rplc", "ok": False, "error": "rplc_sheaf.py not found"}
    r = _run(
        [sys.executable, str(script), "--domain", domain, "--n", str(n), "--steps", "2", "--seed", "0"],
        cwd=str(script.parent),
        timeout=180,
    )
    opened = None
    verify_ok = None
    try:
        # stdout is JSON cert
        data = json.loads(r.get("stdout") or "{}")
        opened = (data.get("certificate") or {}).get("opened_steps")
        verify_ok = (data.get("verify") or {}).get("ok")
    except Exception:
        pass
    return {
        "mode": "rplc",
        "domain": domain,
        "n": n,
        "opened_steps": opened,
        "verify_ok": verify_ok,
        **r,
    }


def measure_eref(workspace: str) -> dict[str, Any]:
    """Sequence-prior E_ref sample if topology-sees-sequence present."""
    roots = [Path(workspace)]
    try:
        from workspace import workspace_root

        roots.append(workspace_root() / "topology-sees-sequence")
    except Exception:
        pass
    for root in roots:
        script = root / "scripts" / "eref_prior.py"
        derive = root / "derive.py"
        if script.exists() and derive.exists():
            r = _run([sys.executable, str(script)], cwd=str(root), timeout=600)
            return {"mode": "eref", "root": str(root), **r}
        if derive.exists():
            code = (
                "from derive import strain; "
                "print(strain('A'*12)['E'], strain('P'*12)['E'], "
                "strain('P'*12, omega_mode='cis_pro')['E'])"
            )
            r = _run([sys.executable, "-c", code], cwd=str(root), timeout=60)
            return {"mode": "eref_quick", "root": str(root), **r}
    return {"mode": "eref", "ok": False, "error": "topology-sees-sequence derive.py not found"}


def lm_list_models(base_url: str = "http://127.0.0.1:1234/v1") -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", [])]
        return {"ok": True, "models": ids, "base_url": base_url}
    except Exception as e:
        return {"ok": False, "error": str(e), "base_url": base_url}


def lm_chat(
    prompt: str,
    system: str = "You are a precise local scout. Be terse. Flag uncertainty.",
    model: str | None = None,
    base_url: str = "http://127.0.0.1:1234/v1",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> dict[str, Any]:
    models = lm_list_models(base_url)
    if not models.get("ok"):
        return {"mode": "lm_scout", "ok": False, "error": models.get("error"), "hint": "Start LM Studio server on :1234"}
    ids = models.get("models") or []
    # prefer small/fast for scouts unless specified
    if not model:
        for pref in ("liquid/lfm2.5-1.2b", "ibm/granite-4-h-tiny", "prism-ml/bonsai-27b"):
            if pref in ids:
                model = pref
                break
        model = model or (ids[0] if ids else None)
    if not model:
        return {"mode": "lm_scout", "ok": False, "error": "no models loaded in LM Studio"}

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return {
            "mode": "lm_scout",
            "ok": True,
            "model": model,
            "content": content,
            "usage": data.get("usage"),
        }
    except urllib.error.HTTPError as e:
        return {"mode": "lm_scout", "ok": False, "error": f"HTTP {e.code}: {e.read()[:500]}", "model": model}
    except Exception as e:
        return {"mode": "lm_scout", "ok": False, "error": str(e), "model": model}


def measure_projection(
    workspace: str,
    human_text: str,
    domains: str = "code,rplc,eref,field",
) -> dict[str, Any]:
    """Bilateral language projection across domains (glue measure)."""
    from language_projection import bilateral_measure

    doms = [d.strip() for d in (domains or "").split(",") if d.strip()]
    text = human_text or "OPEN STOP residue measure audit restrict certify"
    report = bilateral_measure(text, workspace, domains=doms or None)
    # ok means ran; openable is separate
    report["ok"] = True
    return report


def measure(
    mode: str,
    workspace: str,
    prompt: str = "",
    model: str | None = None,
    domain: str = "frb",
    lm_base: str = "http://127.0.0.1:1234/v1",
    domains: str = "code,rplc,eref,field",
    parallel: bool = True,
) -> dict[str, Any]:
    mode = (mode or "smoke").lower()
    if mode in ("smoke", "test"):
        return measure_smoke(workspace)
    if mode == "rplc":
        return measure_rplc(workspace, domain=domain)
    if mode in ("eref", "topology", "sequence_prior"):
        return measure_eref(workspace)
    if mode in ("lm", "lm_scout", "scout"):
        if not model:
            models = lm_list_models(lm_base)
            if models.get("ok"):
                from resource_plane import pick_lm_model

                model = pick_lm_model(models.get("models") or [], prefer_deep=False)
        return lm_chat(
            prompt or "Summarize workspace risk in 5 bullets.",
            model=model,
            base_url=lm_base,
        )
    if mode in ("lm_models", "models"):
        return {"mode": "lm_models", **lm_list_models(lm_base)}
    if mode in ("project", "projection", "language", "bilateral", "align"):
        return measure_projection(workspace, prompt, domains=domains)
    if mode in ("resource", "resources", "hw"):
        from resource_plane import plan_utilization

        return {"mode": "resource", **plan_utilization()}
    if mode in ("all", "parallel", "fanout"):
        from resource_plane import parallel_map, plan_utilization

        jobs = [
            ("smoke", lambda: measure_smoke(workspace)),
            ("rplc", lambda: measure_rplc(workspace, domain=domain)),
            ("projection", lambda: measure_projection(workspace, prompt, domains=domains)),
            ("lm_models", lambda: lm_list_models(lm_base)),
        ]
        if parallel:
            bundle = parallel_map(jobs)
            parts = bundle.get("results") or {}
            ok = bundle.get("ok", True) and all(
                (p or {}).get("ok", True) for p in parts.values() if isinstance(p, dict)
            )
            return {
                "mode": "all_parallel",
                "ok": ok,
                "parts": parts,
                "errors": bundle.get("errors"),
                "workers": bundle.get("workers"),
                "resource": bundle.get("resource"),
                "plan": plan_utilization().get("plan"),
            }
        parts = {n: fn() for n, fn in jobs}
        ok = all(p.get("ok") for p in parts.values() if isinstance(p, dict) and "ok" in p)
        return {"mode": "all", "ok": ok, "parts": parts}
    return {"mode": mode, "ok": False, "error": f"unknown measure mode: {mode}"}
