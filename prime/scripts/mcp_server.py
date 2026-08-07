#!/usr/bin/env python3
"""
Prime Session MCP — dynamic OPEN|STOP companion for the whole development arc.

Tools:
  session.start / session.status
  restrict
  graph.plan / graph.advance / graph.show
  measure
  condition.pulse / condition.read
  audit
  claim.record
  cert.write
  ask.need / ask.answer
  lm.scout / lm.models
  meta.loop  (one-shot meta-meta→… planning helper)

Design law: restrict → measure → audit → OPEN|STOP. Residue never forced.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time  # noqa: E402

from condition_gate import pulse_condition, read_condition  # noqa: E402
from language_projection import (  # noqa: E402
    align_projections,
    extract_domain,
    extract_human,
    bilateral_measure,
)
from measures import measure as run_measure, lm_chat, lm_list_models  # noqa: E402
from session_store import SessionStore  # noqa: E402

STATE_DIR = Path(os.environ.get("PRIME_STATE_DIR", str(ROOT.parent / "state")))
STORE = SessionStore(STATE_DIR)


def _ok(d: dict[str, Any]) -> dict[str, Any]:
    d.setdefault("law", "restrict→measure→audit→OPEN|STOP")
    return d


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_session_start(workspace: str = "", intent: str = "", modes: str = "code,lm,rplc") -> dict:
    ws = workspace or os.environ.get("PRIME_WORKSPACE") or str(Path.cwd())
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    return _ok(STORE.start(ws, intent=intent, modes=mode_list))


def tool_session_status() -> dict:
    return _ok(STORE.status())


def tool_restrict(
    goal: str,
    non_goals: str = "",
    success: str = "",
    constraints: str = "",
) -> dict:
    ng = [x.strip() for x in non_goals.split(";") if x.strip()] or [
        "Force-OPEN without measures",
        "Claim discovery without controls",
    ]
    sc = [x.strip() for x in success.split(";") if x.strip()] or [
        "measures recorded",
        "audit OPEN or honest STOP",
        "certificate written",
    ]
    cs = [x.strip() for x in constraints.split(";") if x.strip()] or [
        "Residue never forced",
        "Plot compute graph before CODE",
        "Ask human when accuracy requires it",
    ]
    return _ok(STORE.set_restrict(goal, ng, sc, cs))


def tool_graph_plan(intent: str = "", modes: str = "code,lm,rplc") -> dict:
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    intent = intent or (STORE.s.get("intent") or "")
    return _ok(STORE.graph_plan(intent, modes=mode_list))


def tool_graph_advance(to: str, note: str = "") -> dict:
    payload = {"note": note} if note else None
    return _ok(STORE.graph_advance(to, payload))


def tool_graph_show() -> dict:
    g = STORE.s.get("graph") or {}
    return _ok(
        {
            "ok": True,
            "current": g.get("current"),
            "path": g.get("path"),
            "mermaid": g.get("mermaid"),
            "nodes": {k: v.get("status") for k, v in (g.get("nodes") or {}).items()},
            "adjacency": g.get("adjacency"),
        }
    )


def tool_measure(
    mode: str = "smoke",
    prompt: str = "",
    model: str = "",
    domain: str = "frb",
    domains: str = "code,rplc,eref,field",
) -> dict:
    ws = STORE.s.get("workspace") or str(Path.cwd())
    if not STORE.s.get("session_id"):
        return _ok({"ok": False, "error": "call session.start first"})
    # phase nudge
    if STORE.s.get("phase") in (None, "idle"):
        return _ok({"ok": False, "error": "session not started"})
    STORE.graph_advance("MEASURE", {"mode": mode})
    # default human text for projection: intent + restrict goal
    human_src = prompt or STORE.s.get("intent") or ""
    if STORE.s.get("restrict"):
        human_src = (human_src + " " + str((STORE.s["restrict"] or {}).get("goal") or "")).strip()
    report = run_measure(
        mode=mode,
        workspace=ws,
        prompt=human_src,
        model=model or None,
        domain=domain,
        lm_base=(STORE.s.get("lm") or {}).get("base_url", "http://127.0.0.1:1234/v1"),
        domains=domains,
    )
    if mode in ("project", "projection", "language", "bilateral", "align", "all"):
        STORE.record_alignment(report if mode != "all" else report.get("parts", {}).get("projection", report))
    STORE.add_measure(report)
    return _ok({"ok": bool(report.get("ok", True)), "report": report, "phase": STORE.s.get("phase")})


def tool_project_human(text: str = "") -> dict:
    """Project human-side language stalk (intent/restrict)."""
    raw = text or STORE.s.get("intent") or ""
    if STORE.s.get("restrict"):
        r = STORE.s["restrict"]
        raw = f"{raw}\n{r.get('goal','')}\n{' '.join(r.get('success_checks') or [])}\n{' '.join(r.get('constraints') or [])}"
    if not raw.strip():
        return _ok({"ok": False, "error": "no human language text; pass text or set intent/restrict"})
    STORE.graph_advance("PROJECT", {"side": "human"})
    proj = extract_human(raw)
    d = proj.to_dict()
    d["n_features"] = len(d.get("features") or [])
    STORE.record_projection("human", "language", d)
    STORE.add_measure({"mode": "project_human", "ok": True, "fingerprint": d.get("fingerprint"), "interface": d.get("interface"), "n_features": d["n_features"]})
    return _ok({"ok": True, "projection": d, "thesis": "human language is a projection, not the manifold"})


def tool_project_domain(domain: str = "code", text: str = "") -> dict:
    """Project domain-side language (code|rplc|eref|field|lm|custom)."""
    ws = STORE.s.get("workspace") or str(Path.cwd())
    STORE.graph_advance("PROJECT", {"side": "domain", "domain": domain})
    proj = extract_domain(domain, ws, text=text)
    d = proj.to_dict()
    STORE.record_projection("domain", proj.domain, d)
    STORE.add_measure({"mode": "project_domain", "ok": True, "domain": proj.domain, **{k: d[k] for k in ("interface", "fingerprint", "n_features") if k in d}})
    # n_features convenience
    d["n_features"] = len(d.get("features") or [])
    return _ok({"ok": True, "projection": d, "thesis": "domain language is a projection, not the manifold"})


def tool_project_align(domain: str = "") -> dict:
    """
    Align human ↔ domain projections on shared interface.
    If domain empty: bilateral across code,rplc,eref,field.
    """
    ws = STORE.s.get("workspace") or str(Path.cwd())
    human_d = (STORE.s.get("projections") or {}).get("human")
    if not human_d:
        # auto human from intent
        tool_project_human()
        human_d = (STORE.s.get("projections") or {}).get("human")
    if not human_d:
        return _ok({"ok": False, "error": "no human projection"})

    STORE.graph_advance("ALIGN", {"domain": domain or "multi"})
    if domain:
        doms = (STORE.s.get("projections") or {}).get("domains") or {}
        if domain not in doms:
            tool_project_domain(domain)
            doms = (STORE.s.get("projections") or {}).get("domains") or {}
        if domain not in doms:
            return _ok({"ok": False, "error": f"domain projection missing: {domain}"})
        from language_projection import Projection

        def _p(d: dict) -> Projection:
            return Projection(
                side=d.get("side") or "human",
                domain=d.get("domain") or "language",
                raw=d.get("raw") or "",
                features=list(d.get("features") or []),
                interface=list(d.get("interface") or []),
                fingerprint=d.get("fingerprint") or "",
                t=float(d.get("t") or time.time()),
                meta=dict(d.get("meta") or {}),
            )

        h = _p(human_d)
        dproj = _p(doms[domain])
        report = align_projections(h, dproj)
    else:
        # Prefer full human raw stalk (intent + restrict), not a truncated session intent only
        intent = (
            (human_d.get("raw") if human_d else None)
            or STORE.s.get("intent")
            or ""
        )
        report = bilateral_measure(intent, ws)

    STORE.record_alignment(report)
    STORE.add_measure(report)
    # early park if all frustrated
    if report.get("all_frustrated"):
        STORE.s.setdefault("residue", []).append(
            {"kind": "language_glue", "text": "all domain projections frustrated vs human language", "t": time.time()}
        )
        STORE.save()
    return _ok(report)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def tool_condition_pulse(artifact: str, context: str = "prime") -> dict:
    STORE.graph_advance("CONDITION", {"ctx": context})
    r = _jsonable(pulse_condition(artifact, context))
    STORE.add_measure({"mode": "condition", **{k: v for k, v in r.items() if k != "pulse_summary"}})
    if not r.get("kernel_member"):
        STORE.graph_block("CONDITION", str(r.get("obstruction") or "obstructed"))
    return _ok(r)


def tool_condition_read() -> dict:
    return _ok(read_condition())


def tool_audit(verdict: str, reasons: str = "") -> dict:
    reasons_l = [x.strip() for x in reasons.split(";") if x.strip()] or ["unspecified"]
    measures = STORE.s.get("measures") or []
    if verdict.upper() == "OPEN":
        fails = [m for m in measures if m.get("ok") is False]
        if fails:
            return _ok(
                STORE.add_audit(
                    "STOP",
                    reasons_l + [f"{len(fails)} measure(s) failed — cannot OPEN"],
                    {"failed_modes": [f.get("mode") for f in fails]},
                )
            )
        # condition obstruction blocks OPEN
        conds = [m for m in measures if m.get("mode") == "condition"]
        if conds:
            last_c = conds[-1]
            if last_c.get("kernel_member") is False or str(last_c.get("verdict", "")).upper() == "OBSTRUCTED":
                return _ok(
                    STORE.add_audit(
                        "STOP",
                        reasons_l + ["condition gate OBSTRUCTED — cannot OPEN"],
                        {"condition": last_c.get("verdict"), "energy": last_c.get("energy")},
                    )
                )
        # bilateral language: all-frustrated glue blocks OPEN
        aligns = STORE.s.get("alignments") or []
        if aligns:
            last_a = aligns[-1]
            if last_a.get("all_frustrated") is True:
                return _ok(
                    STORE.add_audit(
                        "STOP",
                        reasons_l + ["language projections frustrated on all domains — no glue section"],
                        {"mean_align": last_a.get("mean_align"), "domains": last_a.get("domains")},
                    )
                )
    return _ok(STORE.add_audit(verdict.upper(), reasons_l))


def tool_claim_record(status: str, text: str, evidence: str = "") -> dict:
    STORE.graph_advance("CLAIM", {"status": status})
    return _ok(STORE.claim(status, text, evidence))


def tool_cert_write(path: str = "") -> dict:
    return _ok(STORE.write_cert(path or None))


def tool_ask_need(question: str, why: str, options: str = "") -> dict:
    opts = [x.strip() for x in options.split(";") if x.strip()]
    return _ok(STORE.need_question(question, why, opts))


def tool_ask_answer(question_id: str, answer: str) -> dict:
    return _ok(STORE.answer_question(question_id, answer))


def tool_lm_models() -> dict:
    from lm_studio_client import LMStudio, resource_aware_roster

    base = (STORE.s.get("lm") or {}).get("base_url", "http://127.0.0.1:1234/v1")
    # native catalog preferred
    host = base.replace("/v1", "").rstrip("/")
    if host.endswith("1234/v1"):
        host = "http://127.0.0.1:1234"
    try:
        native = LMStudio(host if "://" in host else "http://127.0.0.1:1234").list_models_native()
        roster = resource_aware_roster("http://127.0.0.1:1234")
        return _ok({**native, "openai_compat": lm_list_models(base), "roster": roster.get("recommendation")})
    except Exception:
        return _ok(lm_list_models(base))


def tool_lm_scout(prompt: str, model: str = "", system: str = "") -> dict:
    base = (STORE.s.get("lm") or {}).get("base_url", "http://127.0.0.1:1234/v1")
    STORE.graph_advance("LM_SCOUT", {})
    r = lm_chat(
        prompt,
        system=system or "You are a local Prime scout. Precise, terse, flag uncertainty.",
        model=model or None,
        base_url=base,
    )
    STORE.add_measure(r)
    return _ok(r)


def tool_lm_load(model: str, context_length: int = 0) -> dict:
    from lm_studio_client import LMStudio

    # 0 → ctx_policy (32k daily / 128k LFM); do not force UI 4096
    r = LMStudio().load(model, context_length=context_length or None)
    return _ok(r)


def tool_lm_unload(instance_id: str) -> dict:
    from lm_studio_client import LMStudio

    r = LMStudio().unload(instance_id)
    return _ok(r)


def tool_lm_embed(text: str = "") -> dict:
    from lm_studio_client import LMStudio

    raw = text or STORE.s.get("intent") or ""
    if not raw:
        return _ok({"ok": False, "error": "no text"})
    r = LMStudio().embed(raw)
    # drop full vector from session measure by default (huge); keep dim + norm signal
    if r.get("ok") and r.get("embedding"):
        emb = r["embedding"]
        r = {
            **{k: v for k, v in r.items() if k != "embedding"},
            "preview": emb[:8],
            "l2": round(sum(x * x for x in emb) ** 0.5, 4),
        }
    STORE.add_measure({"mode": "lm_embed", **r})
    return _ok(r)


def tool_enter_projection(
    prompt: str = "",
    models: str = "",
    embed: bool = True,
    mode: str = "dual",
) -> dict:
    """
    Every enter → dual_enter (aboutness + NLI + cert face). Default mode=dual.
    MEASURE only — never OPEN authority. Cosine never promotes OPEN.
    """
    from lm_studio_client import enter_projection

    text = prompt or STORE.s.get("intent") or ""
    if STORE.s.get("restrict"):
        text = (text + "\n" + str(STORE.s["restrict"].get("goal") or "")).strip()
    if not text:
        return _ok({"ok": False, "error": "no prompt/intent"})
    STORE.graph_advance("LM_SCOUT", {"mode": mode or "dual"})
    model_list = [m.strip() for m in models.split(",") if m.strip()] or None
    r = enter_projection(
        text,
        models=model_list,
        embed=bool(embed),
        mode=mode or "dual",
    )
    # strip huge embed vectors if nested
    STORE.add_measure(
        {
            k: v
            for k, v in (r.items() if isinstance(r, dict) else [])
            if k not in ("embeddings",)
        }
        if isinstance(r, dict)
        else r
    )
    if r.get("ok") and r.get("outputs"):
        bits = []
        for k, o in r["outputs"].items():
            if not isinstance(o, dict) or not o.get("ok"):
                continue
            if o.get("parsed"):
                bits.append(f"{k}: {o['parsed']}")
            else:
                bits.append(f"{k}: {(o.get('content') or o.get('raw') or '')}")
        joined = "\n".join(bits)
        if joined:
            from language_projection import make_projection

            proj = make_projection("domain", "lfm_ops", joined)
            STORE.record_projection("domain", "lfm_ops", proj.to_dict())
    # store fiber + dual cert face
    rid = r.get("last_response_id") or (r.get("fiber") or {}).get("response_id")
    STORE.s.setdefault("lm", {})
    if rid:
        STORE.s["lm"]["last_response_id"] = rid
    if isinstance(r, dict):
        STORE.s["lm"]["cert_face"] = (r.get("cert_face") or {}).get("face")
        STORE.s["lm"]["nli_label"] = (r.get("agreement") or {}).get("label")
        STORE.s["lm"]["layered_verdict"] = r.get("verdict")
        STORE.s["lm"]["operator_summary"] = r.get("operator_summary")
    STORE.save()
    return _ok(r)


def tool_lms_ensure(context_length: int = 0) -> dict:
    """L1 residency: one LFM + embed; unload duplicate/deep thrash."""
    from ctx_policy import resolve_load_context
    from lms_layers import DEFAULT_LFM, l1_ensure_substrate
    from jina_service import ensure_jina

    jina = ensure_jina()
    if not context_length:
        context_length = int(
            resolve_load_context(DEFAULT_LFM, purpose="chat")["context_length"]
        )
    r = l1_ensure_substrate(context_length=context_length)
    r["jina_service"] = {
        "ok": jina.get("ok"),
        "status": jina.get("status"),
        "base": jina.get("base"),
    }
    r["context_length"] = context_length
    STORE.add_measure({"mode": "lms_ensure", **{k: v for k, v in r.items() if k != "actions"}})
    return _ok(r)


def tool_lms_layers(prompt: str = "", action: str = "matrix") -> dict:
    """
    LMS layered stack control surface.
    action: matrix|health|catalog|ensure|ops|home|policy|logs
    """
    from lms_layers import (
        l0_health,
        l1_catalog,
        l1_ensure_substrate,
        layer_matrix,
        layered_enter,
    )

    a = (action or "matrix").lower().strip()
    if a == "matrix":
        return _ok({"ok": True, **layer_matrix()})
    if a == "health":
        return _ok(l0_health())
    if a == "catalog":
        return _ok(l1_catalog())
    if a == "ensure":
        return tool_lms_ensure()
    if a in ("home", "local", "snapshot"):
        from lms_home import snapshot

        return _ok(snapshot(include_log_scan=True))
    if a == "policy":
        from lms_home import derived_policy

        return _ok(derived_policy())
    if a == "logs":
        from lms_home import scan_server_log

        return _ok(scan_server_log())
    if a in ("ops", "enter", "layered"):
        text = prompt or STORE.s.get("intent") or ""
        if not text:
            return _ok({"ok": False, "error": "prompt required for ops"})
        return tool_enter_projection(prompt=text, mode="lfm_ops", embed=True)
    return _ok({
        "ok": False,
        "error": f"unknown action {action}; use matrix|health|catalog|ensure|ops|home|policy|logs",
    })


def tool_meta_loop(intent: str) -> dict:
    """Plot meta-meta → meta → restrict skeleton for an intent (does not OPEN)."""
    from resource_plane import plan_utilization

    r_start = tool_session_start(
        workspace=STORE.s.get("workspace") or str(Path.cwd()),
        intent=intent,
        modes="code,lm,rplc,research,project",
    )
    g = tool_graph_plan(intent=intent, modes="code,lm,rplc,research,project")
    res = plan_utilization()
    return _ok(
        {
            "ok": True,
            "message": "META_META complete: graph plotted + resource plan. Next: exchange (LFM) → measure → audit.",
            "session": r_start,
            "graph": g.get("graph") or g,
            "resource_plan": res.get("plan"),
            "resource_snapshot": res.get("snapshot"),
            "next_nodes": ["RESTRICT", "PROJECT", "ALIGN", "LM_SCOUT", "MEASURE", "AUDIT"],
            "human_gate": "Call ask.need if any success criterion is ambiguous.",
            "mandatory": "On every user enter after start: call exchange (LFM does work).",
        }
    )


def tool_exchange(
    prompt: str = "",
    include_domain_measures: bool = True,
    domains: str = "code,rplc,eref,field",
) -> dict:
    """
    Full modality exchange — the missing continuous loop.

    Human language ↔ domain projections ↔ LFM orthogonal ops (SCOUT/FALSIFY/GLUE/VERDICT)
    ↔ optional smoke/rplc measures. One call. MEASURE only; never OPEN alone.

    Operator MUST call this on (almost) every user enter after session start.
    """
    if not STORE.s.get("session_id"):
        tool_meta_loop(prompt or "(exchange without prior session)")
    text = (prompt or STORE.s.get("intent") or "").strip()
    if STORE.s.get("restrict"):
        text = (text + "\nGOAL: " + str(STORE.s["restrict"].get("goal") or "")).strip()
    if not text:
        return _ok({"ok": False, "error": "no prompt — pass user enter text"})

    # 1) Dual enter: aboutness retrieve + LFM roles + NLI agreement + cert face
    lfm = tool_enter_projection(prompt=text, mode="dual", embed=True)

    # 2) Bilateral language projection (symbol/interface — separate from nomic)
    tool_project_human(text)
    for d in [x.strip() for x in domains.split(",") if x.strip()]:
        tool_project_domain(d)
    align = tool_project_align("")

    # 3) Domain instruments (CPU) — optional but default on
    domain_m = None
    if include_domain_measures:
        domain_m = tool_measure_parallel(
            prompt=text,
            domains=domains,
            include_lm_scout=False,
        )

    agree = lfm.get("agreement") if isinstance(lfm.get("agreement"), dict) else {}
    about = lfm.get("aboutness") if isinstance(lfm.get("aboutness"), dict) else {}
    face = lfm.get("cert_face") if isinstance(lfm.get("cert_face"), dict) else {}
    roles_out = {}
    for k, v in (lfm.get("outputs") or {}).items():
        if not isinstance(v, dict):
            continue
        bit = v.get("parsed") or v.get("raw") or v.get("content") or ""
        if isinstance(bit, dict):
            bit = json.dumps(bit, ensure_ascii=False)
        roles_out[k] = str(bit)[:320]

    card = {
        "ok": bool(lfm.get("ok", True)) and bool(align.get("ok", True)),
        "mode": "modality_exchange_dual",
        "thesis": (
            "Dual enter: jina aboutness + NLI agreement + LFM roles + domain stalks. "
            "Cosine never promotes OPEN. Cert face is measure only."
        ),
        "human_preview": text[:280],
        "cert_face": face,
        "operator_summary": lfm.get("operator_summary"),
        "lfm": {
            "ok": lfm.get("ok"),
            "mode": lfm.get("mode") or lfm.get("enter_mode"),
            "verdict": lfm.get("verdict"),
            "fatal_flag": lfm.get("fatal_flag"),
            "mean_cosine": about.get("mean_cosine") or (lfm.get("embeddings") or {}).get("mean_cosine"),
            "aboutness_not_agreement": True,
            "agreement": {
                "label": agree.get("label"),
                "confidence": agree.get("confidence"),
                "agrees": agree.get("agrees"),
                "gate": agree.get("gate"),
                "reason": (agree.get("reason") or "")[:200],
            },
            "roles": roles_out,
            "retrieval_hits": (lfm.get("retrieval") or {}).get("n_hits"),
        },
        "projection": {
            "mean_align": align.get("mean_align"),
            "any_glue_ok": align.get("any_glue_ok"),
            "all_frustrated": align.get("all_frustrated"),
            "best_domain": align.get("best_domain"),
            "note": "projection glue is interface/symbol — not nomic agreement",
        },
        "domain_measures": None
        if not domain_m
        else {
            "ok": domain_m.get("ok"),
            "workers": (domain_m.get("report") or {}).get("workers"),
            "parts": list(((domain_m.get("report") or {}).get("parts") or {}).keys()),
        },
        "not_open_authority": True,
        "next": (
            "Operator may CODE only if cert_face not STOP, not fatal, not all_frustrated; "
            "then MEASURE/AUDIT. Production OPEN only via audit with domain measures."
        ),
    }
    STORE.add_measure({
        "mode": "modality_exchange_dual",
        "ok": card["ok"],
        "face": face.get("face"),
        "nli": agree.get("label"),
        "lfm_verdict": lfm.get("verdict"),
    })
    STORE.graph_advance("ALIGN", {"via": "exchange_dual"})
    return _ok(card)


def tool_resource_status() -> dict:
    from resource_plane import plan_utilization

    return _ok(plan_utilization())


def tool_deep_ingest(path: str, goal: str = "", max_hours: float = 12.0) -> dict:
    from deep_loop import DeepLoop

    loop = DeepLoop()
    try:
        r = loop.ingest(path, goal=goal, max_hours=max_hours)
        r["status_md"] = loop.status_md()
        return _ok(r)
    except Exception as e:
        return _ok({"ok": False, "error": str(e)})


def tool_deep_tick() -> dict:
    from deep_loop import DeepLoop

    loop = DeepLoop()
    loop.load()
    r = loop.tick()
    return _ok(r)


def tool_deep_status() -> dict:
    from deep_loop import DeepLoop

    loop = DeepLoop()
    j = loop.load()
    if not j:
        return _ok({"ok": False, "error": "no deep job"})
    return _ok(
        {
            "ok": True,
            "job_id": j.job_id,
            "status": j.status,
            "ticks": j.ticks,
            "progress": loop._progress(),
            "source": j.source_path,
            "brief": j.final_brief_path,
            "md": loop.status_md(),
        }
    )


def tool_deep_run(path: str = "", max_hours: float = 12.0, sleep: float = 0.2) -> dict:
    """Foreground run-until-done (can be long). Prefer scheduler for multi-hour."""
    from deep_loop import DeepLoop

    loop = DeepLoop()
    if path:
        loop.ingest(path, max_hours=max_hours)
    else:
        loop.load()
    r = loop.run_until_done(sleep_s=sleep)
    return _ok(r)


def tool_doc_parse(path: str = "", query: str = "", k: int = 5) -> dict:
    """
    Grok-side dimensional parse: PDF/text → chunk+embed index → retrieve pack.
    Returns compact retrieval for Grok + path to full index; LFM gets pack_for_lfm text.
    """
    import hashlib

    from dimensional_parse import (
        build_index,
        pack_for_grok,
        pack_for_lfm,
        retrieve,
        save_index,
    )
    from deep_loop import load_document

    # load_document lives in deep_loop
    p = path or ""
    if not p:
        # use deep job source if any
        from deep_loop import DeepLoop

        loop = DeepLoop()
        j = loop.load()
        if j:
            p = j.source_path
    if not p:
        return _ok({"ok": False, "error": "path required"})
    try:
        text, meta = load_document(Path(p))
    except Exception as e:
        return _ok({"ok": False, "error": str(e)})

    idx_dir = STATE_DIR / "doc_parse"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / (hashlib.sha256(meta["sha256"].encode()).hexdigest()[:16] + ".json")
    if idx_path.exists():
        from dimensional_parse import load_index

        index = load_index(idx_path)
        built = False
    else:
        index = build_index(text, embed=True, max_chunks=64)
        save_index(index, idx_path)
        built = True

    q = query or STORE.s.get("intent") or meta.get("path") or "main thesis"
    hits = retrieve(index, q, k=int(k) or 5)
    grok_pack = pack_for_grok(index, hits, q)
    lfm_pack = pack_for_lfm(q, hits)
    (idx_dir / "last_grok_pack.json").write_text(json.dumps(grok_pack, indent=2), encoding="utf-8")


def tool_kb_build(
    roots: str = "",
    max_files: int = 20,
    query: str = "",
) -> dict:
    """
    Multi-file KB manifold: past corpus → embed stalks → present retrieve.
    Default Job1 family is jina-v5 (dim=1024) when jina :8765 is up; nomic is
    fallback. Callers must honor returned embed_family and dim (never hardcode 768).
    Default roots: prime docs + deep state (gated, not whole drive thrash).
    """
    from kb_index import DEFAULT_ROOTS, build_kb_index, default_out

    root_list = [Path(x.strip()) for x in roots.split(";") if x.strip()] if roots else list(DEFAULT_ROOTS)
    out = STATE_DIR / "kb" / "manifold_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # prefer shared default under prime/state/kb
    try:
        out = default_out()
        out.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    r = build_kb_index(
        root_list,
        out,
        embed=True,
        max_files=int(max_files) or 20,
        max_chunks_per_file=10,
        max_total_chunks=80,
        query_probe=query
        or STORE.s.get("intent")
        or "geometry manifold LFM nomic OPEN STOP residue never forced",
    )
    STORE.add_measure({"mode": "kb_build", **{k: v for k, v in r.items() if k not in ("sources", "probe")}})
    return _ok(r)


def tool_kb_query(query: str = "", k: int = 5) -> dict:
    """Retrieve from KB manifold into present enter (time-travel handoff)."""
    from kb_index import default_out, query_kb

    q = query or STORE.s.get("intent") or ""
    if not q:
        return _ok({"ok": False, "error": "query required"})
    path = default_out()
    if not path.is_file():
        # try state dir
        alt = STATE_DIR / "kb" / "manifold_index.json"
        path = alt if alt.is_file() else path
    if not path.is_file():
        return _ok({"ok": False, "error": f"no index at {path}; call kb_build first"})
    r = query_kb(path, q, k=int(k) or 5)
    # strip embeds from measure
    STORE.add_measure({
        "mode": "kb_query",
        "ok": True,
        "query": q[:200],
        "n_hits": len(r.get("hits") or []),
        "top_scores": [h.get("score") for h in (r.get("hits") or [])[:4]],
    })
    # drop raw embed from response if any
    hits = [{kk: vv for kk, vv in h.items() if kk != "embed"} for h in (r.get("hits") or [])]
    return _ok({**r, "hits": hits, "lfm_pack_preview": (r.get("lfm_pack") or "")[:1500]})
    (idx_dir / "last_lfm_pack.txt").write_text(lfm_pack, encoding="utf-8")
    return _ok(
        {
            "ok": True,
            "built_index": built,
            "index_path": str(idx_path),
            "source": meta,
            "grok_pack": grok_pack,
            "lfm_pack_preview": lfm_pack[:1500],
            "lfm_pack_path": str(idx_dir / "last_lfm_pack.txt"),
            "handoff": (
                "Grok uses grok_pack for planning/code. "
                "LFM roles should consume full lfm_pack text (path). "
                "Vectors stay in index for cosine retrieval — not dumped into chat."
            ),
        }
    )


def tool_measure_parallel(
    prompt: str = "",
    domain: str = "frb",
    domains: str = "code,rplc,eref,field",
    include_lm_scout: bool = False,
) -> dict:
    """Fan-out CPU measures within RAM budget; optional one LM scout on preferred small model."""
    ws = STORE.s.get("workspace") or str(Path.cwd())
    if not STORE.s.get("session_id"):
        return _ok({"ok": False, "error": "call session.start first"})
    human_src = prompt or STORE.s.get("intent") or ""
    STORE.graph_advance("MEASURE", {"mode": "parallel"})
    report = run_measure(
        mode="all",
        workspace=ws,
        prompt=human_src,
        domain=domain,
        domains=domains,
        lm_base=(STORE.s.get("lm") or {}).get("base_url", "http://127.0.0.1:1234/v1"),
        parallel=True,
    )
    if include_lm_scout:
        scout = run_measure(
            mode="lm",
            workspace=ws,
            prompt=human_src or "List 5 risks for this session intent.",
            lm_base=(STORE.s.get("lm") or {}).get("base_url", "http://127.0.0.1:1234/v1"),
        )
        report.setdefault("parts", {})["lm_scout"] = scout
    STORE.add_measure(report)
    if report.get("parts", {}).get("projection"):
        STORE.record_alignment(report["parts"]["projection"])
    return _ok({"ok": bool(report.get("ok", True)), "report": report})


TOOLS: dict[str, dict[str, Any]] = {
    "session_start": {
        "fn": tool_session_start,
        "description": "Start Prime session: state store + compute graph (META_META). Call first.",
        "schema": {
            "type": "object",
            "properties": {
                "workspace": {"type": "string"},
                "intent": {"type": "string"},
                "modes": {"type": "string", "description": "comma: code,lm,rplc,research,harness"},
            },
        },
    },
    "session_status": {
        "fn": tool_session_status,
        "description": "Session card: phase, path, claims, residue, open questions.",
        "schema": {"type": "object", "properties": {}},
    },
    "restrict": {
        "fn": tool_restrict,
        "description": "Lock goal, non-goals, success checks, constraints. Required before OPEN.",
        "schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "non_goals": {"type": "string"},
                "success": {"type": "string"},
                "constraints": {"type": "string"},
            },
            "required": ["goal"],
        },
    },
    "graph_plan": {
        "fn": tool_graph_plan,
        "description": "Replot compute graph for intent (see nodes before executing).",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "modes": {"type": "string"},
            },
        },
    },
    "graph_advance": {
        "fn": tool_graph_advance,
        "description": "Advance along legal law edge to node (RESTRICT, MEASURE, AUDIT, ...).",
        "schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["to"],
        },
    },
    "graph_show": {
        "fn": tool_graph_show,
        "description": "Show mermaid + node statuses + path.",
        "schema": {"type": "object", "properties": {}},
    },
    "measure": {
        "fn": tool_measure,
        "description": "Instrument: smoke|rplc|eref|lm|lm_models|project|all. project=bilateral language glue.",
        "schema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "domain": {"type": "string"},
                "domains": {"type": "string", "description": "for project mode: code,rplc,eref,field"},
            },
        },
    },
    "project_human": {
        "fn": tool_project_human,
        "description": "Project human-side language stalk (intent/restrict). Language is projection, not ground.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    },
    "project_domain": {
        "fn": tool_project_domain,
        "description": "Project domain-side language: code|rplc|eref|field|lm. Domain speak is also projection.",
        "schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "text": {"type": "string"},
            },
        },
    },
    "project_align": {
        "fn": tool_project_align,
        "description": "Align human↔domain projections on shared interface. Frustrated glue blocks OPEN.",
        "schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "empty = bilateral multi-domain"},
            },
        },
    },
    "condition_pulse": {
        "fn": tool_condition_pulse,
        "description": "Sheaf/condition gate on artifact before emit. OBSTRUCTED ⇒ do not OPEN.",
        "schema": {
            "type": "object",
            "properties": {
                "artifact": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["artifact"],
        },
    },
    "condition_read": {
        "fn": tool_condition_read,
        "description": "Read condition transducer state.",
        "schema": {"type": "object", "properties": {}},
    },
    "audit": {
        "fn": tool_audit,
        "description": "OPEN or STOP under controls. Force-OPEN without measures/restrict is blocked.",
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "description": "OPEN or STOP"},
                "reasons": {"type": "string", "description": "semicolon-separated"},
            },
            "required": ["verdict"],
        },
    },
    "claim_record": {
        "fn": tool_claim_record,
        "description": "Record OPEN|RESIDUE|CONTESTED claim (OPEN only if last audit OPEN).",
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "text": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["status", "text"],
        },
    },
    "cert_write": {
        "fn": tool_cert_write,
        "description": "Write prime_cert.json for the session.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
    "ask_need": {
        "fn": tool_ask_need,
        "description": "ACCURACY GATE: halt for human question. Use when answer would be guessed.",
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "why": {"type": "string"},
                "options": {"type": "string"},
            },
            "required": ["question", "why"],
        },
    },
    "ask_answer": {
        "fn": tool_ask_answer,
        "description": "Record human answer to ask_need question.",
        "schema": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["question_id", "answer"],
        },
    },
    "lm_models": {
        "fn": tool_lm_models,
        "description": "List LM Studio models on :1234.",
        "schema": {"type": "object", "properties": {}},
    },
    "lm_scout": {
        "fn": tool_lm_scout,
        "description": "Single-fiber LM Studio scout. Measure only — not OPEN.",
        "schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "system": {"type": "string"},
            },
            "required": ["prompt"],
        },
    },
    "lm_load": {
        "fn": tool_lm_load,
        "description": "Native POST /api/v1/models/load — make a model fiber resident (RAM-aware context_length).",
        "schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "context_length": {"type": "integer"},
            },
            "required": ["model"],
        },
    },
    "lm_unload": {
        "fn": tool_lm_unload,
        "description": "Native POST /api/v1/models/unload — free a resident fiber by instance_id.",
        "schema": {
            "type": "object",
            "properties": {"instance_id": {"type": "string"}},
            "required": ["instance_id"],
        },
    },
    "lm_embed": {
        "fn": tool_lm_embed,
        "description": (
            "OpenAI-compat embeddings — Job1 aboutness metric on language stalks. "
            "Default: jina-v5-small (dim=1024) via :8765; nomic is fallback. "
            "Parse returned embed_family and dim; never assume 768."
        ),
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    },
    "enter_projection": {
        "fn": tool_enter_projection,
        "description": (
            "One enter → L0–L7 layered LFM ops (JSON SCOUT/FALSIFY/GLUE/VERDICT) default; "
            "mode=multi_model|legacy_roles optional. MEASURE only."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "models": {"type": "string", "description": "default liquid/lfm2.5-1.2b"},
                "embed": {"type": "boolean"},
                "mode": {"type": "string", "description": "lfm_ops|layered|multi_model|legacy_roles"},
            },
        },
    },
    "lms_ensure": {
        "fn": tool_lms_ensure,
        "description": "L1 residency gate: keep one LFM + nomic embed; unload duplicate/deep thrash.",
        "schema": {
            "type": "object",
            "properties": {"context_length": {"type": "integer"}},
        },
    },
    "lms_layers": {
        "fn": tool_lms_layers,
        "description": (
            "LMS layered stack L0.5–L7: action=matrix|health|catalog|ensure|ops|home|policy|logs. "
            "home/logs read ~/.lmstudio + server-logs for residency/context gates."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "matrix|health|catalog|ensure|ops|home|policy|logs",
                },
                "prompt": {"type": "string", "description": "required when action=ops"},
            },
        },
    },
    "meta_loop": {
        "fn": tool_meta_loop,
        "description": "One-shot META_META: start session + plot full compute graph for intent.",
        "schema": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
    "exchange": {
        "fn": tool_exchange,
        "description": (
            "MANDATORY modality exchange on user enter: LFM SCOUT/FALSIFY/GLUE/VERDICT "
            "+ bilateral domain projection + optional smoke/rplc. MEASURE only."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "include_domain_measures": {"type": "boolean"},
                "domains": {"type": "string"},
            },
        },
    },
    "resource_status": {
        "fn": tool_resource_status,
        "description": "CPU/GPU/NPU/RAM snapshot + utilization plan (within reason).",
        "schema": {"type": "object", "properties": {}},
    },
    "measure_parallel": {
        "fn": tool_measure_parallel,
        "description": "Parallel CPU fan-out: smoke+rplc+projection+lm_models; optional LM scout on small model.",
        "schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "domain": {"type": "string"},
                "domains": {"type": "string"},
                "include_lm_scout": {"type": "boolean"},
            },
        },
    },
    "deep_ingest": {
        "fn": tool_deep_ingest,
        "description": "Load deep research PDF/text into long-horizon verify job (hours ok).",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "goal": {"type": "string"},
                "max_hours": {"type": "number"},
            },
            "required": ["path"],
        },
    },
    "deep_tick": {
        "fn": tool_deep_tick,
        "description": "Advance deep job one work item (LFM verify). Call repeatedly or via scheduler.",
        "schema": {"type": "object", "properties": {}},
    },
    "deep_status": {
        "fn": tool_deep_status,
        "description": "Deep job progress, OPEN/STOP/residue counts, brief path.",
        "schema": {"type": "object", "properties": {}},
    },
    "deep_run": {
        "fn": tool_deep_run,
        "description": "Run deep job until done (foreground; can be long).",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_hours": {"type": "number"},
                "sleep": {"type": "number"},
            },
        },
    },
    "doc_parse": {
        "fn": tool_doc_parse,
        "description": "Dimensional parse PDF/text: chunk+embed index, retrieve top-k for Grok+LFM handoff packs.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
        },
    },
    "kb_build": {
        "fn": tool_kb_build,
        "description": (
            "Build multi-file KB manifold index (default jina dim=1024; nomic fallback). "
            "Honor embed_family/dim in the response. "
            "Time-travel: past corpus → present retrieval. roots=semicolon paths."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "roots": {"type": "string", "description": "semicolon-separated roots"},
                "max_files": {"type": "integer"},
                "query": {"type": "string", "description": "optional probe query after build"},
            },
        },
    },
    "kb_query": {
        "fn": tool_kb_query,
        "description": "Query KB manifold index → dimensional pack for Grok/LFM. MEASURE only.",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}


def _dispatch(name: str, arguments: dict[str, Any]) -> Any:
    spec = TOOLS.get(name)
    if not spec:
        return {"ok": False, "error": f"unknown tool {name}"}
    fn: Callable = spec["fn"]
    # filter kwargs to function params roughly
    import inspect

    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in (arguments or {}).items() if k in sig.parameters}
    try:
        return fn(**kwargs)
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()[-2000:]}


# ---------------------------------------------------------------------------
# FastMCP if available
# ---------------------------------------------------------------------------
try:
    from fastmcp import FastMCP  # type: ignore

    HAVE_FASTMCP = True
except Exception:
    HAVE_FASTMCP = False

if HAVE_FASTMCP:
    mcp = FastMCP("prime-session")

    @mcp.tool()
    def session_start(workspace: str = "", intent: str = "", modes: str = "code,lm,rplc") -> dict:
        return tool_session_start(workspace, intent, modes)

    @mcp.tool()
    def session_status() -> dict:
        return tool_session_status()

    @mcp.tool()
    def restrict(goal: str, non_goals: str = "", success: str = "", constraints: str = "") -> dict:
        return tool_restrict(goal, non_goals, success, constraints)

    @mcp.tool()
    def graph_plan(intent: str = "", modes: str = "code,lm,rplc") -> dict:
        return tool_graph_plan(intent, modes)

    @mcp.tool()
    def graph_advance(to: str, note: str = "") -> dict:
        return tool_graph_advance(to, note)

    @mcp.tool()
    def graph_show() -> dict:
        return tool_graph_show()

    @mcp.tool()
    def measure(
        mode: str = "smoke",
        prompt: str = "",
        model: str = "",
        domain: str = "frb",
        domains: str = "code,rplc,eref,field",
    ) -> dict:
        return tool_measure(mode, prompt, model, domain, domains)

    @mcp.tool()
    def project_human(text: str = "") -> dict:
        return tool_project_human(text)

    @mcp.tool()
    def project_domain(domain: str = "code", text: str = "") -> dict:
        return tool_project_domain(domain, text)

    @mcp.tool()
    def project_align(domain: str = "") -> dict:
        return tool_project_align(domain)

    @mcp.tool()
    def condition_pulse(artifact: str, context: str = "prime") -> dict:
        return tool_condition_pulse(artifact, context)

    @mcp.tool()
    def condition_read() -> dict:
        return tool_condition_read()

    @mcp.tool()
    def audit(verdict: str, reasons: str = "") -> dict:
        return tool_audit(verdict, reasons)

    @mcp.tool()
    def claim_record(status: str, text: str, evidence: str = "") -> dict:
        return tool_claim_record(status, text, evidence)

    @mcp.tool()
    def cert_write(path: str = "") -> dict:
        return tool_cert_write(path)

    @mcp.tool()
    def ask_need(question: str, why: str, options: str = "") -> dict:
        return tool_ask_need(question, why, options)

    @mcp.tool()
    def ask_answer(question_id: str, answer: str) -> dict:
        return tool_ask_answer(question_id, answer)

    @mcp.tool()
    def lm_models() -> dict:
        return tool_lm_models()

    @mcp.tool()
    def lm_scout(prompt: str, model: str = "", system: str = "") -> dict:
        return tool_lm_scout(prompt, model, system)

    @mcp.tool()
    def lm_load(model: str, context_length: int = 0) -> dict:
        return tool_lm_load(model, context_length)

    @mcp.tool()
    def lm_unload(instance_id: str) -> dict:
        return tool_lm_unload(instance_id)

    @mcp.tool()
    def lm_embed(text: str = "") -> dict:
        return tool_lm_embed(text)

    @mcp.tool()
    def enter_projection(prompt: str = "", models: str = "", embed: bool = True, mode: str = "lfm_ops") -> dict:
        return tool_enter_projection(prompt, models, embed, mode)

    @mcp.tool()
    def lms_ensure(context_length: int = 0) -> dict:
        return tool_lms_ensure(context_length)

    @mcp.tool()
    def lms_layers(prompt: str = "", action: str = "matrix") -> dict:
        return tool_lms_layers(prompt, action)

    @mcp.tool()
    def meta_loop(intent: str) -> dict:
        return tool_meta_loop(intent)

    @mcp.tool()
    def exchange(
        prompt: str = "",
        include_domain_measures: bool = True,
        domains: str = "code,rplc,eref,field",
    ) -> dict:
        return tool_exchange(prompt, include_domain_measures, domains)

    @mcp.tool()
    def resource_status() -> dict:
        return tool_resource_status()

    @mcp.tool()
    def measure_parallel(
        prompt: str = "",
        domain: str = "frb",
        domains: str = "code,rplc,eref,field",
        include_lm_scout: bool = False,
    ) -> dict:
        return tool_measure_parallel(prompt, domain, domains, include_lm_scout)

    @mcp.tool()
    def deep_ingest(path: str, goal: str = "", max_hours: float = 12.0) -> dict:
        return tool_deep_ingest(path, goal, max_hours)

    @mcp.tool()
    def deep_tick() -> dict:
        return tool_deep_tick()

    @mcp.tool()
    def deep_status() -> dict:
        return tool_deep_status()

    @mcp.tool()
    def deep_run(path: str = "", max_hours: float = 12.0, sleep: float = 0.2) -> dict:
        return tool_deep_run(path, max_hours, sleep)

    @mcp.tool()
    def doc_parse(path: str = "", query: str = "", k: int = 5) -> dict:
        return tool_doc_parse(path, query, k)

    @mcp.tool()
    def kb_build(roots: str = "", max_files: int = 20, query: str = "") -> dict:
        return tool_kb_build(roots, max_files, query)

    @mcp.tool()
    def kb_query(query: str = "", k: int = 5) -> dict:
        return tool_kb_query(query, k)

    def main():
        print("prime-session FastMCP online", file=sys.stderr)
        mcp.run()

else:
    # Minimal MCP JSON-RPC stdio (initialize, tools/list, tools/call)
    def main():
        def reply(msg_id, result=None, error=None):
            out = {"jsonrpc": "2.0", "id": msg_id}
            if error is not None:
                out["error"] = error
            else:
                out["result"] = result
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()

        print("prime-session minimal MCP online", file=sys.stderr)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}

            if method == "initialize":
                reply(
                    mid,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "prime-session", "version": "1.0.0"},
                    },
                )
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                tools = []
                for name, spec in TOOLS.items():
                    tools.append(
                        {
                            "name": name,
                            "description": spec["description"],
                            "inputSchema": spec.get("schema") or {"type": "object", "properties": {}},
                        }
                    )
                reply(mid, {"tools": tools})
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                result = _dispatch(name, args)
                reply(
                    mid,
                    {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                        "isError": not bool(result.get("ok", True))
                        if isinstance(result, dict) and "ok" in result
                        else False,
                    },
                )
            elif method == "ping":
                reply(mid, {})
            else:
                if mid is not None:
                    reply(mid, error={"code": -32601, "message": f"Method not found: {method}"})


if __name__ == "__main__":
    main()
