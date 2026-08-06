#!/usr/bin/env python3
"""
Truth plane — every-request instrument stack + optional measure loop.

Architecture (hybrid — NOT "off LMS")
-------------------------------------
  LMS :1234     chat fibers only (SCOUT small / PRESERVE frankenstein)
  :8765 jina    Job1 aboutness embeds (llama-server --embedding)
  torch/ORT     Job2 DeBERTa NLI + Job1.5 neural rerank (off-LMS instruments)
  never         cosine → OPEN; local fiber self-certify production OPEN

Fiber modes
-----------
  scout     daily draft: LFM/Ministral; UNLOAD frankenstein (HEAVY)
  preserve  identity / high-stakes chain: frankenstein ALONE @ policy ctx
  auto      scout unless PRIME_FIBER_MODE=preserve or purpose=preserve

Every request (request_plane)
-----------------------------
  1. seamless substrate (jina + fiber mode + warm NLI/rerank)
  2. dual_enter (retrieve hybrid+neural, roles, aboutness diagnostic)
  3. force Job2 DeBERTa (prefer mutual on claim-shaped hyps)
  4. cert_face — OPEN only as CANDIDATE; contradiction STOP
  5. optional truth_loop: re-MEASURE until stable or max rounds

Env
---
  PRIME_FIBER_MODE       scout | preserve | auto   (default auto→scout)
  PRIME_TRUTH_LOOP       1 to enable loop on request_plane (default 0)
  PRIME_TRUTH_ROUNDS     max loop rounds (default 3)
  PRIME_WARM_INSTRUMENTS 1 warm DeBERTa+rerank on substrate (default 1)
  PRIME_ACCEL            dml | cpu | auto          (default auto)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

FiberMode = Literal["scout", "preserve", "auto"]

STATE = Path(__file__).resolve().parent.parent / "state"
STATE.mkdir(parents=True, exist_ok=True)

PRESERVE_KEYS = (
    "frankenstein",
    "frankenstein-2.0",
    "thedrummer/frankenstein",
)

# Precision domains — loop applies extra mutual-NLI + residue refusal
TRUTH_DOMAINS = ("math", "code", "physics", "technology", "claim", "audit", "general")


def _atomic_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (tmp + replace) for concurrent readers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def resolve_fiber_mode(
    mode: str | None = None,
    *,
    purpose: str | None = None,
) -> FiberMode:
    raw = (mode or os.environ.get("PRIME_FIBER_MODE") or "auto").lower().strip()
    if purpose and purpose.lower() in ("preserve", "identity", "holonomy", "certify"):
        return "preserve"
    if raw in ("scout", "preserve"):
        return raw  # type: ignore[return-value]
    return "scout"  # auto → scout for daily; preserve is explicit


def frankenstein_required(mode: FiberMode | None = None) -> bool:
    """True only when PRESERVE fiber is the job. Daily dual_enter does NOT need it."""
    m = mode or resolve_fiber_mode()
    return m == "preserve"


def frankenstein_loaded(
    base: str = "http://127.0.0.1:1234",
    *,
    mode: FiberMode | None = None,
) -> dict[str, Any]:
    req = frankenstein_required(mode)
    try:
        from lms_layers import l1_catalog

        cat = l1_catalog(base=base)
        for m in cat.get("models") or []:
            key = (m.get("key") or "").lower()
            if any(p in key for p in PRESERVE_KEYS) and m.get("loaded"):
                ctx = 0
                for inst in m.get("loaded_instances") or []:
                    cfg = inst.get("config") or {}
                    ctx = max(ctx, int(cfg.get("context_length") or 0))
                return {
                    "loaded": True,
                    "key": m.get("key"),
                    "ctx": ctx,
                    "required": req,
                }
        return {"loaded": False, "required": req}
    except Exception as e:
        return {"loaded": False, "error": str(e), "required": req}


def architecture_map() -> dict[str, Any]:
    return {
        "lms_role": "chat fibers only (SCOUT / PRESERVE)",
        "lms_base": "http://127.0.0.1:1234",
        "job1_aboutness": {
            "engine": "jina-v5 via llama-server --embedding",
            "base": os.environ.get("PRIME_JINA_BASE", "http://127.0.0.1:8765"),
            "lms": False,
            "note": "never LMS /v1/embeddings for jina (silent nomic remap)",
        },
        "job1_5_rerank": {
            "engine": "jina-reranker-v3 (transformers) or ladder",
            "lms": False,
        },
        "job2_agreement": {
            "engine": "DeBERTa cross-encoder NLI (prefer); LFM NLI fallback",
            "lms": "fallback only",
            "owns_open_gate": True,
        },
        "npu_hexagon": {
            "present": "Snapdragon X Plus Hexagon",
            "used_today": "measure fabric (HTP QDQ stress/smoke); not product NLI",
            "path": "onnxruntime-qnn plugin + QDQ graphs; CPU remains OPEN authority until E3 parity",
            "run_id": "npu-htp-2026-08-06",
            "target_jobs": ["Job2 NLI (post-parity)", "always-on measure fabric"],
        },
        "frankenstein": {
            "when": "PRESERVE / identity / holonomy only",
            "not_for": "daily SCOUT dual_enter / aboutness / NLI instruments",
            "co_load": "FORBIDDEN with other heavies during identity measure",
        },
        "law": (
            "aboutness must not promote OPEN; NLI owns agreement; "
            "residue never forced; production OPEN needs domain audit"
        ),
    }


_ACCEL_CACHE: dict[str, Any] | None = None
_ACCEL_CACHE_TS: float = 0.0


def accel_status(*, force: bool = False) -> dict[str, Any]:
    """Probe acceleration: QNN Hexagon NPU (plugin), DML, CPU torch.

    Cached ~60s unless force=True (avoid re-import/probe every request).
    """
    global _ACCEL_CACHE, _ACCEL_CACHE_TS
    now = time.time()
    if (
        not force
        and _ACCEL_CACHE is not None
        and (now - _ACCEL_CACHE_TS) < 60.0
    ):
        return dict(_ACCEL_CACHE)

    out: dict[str, Any] = {
        "preference": (os.environ.get("PRIME_ACCEL") or "auto").lower(),
        "ort": None,
        "torch": None,
        "hexagon_npu": None,
    }
    try:
        import onnxruntime as ort

        prov = ort.get_available_providers()
        out["ort"] = {
            "version": ort.__version__,
            "providers_builtin": prov,
            "dml": "DmlExecutionProvider" in prov,
        }
    except Exception as e:
        out["ort"] = {"ok": False, "error": str(e)}
    # Register QNN plugin — only way Hexagon HTP appears
    try:
        from npu_qnn import register

        npu = register()
        out["hexagon_npu"] = {
            "ok": npu.get("ok"),
            "n_qnn_devices": npu.get("n_qnn_devices"),
            "htp_dll": npu.get("htp_dll"),
            "qnn_ver": npu.get("qnn"),
            "registered": npu.get("registered"),
            "note": npu.get("note"),
            "error": npu.get("error"),
        }
        smoke = STATE / "npu_qnn_smoke.json"
        if smoke.is_file():
            try:
                s = json.loads(smoke.read_text(encoding="utf-8"))
                out["hexagon_npu"]["last_smoke"] = s.get("verdict")
                out["hexagon_npu"]["last_on_qnn"] = (s.get("run") or {}).get("on_qnn_ep")
            except Exception:
                pass
    except Exception as e:
        out["hexagon_npu"] = {"ok": False, "error": str(e)}
    try:
        import torch

        out["torch"] = {
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
        }
        pref = out["preference"]
        if pref in ("auto", "cpu", "npu"):
            n = max(2, (os.cpu_count() or 4) // 2)
            torch.set_num_threads(n)
            out["torch"]["threads_set"] = n
    except Exception as e:
        out["torch"] = {"ok": False, "error": str(e)}
    _ACCEL_CACHE = dict(out)
    _ACCEL_CACHE_TS = now
    return out


def warm_instruments() -> dict[str, Any]:
    """Load DeBERTa + reranker into memory so first dual_enter is not cold."""
    if os.environ.get("PRIME_WARM_INSTRUMENTS", "1").strip() in ("0", "false", "no"):
        return {"ok": True, "skipped": True}
    t0 = time.time()
    out: dict[str, Any] = {"ok": True}
    try:
        from entailment_glue import nli_cross_encoder

        r = nli_cross_encoder(
            "E_ref meets production readiness under measured audit.",
            "Under measured audit, E_ref satisfies production readiness criteria.",
        )
        out["nli"] = {
            "ok": r.get("ok"),
            "model": r.get("model"),
            "label": r.get("label"),
            "engine": r.get("engine"),
        }
    except Exception as e:
        out["nli"] = {"ok": False, "error": str(e)[:200]}
        out["ok"] = False
    try:
        from rerank_service import rerank_status, score_docs

        st = rerank_status()
        # light warm predict
        sc = score_docs(
            "ownership guidelines",
            ["Ensure strict adherence to ownership guidelines.", "Carbonara pasta."],
        )
        out["rerank"] = {
            "ok": sc.get("ok") or st.get("ok"),
            "model": st.get("model") or sc.get("model"),
            "kind": st.get("kind"),
            "error": sc.get("error") or st.get("error"),
        }
    except Exception as e:
        out["rerank"] = {"ok": False, "error": str(e)[:200]}
    out["seconds"] = round(time.time() - t0, 2)
    out["accel"] = accel_status()
    return out


def ensure_substrate(
    *,
    mode: str | None = None,
    purpose: str | None = None,
    base: str = "http://127.0.0.1:1234",
    chat_model: str | None = None,
) -> dict[str, Any]:
    """
    Full substrate for any request:
      jina ensure + fiber mode (scout|preserve) + instrument warm + accel
    """
    from residency import seamless_substrate

    fiber_mode = resolve_fiber_mode(mode, purpose=purpose)
    t0 = time.time()
    sub = seamless_substrate(
        chat_model=chat_model,
        base=base,
        fiber_mode=fiber_mode,
    )
    warm = warm_instruments()
    frank = frankenstein_loaded(base=base, mode=fiber_mode)
    # Substrate ok is residency+jina; instrument warm is best-effort (reported separately)
    out = {
        "ok": bool(sub.get("ok")),
        "instruments_ok": bool((warm.get("nli") or {}).get("ok")),
        "fiber_mode": fiber_mode,
        "frankenstein_required": frankenstein_required(fiber_mode),
        "frankenstein": frank,
        "substrate": sub,
        "instruments": warm,
        "architecture": architecture_map(),
        "seconds": round(time.time() - t0, 2),
        "law": architecture_map()["law"],
    }
    # honesty: if preserve required but not loaded → not ok
    if out["frankenstein_required"] and not frank.get("loaded"):
        out["ok"] = False
        out["error"] = "PRESERVE mode requires frankenstein loaded alone"
    _atomic_json(STATE / "truth_plane_last_substrate.json", out)
    return out


def _claims_from_card(card: dict[str, Any], prompt: str) -> list[dict[str, str]]:
    """Extract measurable claim pairs for the truth loop."""
    claims: list[dict[str, str]] = []
    intent = (prompt or "")[:1500]
    outs = card.get("outputs") or {}
    for role in ("VERDICT", "SCOUT", "GLUE", "FALSIFY"):
        o = outs.get(role) or {}
        if not isinstance(o, dict):
            continue
        from metric_text import strip_envelope

        hyp = strip_envelope(o.get("parsed") or o.get("raw") or o.get("content") or "")
        if hyp and len(hyp.strip()) >= 20:
            claims.append({"id": f"role:{role}", "premise": intent, "hypothesis": hyp[:800]})
    # also intent vs self-paraphrase guard (should entail)
    if intent:
        claims.append(
            {
                "id": "intent_self",
                "premise": intent,
                "hypothesis": intent[:500],
                "expect": "entailment",
            }
        )
    return claims[:8]


def truth_loop(
    claims: list[dict[str, str]],
    *,
    max_rounds: int | None = None,
    domain: str = "general",
) -> dict[str, Any]:
    """
    Looped MEASURE: mutual NLI on each claim until stable labels or max rounds.
    Never force-OPEN. Residue explicit.
    """
    from entailment_glue import mutual_entailment, nli_cross_encoder

    max_rounds = max_rounds or int(os.environ.get("PRIME_TRUTH_ROUNDS", "3"))
    domain = domain if domain in TRUTH_DOMAINS else "general"
    history: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []

    for rnd in range(1, max_rounds + 1):
        round_rows = []
        changed = False
        for c in claims:
            prem = c.get("premise") or ""
            hyp = c.get("hypothesis") or ""
            one = nli_cross_encoder(prem, hyp)
            mut = mutual_entailment(prem, hyp)
            row = {
                "id": c.get("id"),
                "round": rnd,
                "one_way": {
                    "label": one.get("label"),
                    "confidence": one.get("confidence"),
                    "model": one.get("model"),
                },
                "mutual": {
                    "agrees": mut.get("agrees"),
                    "gate": mut.get("gate"),
                    "ab": mut.get("ab"),
                    "ba": mut.get("ba"),
                },
                "expect": c.get("expect"),
            }
            # precision domains: contradiction or non-agree → residue
            if mut.get("gate") == "STOP" or one.get("label") == "contradiction":
                row["disposition"] = "STOP"
            elif mut.get("agrees"):
                row["disposition"] = "AGREE_MEASURE"  # not production OPEN
            else:
                row["disposition"] = "RESIDUE"
            round_rows.append(row)
        history.append({"round": rnd, "rows": round_rows})
        if rnd == 1:
            final = round_rows
        else:
            # stability: labels match previous
            prev = {r["id"]: r for r in final}
            for r in round_rows:
                p = prev.get(r["id"])
                if not p or p.get("disposition") != r.get("disposition"):
                    changed = True
            final = round_rows
            if not changed:
                break

    n_stop = sum(1 for r in final if r.get("disposition") == "STOP")
    n_agree = sum(1 for r in final if r.get("disposition") == "AGREE_MEASURE")
    n_res = sum(1 for r in final if r.get("disposition") == "RESIDUE")
    return {
        "ok": True,
        "job": "truth_loop",
        "domain": domain,
        "rounds_run": len(history),
        "max_rounds": max_rounds,
        "n_claims": len(claims),
        "n_stop": n_stop,
        "n_agree_measure": n_agree,
        "n_residue": n_res,
        # <2 rounds → stability unmeasured (not True)
        "stable": (
            (len(history) < max_rounds or not changed)
            if len(history) > 1
            else None
        ),
        "rows": final,
        "history": history
        if os.environ.get("PRIME_TRUTH_HISTORY", "0").strip() in ("1", "true", "yes")
        else None,
        "not_open_authority": True,
        "law": "loop is MEASURE only; residue never forced to OPEN",
    }


def request_plane(
    prompt: str,
    *,
    mode: str | None = None,
    purpose: str | None = None,
    domain: str = "general",
    base: str = "http://127.0.0.1:1234",
    model: str | None = None,
    retrieve_kb: bool = True,
    truth_loop_enabled: bool | None = None,
    max_tokens: int = 140,
) -> dict[str, Any]:
    """
    Canonical every-request entry for max alignment:
      substrate → dual_enter → truth_loop(optional) → operator card
    """
    t0 = time.time()
    fiber_mode = resolve_fiber_mode(mode, purpose=purpose)
    # pick model for mode
    chat_model = model
    if fiber_mode == "preserve" and not chat_model:
        chat_model = _find_frankenstein_key(base) or model

    sub = ensure_substrate(
        mode=fiber_mode,
        purpose=purpose,
        base=base,
        chat_model=chat_model,
    )
    if fiber_mode == "preserve" and sub.get("substrate", {}).get("fiber", {}).get("model"):
        chat_model = sub["substrate"]["fiber"]["model"]

    from dual_enter import dual_enter

    card = dual_enter(
        prompt,
        base=base,
        model=chat_model
        or ((sub.get("substrate") or {}).get("fiber") or {}).get("model"),
        retrieve_kb=retrieve_kb,
        max_tokens=max_tokens,
        ensure=False,  # already ensured with mode
        fiber_mode=fiber_mode,
    )
    card["fiber_mode"] = fiber_mode
    # Attach full substrate (dual_enter skipped ensure)
    card["substrate"] = sub.get("substrate") or card.get("substrate")
    card["truth_plane"] = {
        "substrate_ok": sub.get("ok"),
        "frankenstein": sub.get("frankenstein"),
        "instruments": {
            "nli": (sub.get("instruments") or {}).get("nli"),
            "rerank": (sub.get("instruments") or {}).get("rerank"),
            "accel": (sub.get("instruments") or {}).get("accel"),
        },
        "architecture": architecture_map(),
    }
    fiber_blk = (sub.get("substrate") or {}).get("fiber") or {}

    # Strengthen agreement: if scout hyp exists, also run mutual for operator honesty
    try:
        from entailment_glue import mutual_entailment
        from metric_text import strip_envelope

        outs = card.get("outputs") or {}
        hyp = ""
        for role in ("VERDICT", "SCOUT", "GLUE"):
            o = outs.get(role) or {}
            if isinstance(o, dict):
                hyp = strip_envelope(
                    o.get("parsed") or o.get("raw") or o.get("content") or ""
                )
            if hyp and len(hyp) > 20:
                break
        if hyp:
            mut = mutual_entailment((prompt or "")[:1500], hyp[:800])
            card["mutual_agreement"] = mut
            # if mutual STOP, demote face
            if mut.get("gate") == "STOP":
                face = card.get("cert_face") or {}
                face = dict(face)
                face["face"] = "STOP"
                face["closed"] = True
                face["mutual_demote"] = True
                card["cert_face"] = face
                card["verdict"] = "STOP"
                op = dict(card.get("operator_summary") or {})
                op["face"] = "STOP"
                op["mutual_gate"] = "STOP"
                card["operator_summary"] = op
    except Exception as e:
        card["mutual_agreement"] = {"ok": False, "error": str(e)[:200]}

    # Explicit False from caller wins; else env; precision domains only if not opted out
    if truth_loop_enabled is False:
        do_loop = False
    elif truth_loop_enabled is True:
        do_loop = True
    else:
        do_loop = os.environ.get("PRIME_TRUTH_LOOP", "0").strip() in ("1", "true", "yes")
        if (
            not do_loop
            and domain in ("math", "code", "physics", "technology")
            and os.environ.get("PRIME_TRUTH_LOOP_PRECISION", "1").strip()
            not in ("0", "false", "no")
        ):
            do_loop = True

    if do_loop:
        claims = _claims_from_card(card, prompt)
        card["truth_loop"] = truth_loop(claims, domain=domain)
        # residue / stop from loop elevates cert honesty
        tl = card["truth_loop"]
        if tl.get("n_stop", 0) > 0:
            card["verdict"] = "STOP"
            face = dict(card.get("cert_face") or {})
            face["face"] = "STOP"
            face["truth_loop_stop"] = True
            card["cert_face"] = face
        elif tl.get("n_residue", 0) > 0 and (card.get("cert_face") or {}).get("face") == "OPEN_CANDIDATE":
            face = dict(card.get("cert_face") or {})
            face["face"] = "NEED_INFO"
            face["truth_loop_residue"] = True
            card["cert_face"] = face
            card["verdict"] = "NEED_INFO"

    card["elapsed_s"] = round(time.time() - t0, 2)
    op = dict(card.get("operator_summary") or {})
    # Refresh face from cert_face after any demotion (mutual / truth_loop)
    op["face"] = (card.get("cert_face") or {}).get("face") or op.get("face")
    op["fiber_mode"] = fiber_mode
    op["fiber_model"] = fiber_blk.get("model") or op.get("fiber_model") or chat_model
    op["fiber_ctx"] = fiber_blk.get("loaded_ctx") or op.get("fiber_ctx")
    op["frankenstein_loaded"] = (sub.get("frankenstein") or {}).get("loaded")
    op["frankenstein_required"] = frankenstein_required(fiber_mode)
    op["nli_engine"] = ((sub.get("instruments") or {}).get("nli") or {}).get(
        "engine"
    ) or (card.get("agreement") or {}).get("engine")
    op["nli_model"] = ((sub.get("instruments") or {}).get("nli") or {}).get("model")
    op["rerank_model"] = ((sub.get("instruments") or {}).get("rerank") or {}).get("model")
    op["jina"] = ((sub.get("substrate") or {}).get("jina") or {}).get("status")
    op["truth_loop"] = bool(card.get("truth_loop"))
    op["elapsed_s"] = card["elapsed_s"]
    op["not_open_authority"] = True
    card["operator_summary"] = op

    _atomic_json(
        STATE / "truth_plane_last_request.json",
        {
            "operator_summary": op,
            "cert_face": card.get("cert_face"),
            "fiber_mode": fiber_mode,
            "truth_loop": card.get("truth_loop"),
            "mutual": card.get("mutual_agreement"),
        },
    )
    return card


def _find_frankenstein_key(base: str) -> str | None:
    try:
        from lms_layers import l1_catalog

        for m in (l1_catalog(base=base).get("models") or []):
            key = m.get("key") or ""
            if any(p in key.lower() for p in PRESERVE_KEYS):
                return key
    except Exception:
        pass
    # common key patterns
    return os.environ.get("PRIME_PRESERVE_MODEL") or None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Truth plane")
    ap.add_argument("cmd", nargs="?", default="map", choices=("map", "substrate", "enter", "accel"))
    ap.add_argument("--mode", default=None, help="scout|preserve|auto")
    ap.add_argument("--prompt", default="Measure whether aboutness may promote OPEN.")
    ap.add_argument("--domain", default="technology")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    if a.cmd == "map":
        print(json.dumps({
            "architecture": architecture_map(),
            "fiber_mode": resolve_fiber_mode(a.mode),
            "frankenstein": frankenstein_loaded(),
            "accel": accel_status(),
        }, indent=2))
    elif a.cmd == "accel":
        print(json.dumps(accel_status(), indent=2))
    elif a.cmd == "substrate":
        print(json.dumps(ensure_substrate(mode=a.mode), indent=2, default=str))
    else:
        if a.loop:
            os.environ["PRIME_TRUTH_LOOP"] = "1"
        card = request_plane(a.prompt, mode=a.mode, domain=a.domain)
        print(json.dumps(card.get("operator_summary"), indent=2))
        print("face", (card.get("cert_face") or {}).get("face"))
        print("fiber_mode", card.get("fiber_mode"))
