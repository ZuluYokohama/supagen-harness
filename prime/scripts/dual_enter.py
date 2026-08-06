"""
dual_enter — one enter, two jobs, one cert face.

  Job 1 ABOUTNESS: jina retrieve (optional KB) + diagnostic cosines (nomic fallback)
  Job 2 AGREEMENT: NLI reason→label (never cosine)
  CERT FACE: process + reference gates; OPEN only as CANDIDATE measure

Cosine never promotes OPEN. Contradiction NLI demotes. Certificate is the logogram.
Seamless substrate on every enter (jina ensure + unload heavies + pick fiber).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from metric_text import strip_envelope, strip_prompt_chrome


def _kb_retrieve(query: str, k: int = 4) -> dict[str, Any]:
    """Optional Job 1 retrieve from KB manifold if present."""
    try:
        from dimensional_parse import load_index, pack_for_lfm, retrieve
        from kb_index import default_out

        path = default_out()
        if not path.is_file():
            alt = Path(__file__).resolve().parent.parent / "state" / "kb" / "manifold_index.json"
            path = alt if alt.is_file() else path
        if not path.is_file():
            return {"ok": False, "skipped": True, "reason": "no_kb_index"}
        index = load_index(path)
        hits = retrieve(index, query, k=k)
        return {
            "ok": True,
            "job": "retrieval_aboutness",
            "n_hits": len(hits),
            "hits": [
                {
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "score_kind": h.get("score_kind") or "aboutness",
                    "source": h.get("source"),
                    "title": (h.get("title") or "")[:80],
                }
                for h in hits
            ],
            "lfm_pack": pack_for_lfm(query, hits),
            "index": str(path),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def cert_face(
    *,
    verdict: str,
    agreement: dict[str, Any] | None,
    aboutness: dict[str, Any] | None,
    fatal: bool,
    regime: str,
    prompt_preview: str,
) -> dict[str, Any]:
    """
    Logogram-shaped certificate surface.
    Closes only as MEASURE candidate — production OPEN still needs domain audit.
    """
    agree = agreement or {}
    label = agree.get("label") or "unknown"
    nli_agrees = bool(agree.get("agrees"))
    mean_cos = None
    if aboutness:
        mean_cos = aboutness.get("mean_cosine")
        if mean_cos is None and isinstance(aboutness.get("per_role"), dict):
            vals = list(aboutness["per_role"].values())
            mean_cos = sum(vals) / len(vals) if vals else None

    # Process gate (roles/regime)
    process_ok = regime in ("structured_ops",) and not fatal
    # Reference gate (NLI owns agreement)
    if label == "contradiction":
        ref_gate = "STOP"
        ref_ok = False
    elif nli_agrees:
        ref_gate = "PASS"
        ref_ok = True
    else:
        ref_gate = "NEED_INFO"
        ref_ok = False

    # Candidate OPEN only if both process + reference allow
    v = (verdict or "NEED_INFO").upper()
    if fatal or label == "contradiction":
        face = "STOP"
    elif v in ("OPEN", "OPEN_CANDIDATE") and ref_ok and process_ok:
        face = "OPEN_CANDIDATE"
    elif v == "STOP" or ref_gate == "STOP":
        face = "STOP"
    else:
        face = "NEED_INFO"

    closed = face in ("OPEN_CANDIDATE", "STOP")  # logogram has a hard answer
    return {
        "kind": "prime_cert_face",
        "face": face,
        "closed": closed,
        "process": {
            "regime": regime,
            "fatal": fatal,
            "lfm_verdict": v,
            "ok": process_ok,
        },
        "reference": {
            "nli_label": label,
            "nli_agrees": nli_agrees,
            "nli_confidence": agree.get("confidence"),
            "nli_reason": (agree.get("reason") or "")[:200],
            "gate": ref_gate,
            "ok": ref_ok,
        },
        "aboutness_diagnostic": {
            "mean_cosine": mean_cos,
            "not_agreement": True,
            "note": "never promotes OPEN",
        },
        "prompt_preview": (prompt_preview or "")[:200],
        "law": "restrict→measure→audit→OPEN|STOP; residue never forced",
        "not_open_authority": True,
        "thesis": (
            "Certificate is the logogram: process + reference must both speak. "
            "Aboutness is a chart; NLI is agreement; production OPEN needs domain audit."
        ),
    }


def dual_enter(
    prompt: str,
    *,
    base: str = "http://127.0.0.1:1234",
    model: str | None = None,
    embed: bool = True,
    retrieve_kb: bool = True,
    k_retrieve: int = 4,
    max_tokens: int = 140,
    ensure: bool = True,
    roles: list[str] | None = None,
    fiber_mode: str | None = None,
) -> dict[str, Any]:
    """
    Canonical enter for the whole stack.
    Returns layered LFM ops + retrieval + agreement + cert_face.

    Hybrid architecture (not off-LMS):
      LMS chat fiber (SCOUT small / PRESERVE frankenstein)
      jina :8765 aboutness + DeBERTa NLI + neural rerank (off-LMS instruments)

    Seamless: jina ensure + fiber mode residency + promote chat to policy ctx.
    Frankenstein is NOT required for scout dual_enter.
    """
    t0 = time.time()
    import os as _os

    from lms_layers import DEFAULT_LFM, layered_enter
    from metric_text import pack_to_token_budget, strip_code_fences

    clean = strip_prompt_chrome(prompt or "")
    if not clean.strip():
        clean = (prompt or "").strip()

    mode = (fiber_mode or _os.environ.get("PRIME_FIBER_MODE") or "scout").lower()
    if mode not in ("scout", "preserve"):
        mode = "scout"

    substrate: dict[str, Any] = {"ok": True, "skipped": not ensure, "fiber_mode": mode}
    chat_model = model or DEFAULT_LFM
    instruments: dict[str, Any] = {}
    substrate_ok = not ensure  # if not ensuring, residency will load fiber itself
    if ensure:
        try:
            # Full truth-plane substrate when available (jina + mode + warm NLI/rerank)
            if _os.environ.get("PRIME_TRUTH_PLANE", "1").strip() not in (
                "0",
                "false",
                "no",
            ):
                from truth_plane import ensure_substrate

                plane = ensure_substrate(
                    mode=mode, base=base, chat_model=chat_model
                )
                substrate = plane.get("substrate") or {}
                substrate["fiber_mode"] = mode
                substrate["truth_plane_ok"] = plane.get("ok")
                instruments = plane.get("instruments") or {}
                fiber_m = (substrate.get("fiber") or {}).get("model")
                if fiber_m:
                    chat_model = fiber_m
                substrate_ok = bool(plane.get("ok") or substrate.get("ok"))
            else:
                from residency import seamless_substrate

                substrate = seamless_substrate(
                    chat_model=chat_model, base=base, fiber_mode=mode
                )
                if substrate.get("fiber", {}).get("model"):
                    chat_model = substrate["fiber"]["model"]
                substrate["fiber_mode"] = mode
                substrate_ok = bool(substrate.get("ok"))
        except Exception as e:
            substrate = {"ok": False, "error": str(e), "fiber_mode": mode}
            substrate_ok = False

    retrieval: dict[str, Any] = {"ok": False, "skipped": True}
    work_prompt = clean
    if retrieve_kb:
        retrieval = _kb_retrieve(clean, k=k_retrieve)
        if retrieval.get("ok") and retrieval.get("lfm_pack"):
            # Feed LFM a dimensional pack (aboutness) + intent — not the full KB
            pack = pack_to_token_budget(retrieval["lfm_pack"], max_tokens=2800)
            work_prompt = pack_to_token_budget(
                f"INTENT:\n{clean[:1200]}\n\n"
                f"{pack}\n\n"
                "Operate under design law. Prefer pack evidence. "
                "Residue never forced.",
                max_tokens=3500,
            )

    # Fast enter (PRIME_FAST_ENTER=1): SCOUT+FALSIFY+VERDICT only — still dual metric
    use_roles = roles
    if use_roles is None and _os.environ.get("PRIME_FAST_ENTER", "0") in (
        "1",
        "true",
        "yes",
    ):
        use_roles = ["SCOUT", "FALSIFY", "VERDICT"]

    # Skip residency thrash only when substrate actually loaded the fiber
    card = layered_enter(
        work_prompt,
        base=base,
        model=chat_model,
        embed=embed,
        ensure_residency=not (ensure and substrate_ok),
        roles=use_roles,
    )

    # Sanitize role outputs (fence-wrapped JSON from agentic models)
    outs = card.get("outputs") or {}
    if isinstance(outs, dict):
        for role, o in list(outs.items()):
            if not isinstance(o, dict):
                continue
            for k in ("raw", "content"):
                if isinstance(o.get(k), str):
                    o[k] = strip_code_fences(o[k])
            if isinstance(o.get("parsed"), str):
                o["parsed"] = strip_code_fences(o["parsed"])

    # One retry if roles are empty/placeholder (transient LMS glitch)
    def _role_payload_ok(o: dict) -> bool:
        blob = strip_envelope(o.get("parsed") or o.get("raw") or o.get("content") or "")
        return bool(blob and len(blob.strip()) >= 12)

    thin = not outs or sum(
        1 for o in outs.values() if isinstance(o, dict) and _role_payload_ok(o)
    ) < 2
    if thin:
        try:
            card2 = layered_enter(
                work_prompt,
                base=base,
                model=chat_model,
                embed=embed,
                ensure_residency=True,
            )
            outs2 = card2.get("outputs") or {}
            n2 = sum(
                1 for o in outs2.values() if isinstance(o, dict) and _role_payload_ok(o)
            )
            if n2 > sum(
                1 for o in outs.values() if isinstance(o, dict) and _role_payload_ok(o)
            ):
                card = card2
                outs = outs2
                card["role_retry"] = True
        except Exception as e:
            card["role_retry_error"] = str(e)[:160]

    # Explicit jina aboutness diagnostic on intent vs first role payload
    if embed:
        try:
            from nomic_metric import aboutness as aboutness_pair

            outs2 = card.get("outputs") or {}
            hyp = ""
            for role in ("SCOUT", "VERDICT", "GLUE"):
                o = outs2.get(role) or {}
                if isinstance(o, dict):
                    hyp = strip_envelope(o.get("parsed") or o.get("raw") or o.get("content") or "")
                if hyp and len(hyp) > 12:
                    break
            if hyp:
                ab = aboutness_pair(clean[:1500], hyp[:800])
                jsvc = ab.get("jina_service") or (
                    (substrate.get("jina") if isinstance(substrate, dict) else None)
                )
                card["aboutness_pair"] = {
                    "ok": ab.get("ok"),
                    "cosine": ab.get("cosine"),
                    "family": ab.get("family"),
                    "model": ab.get("model"),
                    "base": ab.get("base"),
                    "warning": ab.get("warning"),
                    "jina_service": jsvc,
                    "not_agreement": True,
                }
                # fold into aboutness diagnostic
                about = card.get("aboutness") if isinstance(card.get("aboutness"), dict) else {}
                about = dict(about or {})
                about["pair_cosine"] = ab.get("cosine")
                about["family"] = ab.get("family")
                about["not_agreement"] = True
                card["aboutness"] = about
        except Exception as e:
            card["aboutness_pair"] = {"ok": False, "error": str(e)[:200]}

    # Ensure agreement present — try VERDICT → SCOUT → GLUE → intent paraphrase
    agreement = card.get("agreement") if isinstance(card.get("agreement"), dict) else {}
    need_nli = (
        not agreement
        or not agreement.get("ok")
        or agreement.get("label") in (None, "unknown", "")
        or agreement.get("error")
    )
    if need_nli:
        try:
            from entailment_glue import glue_agreement

            outs = card.get("outputs") or {}
            hyp = ""
            for role in ("VERDICT", "SCOUT", "GLUE", "FALSIFY"):
                o = outs.get(role) or {}
                if not isinstance(o, dict):
                    continue
                hyp = strip_envelope(o.get("parsed") or o.get("raw") or o.get("content") or "")
                if hyp and len(hyp.strip()) >= 12:
                    break
            if not hyp or len(hyp.strip()) < 12:
                # Fall back: does pack-supported claim agree with intent? use first retrieval title
                hits = (retrieval.get("hits") or []) if retrieval.get("ok") else []
                if hits:
                    hyp = f"Proceed using evidence from: {hits[0].get('title') or hits[0].get('id')}"
                else:
                    hyp = f"Operationalize: {clean[:400]}"
            agreement = glue_agreement(clean[:1800], hyp[:800], prefer="auto", base=base)
            card["agreement"] = agreement
        except Exception as e:
            agreement = {
                "ok": False,
                "error": str(e),
                "label": "unknown",
                "agrees": False,
                "gate": "NEED_INFO",
            }
            card["agreement"] = agreement

    aboutness = card.get("aboutness") or {
        "mean_cosine": card.get("mean_cosine"),
        "not_agreement": True,
    }

    # Policy re-fold with agreement (layered already does; cert face is authority surface)
    verdict = card.get("verdict") or "NEED_INFO"
    if agreement.get("label") == "contradiction" and verdict == "OPEN_CANDIDATE":
        verdict = "STOP"
        card["verdict"] = verdict
        card["policy_reason"] = (
            (card.get("policy_reason") or "") + " | NLI contradiction demotes OPEN_CANDIDATE"
        ).strip(" |")
        card["fatal_flag"] = bool(card.get("fatal_flag"))

    face = cert_face(
        verdict=verdict,
        agreement=agreement,
        aboutness=aboutness if isinstance(aboutness, dict) else {},
        fatal=bool(card.get("fatal_flag")),
        regime=str(card.get("regime") or ""),
        prompt_preview=clean,
    )
    card["cert_face"] = face
    card["retrieval"] = {
        k: v for k, v in retrieval.items() if k != "lfm_pack"
    }
    if retrieval.get("lfm_pack"):
        card["retrieval"]["lfm_pack_chars"] = len(retrieval["lfm_pack"])
    # Slim top hits for operator / runner injection
    top_hits = []
    for h in (retrieval.get("hits") or [])[:3]:
        top_hits.append(
            {
                "score": h.get("score"),
                "cos": h.get("score_cos") or h.get("score"),
                "title": (h.get("title") or "")[:80],
                "id": h.get("id"),
                "method": h.get("method"),
            }
        )
    card["retrieval"]["top"] = top_hits
    card["enter_mode"] = "dual_enter"
    card["substrate"] = substrate
    card["prompt_preview"] = clean[:240]
    card["elapsed_s"] = round(time.time() - t0, 2)
    card["dual"] = {
        "job1_aboutness": True,
        "job2_agreement": True,
        "cosine_promotes_open": False,
        "cert_face": face.get("face"),
        "closed": face.get("closed"),
        "jina_family": (card.get("aboutness_pair") or {}).get("family"),
        "fiber_ctx": (substrate.get("fiber") or {}).get("loaded_ctx")
        if isinstance(substrate, dict)
        else None,
        "kb_top_score": top_hits[0]["score"] if top_hits else None,
    }
    # Flatten operator summary
    pair_cos = None
    if isinstance(card.get("aboutness_pair"), dict):
        pair_cos = card["aboutness_pair"].get("cosine")
    card["operator_summary"] = {
        "face": face.get("face"),
        "nli": agreement.get("label"),
        "nli_agrees": agreement.get("agrees"),
        "nli_confidence": agreement.get("confidence"),
        "aboutness_cos": aboutness.get("mean_cosine")
        if isinstance(aboutness, dict)
        else None,
        "aboutness_pair_cos": pair_cos,
        "aboutness_family": (card.get("aboutness_pair") or {}).get("family")
        or (aboutness.get("family") if isinstance(aboutness, dict) else None),
        "lfm_verdict": verdict,
        "fatal": card.get("fatal_flag"),
        "kb_hits": (retrieval.get("n_hits") if retrieval.get("ok") else 0),
        "kb_top": top_hits[0] if top_hits else None,
        "fiber_ctx": (substrate.get("fiber") or {}).get("loaded_ctx")
        if isinstance(substrate, dict)
        else None,
        "fiber_model": (substrate.get("fiber") or {}).get("model")
        if isinstance(substrate, dict)
        else chat_model,
        "jina": (substrate.get("jina") or {}).get("status")
        if isinstance(substrate, dict)
        else None,
        "fiber_mode": mode,
        "elapsed_s": round(time.time() - t0, 2),
        "not_open_authority": True,
    }
    card["fiber_mode"] = mode
    if instruments:
        card["instruments"] = {
            "nli": instruments.get("nli"),
            "rerank": instruments.get("rerank"),
            "accel": instruments.get("accel"),
        }
        op = card.get("operator_summary")
        if isinstance(op, dict):
            op["nli_engine"] = (instruments.get("nli") or {}).get("engine")
            op["rerank_model"] = (instruments.get("rerank") or {}).get("model")
    return card


def format_operator_md(card: dict[str, Any], prompt: str = "") -> str:
    """Human-facing runner / Grok injection — dual metric honest."""
    op = card.get("operator_summary") or {}
    face = (card.get("cert_face") or {}).get("face") or op.get("face")
    agree = card.get("agreement") or {}
    about = card.get("aboutness") or {}
    ret = card.get("retrieval") or {}
    lines = [
        "## PRIME DUAL ENTER (MEASURE only — jina aboutness + NLI + chat fiber)",
        f"face=**{face}** closed={ (card.get('cert_face') or {}).get('closed') } "
        f"elapsed_s={card.get('elapsed_s')}",
        f"NLI agreement: label={agree.get('label')} conf={agree.get('confidence')} "
        f"agrees={agree.get('agrees')} (Job 2 — owns glue)",
        f"aboutness cos={about.get('mean_cosine') if isinstance(about, dict) else card.get('mean_cosine')} "
        f"(Job 1 diagnostic — never promotes OPEN)",
        f"kb_hits={ret.get('n_hits', 0)} lfm_verdict={card.get('verdict')} fatal={card.get('fatal_flag')}",
        "",
        "### Cert face",
        f"- process: {(card.get('cert_face') or {}).get('process')}",
        f"- reference: label={agree.get('label')} reason={(agree.get('reason') or '')[:160]}",
        "",
        "### Law",
        "- Cosine = aboutness chart only.",
        "- NLI = agreement channel.",
        "- Certificate face is candidate measure — production OPEN needs domain audit.",
        "- Residue never forced.",
        f"- enter: {(prompt or card.get('prompt_preview') or '')[:200]!r}",
    ]
    outs = card.get("outputs") or {}
    if outs:
        lines.append("")
        lines.append("### LFM roles (payload)")
        for role in ("SCOUT", "FALSIFY", "GLUE", "VERDICT"):
            o = outs.get(role) or {}
            if not isinstance(o, dict):
                continue
            bit = o.get("parsed") or o.get("raw") or o.get("content") or ""
            if isinstance(bit, dict):
                bit = json.dumps(bit, ensure_ascii=False)
            bit = strip_envelope(bit) if bit else ""
            if bit:
                lines.append(f"- **{role}**: {str(bit)[:240]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    p = " ".join(sys.argv[1:]) or (
        "Wire dual metric enter: aboutness retrieve + NLI agreement; never force-OPEN."
    )
    r = dual_enter(p, retrieve_kb=True)
    print(json.dumps({
        "ok": r.get("ok"),
        "enter_mode": r.get("enter_mode"),
        "operator_summary": r.get("operator_summary"),
        "cert_face": r.get("cert_face"),
        "dual": r.get("dual"),
        "elapsed_s": r.get("elapsed_s"),
    }, indent=2))
