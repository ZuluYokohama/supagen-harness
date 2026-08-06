"""
LFM as orthogonal operator algebra — not multi-model cosplay.

Mainstream use: one small model = weak chatbot.
Our use: one resident fiber (liquid/lfm2.5-1.2b) + jina aboutness (nomic fallback), driven by
*abstract roles* (restriction maps) under the LMS layered gate stack (L0–L7).

Default path: ``lms_layers.layered_enter`` — JSON-gated SCOUT/FALSIFY/GLUE/VERDICT,
residency consolidation, stateful previous_response_id, embed glue, policy demotion.

Legacy free-text roles remain available via use_layers=False.
"""
from __future__ import annotations

import re
from typing import Any

from lm_studio_client import DEFAULT_BASE, DEFAULT_EMBED, DEFAULT_LFM, LMStudio, cosine

# re-export for callers
ROLES: dict[str, str] = {
    "SCOUT": (
        "ROLE=SCOUT. Map the human intent only. "
        "Output exactly 3 short bullets: WHAT / DOMAIN / SUCCESS. No advice. No OPEN."
    ),
    "FALSIFY": (
        "ROLE=FALSIFY. Attack the plan. "
        "Output exactly 3 short bullets of how this could be wrong or forced. "
        "End with one line: FATAL=yes|no."
    ),
    "GLUE": (
        "ROLE=GLUE. Language is projection from both sides. "
        "Name the shared interface words (open/stop/measure/audit/residue/prior/certify) "
        "that must hold. Output 3 bullets: SHARED / MISSING / RISK."
    ),
    "VERDICT": (
        "ROLE=VERDICT. Design law: restrict→measure→audit→OPEN|STOP. Residue never forced. "
        "You do NOT have authority to OPEN production claims alone. "
        "Output exactly one line: VERDICT=OPEN_CANDIDATE|STOP|NEED_INFO "
        "then one line reason."
    ),
}


def _parse_verdict(text: str) -> str:
    u = (text or "").upper()
    if "NEED_INFO" in u or "NEED INFO" in u:
        return "NEED_INFO"
    if "OPEN_CANDIDATE" in u or re.search(r"\bOPEN\b", u):
        if "STOP" in u and u.find("STOP") < u.find("OPEN"):
            return "STOP"
        return "OPEN_CANDIDATE"
    if "STOP" in u:
        return "STOP"
    return "OTHER"


def lfm_role_pass(
    prompt: str,
    base: str = DEFAULT_BASE,
    model: str = DEFAULT_LFM,
    roles: list[str] | None = None,
    embed: bool = True,
    max_tokens: int = 160,
    context_length: int | None = None,
    previous_response_id: str | None = None,
    integrations: list[Any] | None = None,
    use_layers: bool = True,
) -> dict[str, Any]:
    """
    One enter → orthogonal roles on a single LFM fiber.

    Default ``use_layers=True``: full L0–L7 JSON-gated stack (preferred).
    ``use_layers=False``: legacy free-text sequential roles.
    """
    if use_layers and integrations is None:
        # Prefer dual_enter (KB aboutness + NLI + cert) unless custom roles subset
        if roles is None:
            from dual_enter import dual_enter

            r = dual_enter(
                prompt,
                base=base,
                model=model,
                embed=embed,
                retrieve_kb=True,
            )
            r.setdefault("mode", "lfm_orthogonal_ops")
            if "verdict" not in r:
                r["verdict"] = "NEED_INFO"
            return r
        from lms_layers import layered_enter

        r = layered_enter(
            prompt,
            base=base,
            model=model,
            roles=roles,
            embed=embed,
            ensure_residency=True,
            context_length=context_length,
        )
        r.setdefault("mode", "lfm_orthogonal_ops")
        if "verdict" not in r:
            r["verdict"] = "NEED_INFO"
        return r

    # ---- legacy free-text path ----
    lm = LMStudio(base)
    ens = lm.ensure_loaded(model, context_length=context_length)

    use_roles = roles or list(ROLES.keys())
    outputs: dict[str, Any] = {}
    chain_bits: list[str] = []
    resp_id = previous_response_id

    for role in use_roles:
        sys = ROLES.get(role, ROLES["SCOUT"])
        user = prompt
        if role == "VERDICT" and chain_bits:
            user = (
                f"INTENT:\n{prompt}\n\nPRIOR_ROLES:\n"
                + "\n---\n".join(chain_bits)
                + "\n\nGive VERDICT."
            )
        body_prev = resp_id if role != "SCOUT" and resp_id else None
        r = lm.chat_native(
            user,
            model=model,
            system=sys,
            temperature=0.1 if role in ("FALSIFY", "VERDICT") else 0.15,
            max_tokens=max_tokens,
            previous_response_id=body_prev,
        )
        if integrations and not r.get("ok"):
            r = lm.chat_native(
                user,
                model=model,
                system=sys,
                temperature=0.1,
                max_tokens=max_tokens,
                integrations=integrations,
                context_length=context_length,
            )

        outputs[role] = {
            "ok": r.get("ok"),
            "content": (r.get("content") or "")[:800],
            "stats": r.get("stats"),
            "response_id": r.get("response_id"),
            "error": r.get("error"),
            "tool_calls": r.get("tool_calls"),
        }
        if r.get("ok"):
            chain_bits.append(f"{role}: {(r.get('content') or '')[:400]}")
            if r.get("response_id"):
                resp_id = r["response_id"]

    verdict_text = (outputs.get("VERDICT") or {}).get("content") or ""
    verdict = _parse_verdict(verdict_text)

    emb: dict[str, Any] = {}
    if embed:
        he = lm.embed(prompt)  # Job1 jina default via nomic_metric
        if he.get("ok"):
            sims = {}
            for role, o in outputs.items():
                if not o.get("ok"):
                    continue
                ee = lm.embed(o.get("content") or "")
                if ee.get("ok"):
                    sims[role] = round(cosine(he["embedding"], ee["embedding"]), 4)
            emb = {
                "dim": he.get("dim"),
                "family": he.get("family"),
                "cosine_role_to_human": sims,
                "mean_cosine": round(sum(sims.values()) / max(len(sims), 1), 4) if sims else None,
                "not_agreement": True,
            }

    fatal = False
    fals = (outputs.get("FALSIFY") or {}).get("content") or ""
    if re.search(r"FATAL\s*=\s*yes", fals, re.I):
        fatal = True
        if verdict == "OPEN_CANDIDATE":
            verdict = "STOP"

    return {
        "ok": True,
        "mode": "lfm_orthogonal_ops_legacy",
        "thesis": (
            "One small model + orthogonal abstract operators beats multi-model cosplay. "
            "Structure is the force multiplier."
        ),
        "model": model,
        "roles": use_roles,
        "outputs": outputs,
        "verdict": verdict,
        "fatal_flag": fatal,
        "embeddings": emb,
        "last_response_id": resp_id,
        "regime": (
            "structured_ops"
            if all((outputs.get(r) or {}).get("ok") for r in use_roles)
            else "partial_ops"
        ),
        "not_open_authority": True,
        "sheaf_tag": "role_restriction_helper",
        "note": "MEASURE only. Prefer use_layers=True (JSON gates). OPEN still needs domain audit.",
        "load": ens,
    }
