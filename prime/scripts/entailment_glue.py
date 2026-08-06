"""
Job 2 — AGREEMENT / GLUE (not aboutness).

Instrument: NLI over (premise, hypothesis) jointly.
  Labels: entailment | contradiction | neutral

Primary: LFM structured NLI — reason FIRST, label LAST (label is conclusion, not prior).
Optional: transformers CrossEncoder (DeBERTa-MNLI) if installed.

Null failure (pre-fix): 7/7 neutral; reason said "contradicts" while label=neutral
because label was sampled at token ~5 before reasoning. Fixed by field order.

Design law: glue is MEASURE only. OPEN still requires domain audit + certificate.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from lms_layers import DEFAULT_BASE, DEFAULT_LFM, l2_chat


Label = Literal["entailment", "contradiction", "neutral", "unknown"]

# reason → label → confidence: label is conditioned on analysis, not a prior
NLI_SYSTEM = (
    "ROLE=NLI. Strict natural language inference. "
    "Read PREMISE and HYPOTHESIS. "
    "First write a one-sentence analysis of the logical relation. "
    "Then pick exactly ONE label: entailment OR contradiction OR neutral. "
    "Output STRICT JSON with keys in this order only: "
    '{"reason":"<one sentence analysis>","label":"<one of: entailment, contradiction, neutral>","confidence":0.0} '
    "label must be a single word from that set — never a pipe list, never a template. "
    "confidence in [0,1]. "
    "If the hypothesis negates or denies the premise claim, label is contradiction. "
    "If the hypothesis restates the same claim, label is entailment. "
    "If topics differ with no logical link, label is neutral. "
    "Never invent evidence."
)

# Reject enum-copy / multi-label
_ENUM_COPY = re.compile(
    r"entailment\s*[\|/,]\s*contradiction|entailment\s*[\|/,]\s*neutral|"
    r"contradiction\s*[\|/,]\s*neutral|entailment\|contradiction\|neutral",
    re.I,
)
_CONTRA_HINT = re.compile(
    r"\b(contradict|denies|negates|opposite|incompatible|bypassing those restrictions|"
    r"not the same|directly opposite)\b",
    re.I,
)
_ENTAIL_HINT = re.compile(
    r"\b(same claim|same condition|restates|restate|paraphrase|equivalent|"
    r"follows from|supports the)\b",
    re.I,
)


def _parse_nli(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _norm_label(raw: Any) -> Label:
    s = str(raw or "").strip().lower().strip("\"'")
    # enum template copied as value
    if _ENUM_COPY.search(s) or "|" in s or s.count("entailment") + s.count("contradiction") > 1:
        return "unknown"
    if s in ("entailment", "entails", "entailed"):
        return "entailment"
    if s in ("contradiction", "contradicts", "contradict"):
        return "contradiction"
    if s in ("neutral", "unrelated"):
        return "neutral"
    if "contradict" in s:
        return "contradiction"
    if "entail" in s or s in ("yes", "true", "supports"):
        return "entailment"
    if "neutral" in s or "unknown" in s:
        return "neutral"
    return "unknown"


def _reconcile_label_with_reason(label: Label, reason: str) -> tuple[Label, str | None]:
    """
    If reason clearly says contradicts but label is neutral, trust the reason text.
    That was the measured failure mode: analysis right, label prior wrong.
    """
    r = reason or ""
    if label in ("neutral", "unknown") and _CONTRA_HINT.search(r):
        return "contradiction", "label_overridden_by_reason_contradict_hint"
    if label in ("neutral", "unknown") and _ENTAIL_HINT.search(r) and not _CONTRA_HINT.search(r):
        return "entailment", "label_overridden_by_reason_entail_hint"
    return label, None


def nli_lfm(
    premise: str,
    hypothesis: str,
    model: str = DEFAULT_LFM,
    base: str = DEFAULT_BASE,
    max_tokens: int = 120,
) -> dict[str, Any]:
    """
    Joint NLI via LFM. reason-first schema so label is conditioned on analysis.
    Small context_length — pairs are tiny; 4096 is waste on CPU.
    """
    user = (
        f"PREMISE:\n{(premise or '')[:1800]}\n\n"
        f"HYPOTHESIS:\n{(hypothesis or '')[:800]}\n\n"
        "Write reason first, then a single label, then confidence. JSON only."
    )
    r = l2_chat(
        user,
        model=model,
        system=NLI_SYSTEM,
        temperature=0.0,
        max_tokens=max_tokens,
        store=False,
        context_length=2048,  # per-role sizing
        base=base,
    )
    if not r.get("ok"):
        return {
            "ok": False,
            "job": "agreement_nli",
            "engine": "lfm_nli",
            "error": r.get("error"),
            "label": "unknown",
            "agrees": False,
            "gate": "NEED_INFO",
        }
    raw = r.get("content") or ""
    obj = _parse_nli(raw)
    if not obj:
        return {
            "ok": False,
            "job": "agreement_nli",
            "engine": "lfm_nli",
            "error": "unparseable_nli_json",
            "raw": raw[:400],
            "label": "unknown",
            "agrees": False,
            "gate": "NEED_INFO",
        }
    reason = str(obj.get("reason") or "")
    label = _norm_label(obj.get("label"))
    label, override = _reconcile_label_with_reason(label, reason)
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    # enum-copy → NEED_INFO
    if label == "unknown":
        return {
            "ok": False,
            "job": "agreement_nli",
            "engine": "lfm_nli",
            "error": "bad_or_enum_copy_label",
            "raw_label": obj.get("label"),
            "reason": reason[:300],
            "label": "unknown",
            "agrees": False,
            "gate": "NEED_INFO",
            "raw": raw[:400],
        }
    if label == "entailment" and conf >= 0.4:
        gate = "PASS"
        agrees = True
    elif label == "contradiction":
        gate = "STOP"
        agrees = False
    else:
        gate = "NEED_INFO"
        agrees = False
    return {
        "ok": True,
        "job": "agreement_nli",
        "engine": "lfm_nli",
        "schema": "reason_first",
        "label": label,
        "confidence": round(conf, 3),
        "reason": reason[:300],
        "label_override": override,
        "agrees": agrees,
        "gate": gate,
        "cost": r.get("cost"),
        "not_open_authority": True,
        "note": (
            "reason→label ordering. Agreement measure only. "
            "If null still fails after this, escalate to DeBERTa-MNLI cross-encoder."
        ),
    }


def nli_cross_encoder(
    premise: str,
    hypothesis: str,
    model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
) -> dict[str, Any]:
    """Optional true cross-encoder if sentence-transformers is installed."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        return {
            "ok": False,
            "job": "agreement_nli",
            "engine": "cross_encoder",
            "error": f"sentence_transformers unavailable: {e}",
            "label": "unknown",
            "agrees": False,
            "gate": "NEED_INFO",
        }
    try:
        import numpy as np

        ce = CrossEncoder(model_name)
        scores = ce.predict([(premise[:1500], hypothesis[:500])])
        s = np.array(scores).reshape(-1)
        if s.size == 3:
            labels = ["contradiction", "entailment", "neutral"]
            i = int(s.argmax())
            label = _norm_label(labels[i])
            ex = np.exp(s - s.max())
            conf = float(ex[i] / ex.sum())
        elif s.size == 1:
            conf = float(1 / (1 + np.exp(-s[0])))
            label = (
                "entailment" if conf > 0.55
                else ("contradiction" if conf < 0.45 else "neutral")
            )
        else:
            return {
                "ok": False,
                "engine": "cross_encoder",
                "error": f"unexpected score shape {s.shape}",
                "label": "unknown",
                "agrees": False,
                "gate": "NEED_INFO",
            }
        agrees = label == "entailment" and conf >= 0.45
        gate = "PASS" if agrees else ("STOP" if label == "contradiction" else "NEED_INFO")
        return {
            "ok": True,
            "job": "agreement_nli",
            "engine": "cross_encoder",
            "model": model_name,
            "label": label,
            "confidence": round(conf, 3),
            "agrees": agrees,
            "gate": gate,
            "not_open_authority": True,
        }
    except Exception as e:
        return {
            "ok": False,
            "job": "agreement_nli",
            "engine": "cross_encoder",
            "error": str(e)[:400],
            "label": "unknown",
            "agrees": False,
            "gate": "NEED_INFO",
        }


def glue_agreement(
    human: str,
    domain: str,
    prefer: str = "lfm",
    base: str = DEFAULT_BASE,
) -> dict[str, Any]:
    """
    Job 2 entry: does domain stalk *agree* with human intent (entailment).
    prefer: lfm | cross_encoder | auto
    """
    from metric_text import strip_envelope, strip_prompt_chrome

    human = strip_prompt_chrome(strip_envelope(human) if (human or "").strip().startswith("{") else (human or ""))
    domain = strip_envelope(domain) if domain else ""
    if prefer in ("cross_encoder", "auto"):
        r = nli_cross_encoder(human, domain)
        if r.get("ok") or prefer == "cross_encoder":
            return r
    return nli_lfm(human, domain, base=base)


def interface_jaccard(a: str, b: str, symbols: list[str] | None = None) -> dict[str, Any]:
    """Exact-ish symbol glue for code/field domains."""
    default_syms = symbols or [
        "open", "stop", "measure", "audit", "residue", "restrict", "certify",
        "sheaf", "lambda", "λ1", "e_ref", "opened_steps", "need_info",
        "rplc", "frustration", "h0", "coboundary",
    ]
    ta = (a or "").lower()
    tb = (b or "").lower()
    tok_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
    sa = set(tok_re.findall(a or "")) | {s for s in default_syms if s in ta}
    sb = set(tok_re.findall(b or "")) | {s for s in default_syms if s in tb}
    sa = {x.lower() for x in sa}
    sb = {x.lower() for x in sb}
    if not sa or not sb:
        return {"ok": True, "job": "symbol_jaccard", "jaccard": 0.0, "shared": [], "note": "empty"}
    inter = sa & sb
    union = sa | sb
    j = len(inter) / max(len(union), 1)
    return {
        "ok": True,
        "job": "symbol_jaccard",
        "jaccard": round(j, 4),
        "n_shared": len(inter),
        "shared": sorted(inter)[:40],
        "note": "Code/field: prefer over aboutness cosine for interface glue",
    }


def dual_measure(
    human: str,
    domain: str,
    domain_kind: str = "language",
    base: str = DEFAULT_BASE,
    aboutness_mean: list[float] | None = None,
) -> dict[str, Any]:
    """Dual: aboutness (jina/nomic) diagnostic + agreement (NLI) + optional symbol jaccard."""
    from nomic_metric import aboutness

    about = aboutness(
        human,
        domain,
        a_task="search_query",
        b_task="search_document",
        mean=aboutness_mean,
        base=base,
    )
    agree = glue_agreement(human, domain, prefer="lfm", base=base)
    sym = None
    if domain_kind.lower() in ("code", "field", "rplc", "eref"):
        sym = interface_jaccard(human, domain)

    if agree.get("label") == "contradiction":
        gate = "STOP"
    elif agree.get("agrees"):
        gate = "PASS"
    else:
        gate = "NEED_INFO"

    return {
        "ok": True,
        "job": "dual_measure",
        "aboutness": about,
        "agreement": agree,
        "symbol": sym,
        "gate": gate,
        "agrees": bool(agree.get("agrees")),
        "thesis": (
            "Nomic = aboutness chart. NLI reason→label = agreement attempt. "
            "Certificate = logogram. Cosine never equals entailment."
        ),
        "not_open_authority": True,
    }
