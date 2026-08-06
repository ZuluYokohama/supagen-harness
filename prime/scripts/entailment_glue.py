"""
Job 2 — AGREEMENT / GLUE (not aboutness).

Instrument: NLI over (premise, hypothesis) jointly.
  Labels: entailment | contradiction | neutral

Primary: DeBERTa cross-encoder NLI (PRIME_NLI_MODEL, default nli-deberta-v3-base).
Fallback: LFM structured NLI — reason FIRST, label LAST (label is conclusion, not prior).
Mutual entailment (p≥PRIME_NLI_MUTUAL_P, default 0.80) for strong agreement.

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


# Job2 default: DeBERTa-v3-base MNLI (cached on this kit; large multi-NLI is upgrade path)
import os as _os

DEFAULT_NLI_MODEL = _os.environ.get(
    "PRIME_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
)
def _env_float(name: str, default: float) -> float:
    """Parse env float; invalid/missing → default (never break import)."""
    raw = _os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# Mutual entailment confidence floor for "agrees" (dual direction)
MUTUAL_P = _env_float("PRIME_NLI_MUTUAL_P", 0.80)
# One-way entailment floor — must match accel_nli_ort.ONEWAY_P
ONEWAY_P = _env_float("PRIME_NLI_ONEWAY_P", 0.45)

_CE_CACHE: dict[str, Any] = {}
_CE_GLOBAL = __import__("threading").Lock()
_CE_MODEL_LOCKS: dict[str, Any] = {}


def _get_cross_encoder(model_name: str):
    """Load CrossEncoder with per-model lock so different models can load concurrently."""
    # Fast path: cache hit without any lock contention on unrelated models
    if model_name in _CE_CACHE:
        return _CE_CACHE[model_name]
    with _CE_GLOBAL:
        lock = _CE_MODEL_LOCKS.get(model_name)
        if lock is None:
            lock = __import__("threading").Lock()
            _CE_MODEL_LOCKS[model_name] = lock
    with lock:
        if model_name in _CE_CACHE:
            return _CE_CACHE[model_name]
        from sentence_transformers import CrossEncoder  # type: ignore

        ce = CrossEncoder(model_name)
        _CE_CACHE[model_name] = ce
        return ce


def nli_cross_encoder(
    premise: str,
    hypothesis: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """True cross-encoder NLI (DeBERTa-MNLI). Owns agreement; not aboutness."""
    model_name = model_name or DEFAULT_NLI_MODEL
    try:
        from sentence_transformers import CrossEncoder  # type: ignore  # noqa: F401
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

        ce = _get_cross_encoder(model_name)
        scores = ce.predict([(premise[:1500], hypothesis[:500])])
        s = np.array(scores).reshape(-1)
        # sentence-transformers NLI models: typically [contradiction, entailment, neutral]
        # Verify via config when possible
        default_labels = ["contradiction", "entailment", "neutral"]
        labels = list(default_labels)
        label_source = "default_assumed"
        try:
            id2label = getattr(getattr(ce, "model", None), "config", None)
            id2label = getattr(id2label, "id2label", None) if id2label else None
            if isinstance(id2label, dict) and len(id2label) >= 3:
                # Coerce string keys "0"/"1"/"2" → int (same as accel_nli_ort)
                coerced: dict[int, str] = {}
                for k, v in id2label.items():
                    try:
                        coerced[int(k)] = str(v).lower()
                    except (TypeError, ValueError):
                        continue
                if len(coerced) >= 3:
                    labels = [coerced[i] for i in range(3)]
                    label_source = "model.config.id2label"
        except Exception as e:
            label_source = f"default_after_error:{type(e).__name__}"
        if s.size == 3:
            i = int(s.argmax())
            label = _norm_label(labels[i] if i < len(labels) else "neutral")
            ex = np.exp(s - s.max())
            conf = float(ex[i] / ex.sum())
            probs = {
                _norm_label(labels[j] if j < len(labels) else str(j)): round(
                    float(ex[j] / ex.sum()), 4
                )
                for j in range(3)
            }
        elif s.size == 1:
            # Single-logit binary models: avoid double-sigmoid. If score already in
            # [0,1], treat as p(entail); else apply one sigmoid to raw logit.
            raw = float(s[0])
            if 0.0 <= raw <= 1.0:
                p_ent = raw
                label_source = label_source + "+single_logit_prob"
            else:
                p_ent = float(1.0 / (1.0 + np.exp(-raw)))
                label_source = label_source + "+single_logit_sigmoid"
            # Thresholds coupled to ONEWAY_P only (not a free 0.55 constant)
            if p_ent >= ONEWAY_P:
                label = "entailment"
                conf = p_ent
            elif p_ent <= (1.0 - ONEWAY_P):
                label = "contradiction"
                conf = 1.0 - p_ent
            else:
                label = "neutral"
                conf = 1.0 - abs(2.0 * p_ent - 1.0)
            # Binary contract: report entail vs not-entail (not a fake 3-class dist)
            probs = {
                "entailment": round(p_ent, 4),
                "contradiction": round(1.0 - p_ent, 4),
            }
        else:
            return {
                "ok": False,
                "engine": "cross_encoder",
                "error": f"unexpected score shape {s.shape}",
                "label": "unknown",
                "agrees": False,
                "gate": "NEED_INFO",
            }
        # Gate: entailment needs calibrated conf (ONEWAY_P; mutual uses MUTUAL_P)
        agrees = label == "entailment" and conf >= ONEWAY_P
        gate = "PASS" if agrees else ("STOP" if label == "contradiction" else "NEED_INFO")
        return {
            "ok": True,
            "job": "agreement_nli",
            "engine": "cross_encoder",
            "model": model_name,
            "label": label,
            "confidence": round(conf, 3),
            "probs": probs,
            "agrees": agrees,
            "gate": gate,
            "label_source": label_source,
            "oneway_p": ONEWAY_P,
            "not_open_authority": True,
            "job2_owns_open": False,
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


def _nli_one_way(
    premise: str,
    hypothesis: str,
    *,
    prefer: str = "auto",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Single-direction NLI using same engine order as glue_agreement."""
    if prefer in ("ort", "auto") and _os.environ.get("PRIME_NLI_ORT", "1").strip() not in (
        "0",
        "false",
        "no",
    ):
        r = nli_ort(premise, hypothesis, force_cpu=True)
        if r.get("ok"):
            return r
        if prefer == "ort":
            return r
    return nli_cross_encoder(premise, hypothesis, model_name=model_name)


def mutual_entailment(
    a: str,
    b: str,
    model_name: str | None = None,
    p_floor: float | None = None,
    prefer: str = "auto",
) -> dict[str, Any]:
    """
    Bidirectional entailment: both a→b and b→a must be entailment with conf ≥ p_floor.
    Uses the same engine selection as glue_agreement (ORT→CE under auto).
    Never OPEN authority alone.
    """
    p_floor = MUTUAL_P if p_floor is None else p_floor
    ab = _nli_one_way(a, b, prefer=prefer, model_name=model_name)
    ba = _nli_one_way(b, a, prefer=prefer, model_name=model_name)
    if not ab.get("ok") or not ba.get("ok"):
        return {
            "ok": False,
            "job": "mutual_entailment",
            "error": ab.get("error") or ba.get("error"),
            "ab": ab,
            "ba": ba,
            "agrees": False,
            "gate": "NEED_INFO",
            "not_open_authority": True,
        }
    ab_e = ab.get("label") == "entailment" and float(ab.get("confidence") or 0) >= p_floor
    ba_e = ba.get("label") == "entailment" and float(ba.get("confidence") or 0) >= p_floor
    contra = ab.get("label") == "contradiction" or ba.get("label") == "contradiction"
    agrees = bool(ab_e and ba_e)
    if contra and not agrees:
        gate = "STOP"
    elif agrees:
        gate = "PASS"
    else:
        gate = "NEED_INFO"
    return {
        "ok": True,
        "job": "mutual_entailment",
        "model": ab.get("model"),
        "p_floor": p_floor,
        "ab": {
            "label": ab.get("label"),
            "confidence": ab.get("confidence"),
            "probs": ab.get("probs"),
        },
        "ba": {
            "label": ba.get("label"),
            "confidence": ba.get("confidence"),
            "probs": ba.get("probs"),
        },
        "agrees": agrees,
        "gate": gate,
        "min_entail_conf": round(
            min(
                float(ab.get("confidence") or 0) if ab.get("label") == "entailment" else 0.0,
                float(ba.get("confidence") or 0) if ba.get("label") == "entailment" else 0.0,
            ),
            3,
        ),
        "not_open_authority": True,
        "note": "Mutual entailment owns agreement; aboutness must not promote OPEN.",
    }


def nli_ort(
    premise: str,
    hypothesis: str,
    *,
    force_cpu: bool = True,
) -> dict[str, Any]:
    """ONNX Runtime DeBERTa NLI.

    Product/auto path: force_cpu=True (CPUExecutionProvider only).
    Explicit non-CPU requires force_cpu=False and PRIME_ACCEL opt-in.
    """
    fail = {
        "ok": False,
        "job": "agreement_nli",
        "engine": "ort_nli",
        "error": None,
        "label": "unknown",
        "agrees": False,
        "gate": "NEED_INFO",
        "not_open_authority": True,
        "job2_owns_open": False,
        "force_cpu": force_cpu,
    }
    try:
        from accel_nli_ort import predict

        r = predict(premise, hypothesis, force_cpu=force_cpu)
        if not r.get("ok"):
            fail["error"] = str(r.get("error") or "ort_nli_failed")[:300]
            return fail
        # Success path — ensure law flags always present
        out = dict(r)
        out.setdefault("job", "agreement_nli")
        out.setdefault("not_open_authority", True)
        out["job2_owns_open"] = False
        out["force_cpu"] = force_cpu
        # Belt-and-suspenders: refuse QNN on product path
        prov = str(out.get("provider") or "")
        if force_cpu and "QNN" in prov:
            fail["error"] = f"refused QNN on product Job2 path: {prov}"
            return fail
        return out
    except Exception as e:
        fail["error"] = str(e)[:300]
        return fail


def glue_agreement(
    human: str,
    domain: str,
    prefer: str = "auto",
    base: str = DEFAULT_BASE,
    fiber_mode: str | None = None,
) -> dict[str, Any]:
    """
    Job 2 entry: does domain stalk *agree* with human intent (entailment).
    prefer: auto (ORT→DeBERTa CE→LFM) | ort | cross_encoder | lfm | mutual | htp
    fiber_mode: optional request fiber (preserve|scout); overrides PRIME_FIBER_MODE
    for LFM scout-only guard when dual_enter passes an explicit mode.

    HTP is never first on auto. prefer=htp only runs when measure_fabric
    nli_htp_parity_pass() is green; otherwise falls through to ORT/CE.
    Never authorizes production OPEN.
    """
    from metric_text import strip_envelope, strip_prompt_chrome

    def _clean(s: str) -> str:
        s = s or ""
        if s.strip().startswith("{"):
            s = strip_envelope(s)
        return strip_prompt_chrome(s)

    # Symmetric chrome strip on both stalks before agreement measure
    human = _clean(human)
    domain = _clean(domain)
    if prefer == "mutual":
        return mutual_entailment(human, domain, prefer="auto")

    # Explicit HTP only after E3 parity cert (session-ready QDQ is not enough)
    if prefer in ("htp", "hexagon", "npu"):
        try:
            from measure_fabric import nli_htp_parity_pass

            gate = nli_htp_parity_pass()
        except Exception as e:
            gate = {"ok": False, "reason": str(e)}
        if not gate.get("ok"):
            # refuse HTP — fall through to CPU authority
            prefer = "auto"
            # annotate path taken after CPU result
            _htp_refused = gate
        else:
            _htp_refused = None
            # Product HTP path not implemented until E3 green; still refuse force-OPEN
            # by falling through to ORT (measure fabric documents order only).
            prefer = "auto"
    else:
        _htp_refused = None

    def _annotate(r: dict[str, Any]) -> dict[str, Any]:
        out = dict(r)
        out["job2_owns_open"] = False
        out.setdefault("not_open_authority", True)
        if _htp_refused is not None:
            out["htp_refused"] = _htp_refused
        return out

    known = {
        "auto",
        "ort",
        "cross_encoder",
        "lfm",
        "mutual",
        "htp",
        "hexagon",
        "npu",
    }
    # After HTP remap, prefer may be "auto"; unknown values never reach LFM
    if prefer not in known:
        return _annotate(
            {
                "ok": False,
                "job": "agreement_nli",
                "engine": "none",
                "error": f"unknown prefer={prefer!r}; allowed={sorted(known)}",
                "label": "unknown",
                "agrees": False,
                "gate": "NEED_INFO",
            }
        )

    if prefer == "ort":
        return _annotate(nli_ort(human, domain, force_cpu=True))
    if prefer in ("cross_encoder", "auto"):
        # Prefer ORT CPU when model exported; fall back CE
        if prefer == "auto" and _os.environ.get("PRIME_NLI_ORT", "1").strip() not in (
            "0",
            "false",
            "no",
        ):
            r_ort = nli_ort(human, domain, force_cpu=True)
            if r_ort.get("ok"):
                return _annotate(r_ort)
        r = nli_cross_encoder(human, domain)
        if r.get("ok") or prefer == "cross_encoder":
            return _annotate(r)
        # prefer=auto only continues to LFM scout fallback

    # prefer=lfm explicit, or prefer=auto after ORT+CE exhaustion — SCOUT only
    if prefer not in ("auto", "lfm"):
        return _annotate(
            {
                "ok": False,
                "job": "agreement_nli",
                "engine": "none",
                "error": f"prefer={prefer!r} produced no usable agreement path",
                "label": "unknown",
                "agrees": False,
                "gate": "NEED_INFO",
            }
        )
    fiber = (fiber_mode or _os.environ.get("PRIME_FIBER_MODE") or "scout").lower().strip()
    if fiber != "scout":
        return _annotate(
            {
                "ok": False,
                "job": "agreement_nli",
                "engine": "lfm_nli",
                "error": f"lfm_nli refused in fiber_mode={fiber} (scout only)",
                "label": "unknown",
                "agrees": False,
                "gate": "NEED_INFO",
            }
        )
    return _annotate(nli_lfm(human, domain, base=base))


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
    agree = glue_agreement(human, domain, prefer="auto", base=base)
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
