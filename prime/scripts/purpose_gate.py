"""
Task-purpose gating — measure purpose must match claim type.

One embedder cannot serve two jobs. Generalization:
  every measure has a purpose; every claim requires a purpose;
  OPEN (audit) is refused on mismatch — type error, not a score.

purposes
--------
  RETRIEVE    jina Query:/Document: (nomic fallback search_*)
  TOPICALITY  jina/nomic aboutness cosine (diagnostic only)
  AGREEMENT   NLI channel only (never cosine)
  IDENTITY    exact symbol / jaccard (code/field)
  EXISTENCE   filesystem, tests, rplc, smoke

Would have caught glue_ok on day one: AGREEMENT claim + TOPICALITY instrument.
"""
from __future__ import annotations

from typing import Any, Iterable


# Canonical purpose tags
RETRIEVE = "RETRIEVE"
TOPICALITY = "TOPICALITY"
AGREEMENT = "AGREEMENT"
IDENTITY = "IDENTITY"
EXISTENCE = "EXISTENCE"

ALL = frozenset({RETRIEVE, TOPICALITY, AGREEMENT, IDENTITY, EXISTENCE})

# Claim kinds → required purposes (at least one must be present for audit OPEN)
CLAIM_REQUIRED: dict[str, frozenset[str]] = {
    "aboutness": frozenset({TOPICALITY, RETRIEVE}),
    "agreement": frozenset({AGREEMENT}),
    "same_claim": frozenset({AGREEMENT}),
    "code_symbol": frozenset({IDENTITY}),
    "field_id": frozenset({IDENTITY}),
    "exists": frozenset({EXISTENCE}),
    "tests_pass": frozenset({EXISTENCE}),
    "rplc": frozenset({EXISTENCE}),
    "work_item": frozenset({AGREEMENT, EXISTENCE, TOPICALITY}),  # deep-loop ACCEPTED needs honesty tags
    "production": frozenset({EXISTENCE, AGREEMENT}),  # audit OPEN bar
}


def normalize_purpose(p: str) -> str | None:
    u = (p or "").strip().upper()
    if u in ALL:
        return u
    aliases = {
        "RETRIEVAL": RETRIEVE,
        "ABOUTNESS": TOPICALITY,
        "TOPIC": TOPICALITY,
        "COSINE": TOPICALITY,
        "EMBED": TOPICALITY,
        "NLI": AGREEMENT,
        "ENTAILMENT": AGREEMENT,
        "GLUE": AGREEMENT,
        "JACCARD": IDENTITY,
        "SYMBOL": IDENTITY,
        "CODE": IDENTITY,
        "SMOKE": EXISTENCE,
        "TEST": EXISTENCE,
        "FS": EXISTENCE,
        "FILE": EXISTENCE,
    }
    return aliases.get(u)


def tag_measure(mode: str, **meta: Any) -> dict[str, Any]:
    """Attach purpose tags to a measure report."""
    mode_l = (mode or "").lower()
    purposes: list[str] = []
    if mode_l in ("lm_embed", "embed", "aboutness", "kb_query", "retrieval"):
        purposes = [TOPICALITY, RETRIEVE]
    elif mode_l in ("nli", "agreement", "glue_agreement", "dual_enter"):
        purposes = [AGREEMENT]
        if meta.get("also_aboutness"):
            purposes.append(TOPICALITY)
    elif mode_l in ("symbol", "jaccard", "interface"):
        purposes = [IDENTITY]
    elif mode_l in ("smoke", "rplc", "test", "measure_parallel", "eref"):
        purposes = [EXISTENCE]
    elif mode_l in ("modality_exchange", "modality_exchange_dual", "lfm_ops", "layered"):
        # exchange bundles — list what actually ran
        purposes = [AGREEMENT, TOPICALITY]
        if meta.get("domain_measures"):
            purposes.append(EXISTENCE)
    elif mode_l in ("project", "projection", "language", "align"):
        purposes = [IDENTITY, TOPICALITY]
    else:
        purposes = []
    return {
        "purposes": purposes,
        "mode": mode,
        "purpose_note": "type tags for gate — not scores",
    }


def purposes_from_measures(measures: Iterable[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for m in measures or []:
        if not isinstance(m, dict):
            continue
        for p in m.get("purposes") or []:
            n = normalize_purpose(str(p))
            if n:
                found.add(n)
        # infer from mode if untagged (legacy measures)
        if not m.get("purposes") and m.get("mode"):
            t = tag_measure(str(m.get("mode")))
            found.update(t["purposes"])
        # explicit agreement block
        if m.get("agreement") or m.get("nli_label") or (m.get("job") == "agreement_nli"):
            found.add(AGREEMENT)
        if m.get("mean_cosine") is not None or m.get("aboutness"):
            found.add(TOPICALITY)
        if m.get("jaccard") is not None or m.get("job") == "symbol_jaccard":
            found.add(IDENTITY)
    return found


def gate_claim(
    claim_kind: str,
    measures: Iterable[dict[str, Any]] | None = None,
    present_purposes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Refuse production OPEN when required purposes are missing.
    Returns {ok, gate, required, present, missing, error?}.
    """
    kind = (claim_kind or "production").lower()
    required = set(CLAIM_REQUIRED.get(kind, CLAIM_REQUIRED["production"]))
    present: set[str] = set()
    if present_purposes:
        for p in present_purposes:
            n = normalize_purpose(str(p))
            if n:
                present.add(n)
    if measures is not None:
        present |= purposes_from_measures(measures)
    missing = required - present
    if missing:
        return {
            "ok": False,
            "gate": "STOP",
            "claim_kind": kind,
            "required": sorted(required),
            "present": sorted(present),
            "missing": sorted(missing),
            "error": (
                f"purpose mismatch: claim needs {sorted(required)}, "
                f"have {sorted(present)}, missing {sorted(missing)}. "
                "AGREEMENT claims cannot OPEN on TOPICALITY (cosine) alone."
            ),
            "not_open_authority": True,
        }
    return {
        "ok": True,
        "gate": "PASS",
        "claim_kind": kind,
        "required": sorted(required),
        "present": sorted(present),
        "missing": [],
        "not_open_authority": True,
    }


def refuse_agreement_via_cosine(measure: dict[str, Any]) -> dict[str, Any] | None:
    """Detect classic failure: claim agreement using only cosine/aboutness."""
    if not measure:
        return None
    claims_agreement = any(
        x in str(measure.get("mode", "")).lower()
        for x in ("glue", "agree", "align", "exchange")
    ) or measure.get("glue_ok") is not None
    has_cos = measure.get("mean_cosine") is not None or (
        isinstance(measure.get("embeddings"), dict)
        and "mean_cosine" in (measure.get("embeddings") or {})
    )
    has_nli = bool(measure.get("agreement") or measure.get("nli_label"))
    purposes = set(measure.get("purposes") or [])
    if claims_agreement and has_cos and not has_nli and AGREEMENT not in purposes:
        return {
            "ok": False,
            "gate": "STOP",
            "error": "type error: AGREEMENT claim backed only by TOPICALITY (cosine)",
            "fix": "what can be wrong upstairs that couldn't downstairs? — cosine can't contradict",
        }
    return None
