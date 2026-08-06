"""
Text preparation for metrics — strip envelopes, reject placeholders.

Embedding serialized JSON spends vector mass on constant braces/keys
(verdict, reason, shared, …). Cosine then scores "both are JSON" not content.

NLI and aboutness should see **payload meaning only**.
"""
from __future__ import annotations

import json
import re
from typing import Any


# Law-core words that appear in our own system prompts — GLUE shared must not
# be *only* this set (constant function of the prompt).
LAW_CORE = frozenset({
    "open", "stop", "measure", "audit", "residue", "restrict", "certify",
    "need_info", "need-info", "open_candidate", "prior", "law",
})

# Field *descriptions* from the schema prompt — model emits slot labels, not instances
META_SLOTS = frozenset({
    "model names", "model name", "apis", "api", "file concepts", "file concept",
    "domain term", "domain terms", "gap", "gaps", "specific", "concrete",
    "interface terms", "interface term", "field concepts", "names",
})

# Scheduler / agent chrome that must never become "human intent" for metrics
CHROME_RE = re.compile(
    r"<system-reminder>[\s\S]*?</system-reminder>\s*"
    r"|Scheduled task\s+\S+\s*\(every[^)]*\)\.?[^\n]*\n?"
    r"|Earlier iterations, if any, appear above\.\s*"
    r"|Run the task below\.[^\n]*\n?"
    r"|End with a short status:[^\n]*\n?",
    re.I,
)


def strip_prompt_chrome(text: str) -> str:
    """Remove scheduler/system-reminder wrappers so roles/metrics see the task only."""
    t = text or ""
    t = CHROME_RE.sub("", t)
    # leftover tags
    t = re.sub(r"</?system-reminder>", "", t, flags=re.I)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


PLACEHOLDER_RE = re.compile(
    r"^(?:\.\.\.|…|\.{2,}|n/?a|tbd|todo|none|null|unknown|placeholder|fill.?in|"
    r"string|text|reason|note|risk|missing|\s*)$",
    re.I,
)
FILLER_RE = re.compile(
    r"(task is ongoing|monitoring required|smooth execution|ensure compliance|"
    r"further analysis|as appropriate|etc\.?$)",
    re.I,
)


def is_placeholder(s: Any) -> bool:
    if s is None:
        return True
    if isinstance(s, (list, dict)):
        if not s:
            return True
        if isinstance(s, list):
            return all(is_placeholder(x) for x in s)
        return all(is_placeholder(v) for v in s.values())
    t = str(s).strip()
    if len(t) < 2:
        return True
    if PLACEHOLDER_RE.match(t):
        return True
    if t in ("...", "…", '["..."]', "['...']"):
        return True
    return False


def is_filler_prose(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 12:
        return True
    if FILLER_RE.search(t) and len(t) < 120:
        return True
    # pure template
    if t.count("...") + t.count("…") >= 1 and len(t) < 40:
        return True
    return False


FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON|js|javascript)?\s*\n?([\s\S]*?)\n?```\s*$",
    re.M,
)


def strip_code_fences(text: str) -> str:
    """Ministral/agentic models wrap JSON in ```json fences — strip for parsers."""
    t = (text or "").strip()
    if not t:
        return t
    m = FENCE_RE.match(t)
    if m:
        return m.group(1).strip()
    # partial fence
    t = re.sub(r"^\s*```(?:json|JSON)?\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def parse_json_loose(text: str) -> Any | None:
    """Parse model JSON; tolerate fences and trailing junk."""
    t = strip_code_fences(text or "")
    if not t:
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", t)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate without model tokenizer (~4 chars/token English)."""
    t = text or ""
    if not t:
        return 0
    # whitespace-ish + punctuation split
    words = re.findall(r"\S+", t)
    by_words = int(len(words) * 1.3)
    by_chars = max(1, len(t) // 4)
    return max(by_words, by_chars)


def pack_to_token_budget(text: str, max_tokens: int = 3500, suffix: str = "\n…[capped]") -> str:
    """Hard-cap text to ~max_tokens for LMS loads that still thrash on long packs."""
    t = text or ""
    if estimate_tokens(t) <= max_tokens:
        return t
    # binary-ish shrink by chars
    budget_chars = max(200, max_tokens * 4 - len(suffix))
    if len(t) <= budget_chars:
        return t
    return t[: budget_chars - 1].rsplit(" ", 1)[0] + suffix


def strip_envelope(obj_or_text: Any) -> str:
    """
    Extract meaning-bearing strings for metrics.
    Prefer reason / note / what / risk / attacks body over raw JSON dump.
    """
    if obj_or_text is None:
        return ""
    if isinstance(obj_or_text, dict):
        return _from_dict(obj_or_text)
    t = strip_code_fences(str(obj_or_text).strip())
    if not t:
        return ""
    # try parse JSON envelope
    if t.startswith("{") or t.startswith("["):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return _from_dict(obj)
            if isinstance(obj, list):
                return " ".join(strip_envelope(x) for x in obj if not is_placeholder(x))
        except json.JSONDecodeError:
            # salvage
            m = re.search(r"\{[\s\S]*\}", t)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict):
                        return _from_dict(obj)
                except json.JSONDecodeError:
                    pass
    return t


def _from_dict(d: dict[str, Any]) -> str:
    # priority keys for semantic payload
    preferred = (
        "reason", "note", "what", "domain", "success", "risk",
        "hypothesis", "claim", "content", "text", "summary",
    )
    parts: list[str] = []
    for k in preferred:
        if k in d and not is_placeholder(d[k]):
            v = d[k]
            if isinstance(v, (list, dict)):
                parts.append(strip_envelope(v))
            else:
                parts.append(str(v).strip())
    # attacks / shared / missing as content lists (filter placeholders + pure law-core-only shared)
    if "attacks" in d and isinstance(d["attacks"], list):
        atk = [str(x).strip() for x in d["attacks"] if not is_placeholder(x)]
        if atk:
            parts.append("attacks: " + "; ".join(atk))
    if "shared" in d and isinstance(d["shared"], list):
        shared = [str(x).strip() for x in d["shared"] if not is_placeholder(x)]
        non_law = [x for x in shared if x.lower() not in LAW_CORE]
        # include shared only if there's something beyond law-core echo
        if non_law:
            parts.append("shared: " + ", ".join(shared))
        elif shared and not parts:
            # law-only shared with no other payload = empty for metric
            pass
    if "missing" in d and isinstance(d["missing"], list):
        miss = [str(x).strip() for x in d["missing"] if not is_placeholder(x)]
        if miss:
            parts.append("missing: " + ", ".join(miss))
    # verdict label alone is low content — only keep with reason
    if not parts and d.get("verdict") and not is_placeholder(d.get("reason")):
        parts.append(str(d.get("reason") or ""))
    text = " ".join(p for p in parts if p).strip()
    return text


def validate_role_payload(role: str, obj: dict[str, Any] | None) -> dict[str, Any]:
    """
    Reject schema-valid but contentless role outputs.
    Returns {ok, errors, warnings}.
    """
    if obj is None:
        return {"ok": False, "errors": ["null_payload"], "warnings": []}
    errors: list[str] = []
    warnings: list[str] = []

    def check_str(key: str, min_len: int = 8) -> None:
        v = obj.get(key)
        if is_placeholder(v):
            errors.append(f"placeholder:{key}")
        elif isinstance(v, str) and is_filler_prose(v):
            errors.append(f"filler:{key}")
        elif isinstance(v, str) and len(v.strip()) < min_len:
            errors.append(f"too_short:{key}")

    if role == "SCOUT":
        for k in ("what", "domain", "success"):
            check_str(k, 6)
    elif role == "FALSIFY":
        attacks = obj.get("attacks")
        if not isinstance(attacks, list) or len(attacks) < 1:
            errors.append("attacks_empty")
        else:
            real = [a for a in attacks if not is_placeholder(a) and len(str(a)) > 8]
            if len(real) < 1:
                errors.append("attacks_placeholder_only")
            if len(real) < 2:
                warnings.append("attacks_thin")
        # note optional but if present must not be filler-only
        if "note" in obj and is_placeholder(obj.get("note")):
            warnings.append("note_placeholder")
    elif role == "GLUE":
        shared = obj.get("shared") if isinstance(obj.get("shared"), list) else []
        missing = obj.get("missing") if isinstance(obj.get("missing"), list) else []
        shared_s = [str(x).strip() for x in shared if not is_placeholder(x)]
        missing_s = [str(x).strip() for x in missing if not is_placeholder(x)]
        # reject schema field-description echoes ("model names", "apis", …)
        meta_hits = [x for x in shared_s if x.lower() in META_SLOTS]
        if meta_hits:
            errors.append("shared_meta_slot_labels:" + ",".join(meta_hits[:5]))
            shared_s = [x for x in shared_s if x.lower() not in META_SLOTS]
        missing_s = [x for x in missing_s if x.lower() not in META_SLOTS]
        if not shared_s:
            errors.append("shared_empty_or_placeholder")
        else:
            non_law = [x for x in shared_s if x.lower() not in LAW_CORE]
            if not non_law:
                # constant function of the prompt — same costume as FALSIFY tautology
                errors.append("shared_only_law_core_echo")
        if not missing_s:
            errors.append("missing_empty_or_placeholder")
        check_str("risk", 8)
    elif role == "VERDICT":
        check_str("reason", 12)
        if is_filler_prose(str(obj.get("reason") or "")):
            errors.append("filler:reason")
        v = str(obj.get("verdict") or "").upper()
        if v not in ("OPEN_CANDIDATE", "STOP", "NEED_INFO", "OPEN"):
            errors.append("bad_verdict_label")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "payload_text": strip_envelope(obj),
    }
