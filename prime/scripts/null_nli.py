#!/usr/bin/env python3
"""
NLI null — A paraphrase / B negation / C unrelated.
After reason→label reorder: B should be contradiction if LFM can do the task.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from entailment_glue import glue_agreement  # noqa: E402

PAIRS = [
    (
        "A_paraphrase",
        "E_ref meets production readiness criteria under measured audit",
        "Under measured audit, E_ref satisfies criteria for production readiness",
        "entailment",
    ),
    (
        "B_negation",
        "E_ref is production-ready and certified OPEN",
        "E_ref is not production-ready; residue remains",
        "contradiction",
    ),
    (
        "C_unrelated",
        "E_ref sheaf certificate opens after lambda1 check",
        "Carbonara uses guanciale, egg, pecorino, and black pepper",
        "neutral",
    ),
]


def main() -> None:
    rows = []
    for name, prem, hyp, expect in PAIRS:
        r = glue_agreement(prem, hyp, prefer="lfm")
        rows.append({
            "pair": name,
            "expect": expect,
            "label": r.get("label"),
            "confidence": r.get("confidence"),
            "agrees": r.get("agrees"),
            "gate": r.get("gate"),
            "override": r.get("label_override"),
            "reason": (r.get("reason") or "")[:200],
            "ok": r.get("ok"),
            "error": r.get("error"),
            "hit": r.get("label") == expect,
        })
    n_hit = sum(1 for x in rows if x.get("hit"))
    b = next(x for x in rows if x["pair"] == "B_negation")
    report = {
        "ok": True,
        "schema": "reason_first",
        "hits": n_hit,
        "n": len(rows),
        "rows": rows,
        "B_is_contradiction": b.get("label") == "contradiction",
        "verdict": (
            "EARNED" if b.get("label") == "contradiction" and n_hit >= 2
            else ("PARTIAL" if b.get("label") == "contradiction" else "FAILED_NULL")
        ),
        "reading": (
            "B flipped to contradiction → field-order decidability gain."
            if b.get("label") == "contradiction"
            else "B still not contradiction → LFM-NLI cannot do Job 2; escalate to DeBERTa-MNLI."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
