#!/usr/bin/env python3
"""
Null test for the aboutness instrument (Job 1).

Pairs:
  A — two paraphrases of the same claim          → ceiling
  B — claim and its negation                     → contradiction blindness
  C — claim about E_ref vs claim about pasta     → floor (boilerplate baseline)

For each pair, score:
  1) envelope JSON (legacy contamination)
  2) stripped payload only
  3) stripped + family-correct task prefixes

Default family: jina (PRIME_EMBED_FAMILY). Use --family nomic|both to compare.

Measured targets (jina Query:/Document:): C-floor < 0.30, A−C range ~0.83.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metric_text import strip_envelope  # noqa: E402
from nomic_metric import (  # noqa: E402
    aboutness,
    apply_prefix,
    cosine,
    default_embed_model,
    embed,
    resolve_family,
)


# --- fixtures (same envelope shape as role outputs) ---
A1 = {
    "verdict": "OPEN_CANDIDATE",
    "reason": "E_ref meets production readiness criteria under measured audit",
}
A2 = {
    "verdict": "OPEN_CANDIDATE",
    "reason": "Under measured audit, E_ref satisfies criteria for production readiness",
}
B1 = {
    "verdict": "OPEN_CANDIDATE",
    "reason": "E_ref is production-ready and certified OPEN",
}
B2 = {
    "verdict": "STOP",
    "reason": "E_ref is not production-ready; residue remains",
}
C1 = {
    "verdict": "OPEN_CANDIDATE",
    "reason": "E_ref sheaf certificate opens after lambda1 check",
}
C2 = {
    "verdict": "OPEN_CANDIDATE",
    "reason": "Carbonara uses guanciale, egg, pecorino, and black pepper",
}
FILL = {
    "verdict": "OPEN_CANDIDATE",
    "reason": "Task is ongoing; monitoring required to ensure smooth execution.",
}


def score_pair(name: str, left: dict, right: dict, family: str) -> dict:
    env_l = json.dumps(left, ensure_ascii=False)
    env_r = json.dumps(right, ensure_ascii=False)
    strip_l = strip_envelope(left)
    strip_r = strip_envelope(right)
    model = default_embed_model(family)

    e1 = embed(env_l, task="none", family=family, model=model, use_cache=False)
    e2 = embed(env_r, task="none", family=family, model=model, use_cache=False)
    cos_env_raw = (
        cosine(e1["embedding"], e2["embedding"])
        if e1.get("ok") and e2.get("ok")
        else None
    )

    s1 = embed(strip_l, task="none", family=family, model=model, use_cache=False)
    s2 = embed(strip_r, task="none", family=family, model=model, use_cache=False)
    cos_strip_raw = (
        cosine(s1["embedding"], s2["embedding"])
        if s1.get("ok") and s2.get("ok")
        else None
    )

    ab = aboutness(
        strip_l,
        strip_r,
        a_task="search_query",
        b_task="search_document",
        family=family,
        model=model,
    )

    return {
        "pair": name,
        "family": resolve_family(family=family),
        "model": ab.get("model") or model,
        "base": ab.get("base"),
        "warning": ab.get("warning") or e1.get("warning") or s1.get("warning"),
        "strip_l": strip_l,
        "strip_r": strip_r,
        "prefix_preview_q": apply_prefix(strip_l, "search_query", family=family)[:60],
        "cos_envelope_no_prefix": round(cos_env_raw, 4) if cos_env_raw is not None else None,
        "cos_stripped_no_prefix": round(cos_strip_raw, 4) if cos_strip_raw is not None else None,
        "cos_stripped_prefixed": ab.get("cosine"),
        "ok": ab.get("ok"),
        "error": ab.get("error"),
    }


def run_family(family: str) -> dict:
    rows = [
        score_pair("A_paraphrase_ceiling", A1, A2, family),
        score_pair("B_negation_contradiction", B1, B2, family),
        score_pair("C_unrelated_floor", C1, C2, family),
        score_pair("FILL_vs_C2", FILL, C2, family),
        score_pair("FILL_vs_B1", FILL, B1, family),
    ]
    a = next(r for r in rows if r["pair"].startswith("A_"))
    c = next(r for r in rows if r["pair"].startswith("C_"))
    b = next(r for r in rows if r["pair"].startswith("B_"))

    def spread(key: str) -> float | None:
        av, cv = a.get(key), c.get(key)
        if av is None or cv is None:
            return None
        return round(float(av) - float(cv), 4)

    report: dict = {
        "ok": all(r.get("ok") is not False for r in rows if r["pair"].startswith(("A_", "B_", "C_"))),
        "family": resolve_family(family=family),
        "model": a.get("model"),
        "base": a.get("base"),
        "rows": rows,
        "dynamic_range_A_minus_C": {
            "envelope_no_prefix": spread("cos_envelope_no_prefix"),
            "stripped_no_prefix": spread("cos_stripped_no_prefix"),
            "stripped_prefixed": spread("cos_stripped_prefixed"),
        },
        "contradiction_gap_A_minus_B": {
            "stripped_prefixed": (
                round(
                    float(a["cos_stripped_prefixed"] or 0)
                    - float(b["cos_stripped_prefixed"] or 0),
                    4,
                )
                if a.get("cos_stripped_prefixed") is not None
                and b.get("cos_stripped_prefixed") is not None
                else None
            ),
        },
        "floor_C_prefixed": c.get("cos_stripped_prefixed"),
        "floor_lt_0_30": (
            c.get("cos_stripped_prefixed") is not None
            and float(c["cos_stripped_prefixed"]) < 0.30
        ),
        "verdict": None,
        "reading": [],
        "warning": next((r.get("warning") for r in rows if r.get("warning")), None),
    }

    if any(r.get("error") for r in rows if r["pair"].startswith(("A_", "C_"))):
        report["verdict"] = "INSTRUMENT_DOWN"
        report["reading"].append(
            f"Embed failed ({family}): "
            + (a.get("error") or c.get("error") or "unknown")
        )
        if family == "jina":
            report["reading"].append(
                "Start jina side server: "
                "python prime/scripts/start_jina_embed.py   "
                "or llama-server --embedding --port 8765 -m <v5-nano-F16.gguf>"
            )
        return report

    dr = report["dynamic_range_A_minus_C"]["stripped_prefixed"]
    if dr is not None and dr < 0.15:
        report["verdict"] = "MIRROR"
        report["reading"].append(
            f"Dynamic range A−C={dr} < 0.15 → aboutness instrument has almost no separation."
        )
    elif dr is not None:
        report["verdict"] = "HAS_RANGE"
        report["reading"].append(
            f"Dynamic range A−C={dr} on stripped+prefix — usable for aboutness only."
        )

    floor = report["floor_C_prefixed"]
    if floor is not None and float(floor) < 0.30:
        report["reading"].append(
            f"C-floor={floor} < 0.30 → real retrieval instrument (jina-class)."
        )
    elif floor is not None:
        report["reading"].append(
            f"C-floor={floor} ≥ 0.30 → compressed floor (nomic-class); usable but narrower."
        )

    cg = report["contradiction_gap_A_minus_B"]["stripped_prefixed"]
    if cg is not None and abs(cg) < 0.08:
        report["reading"].append(
            f"Contradiction gap A−B={cg} tiny → cosine blind to negation (need NLI)."
        )
    elif cg is not None:
        report["reading"].append(
            f"Contradiction gap A−B={cg} — still not a negation gate; DeBERTa stays Job2."
        )

    fill = next(r for r in rows if r["pair"].startswith("FILL_vs_C"))
    report["reading"].append(
        f"FILL vs pasta: env={fill.get('cos_envelope_no_prefix')} "
        f"strip+prefix={fill.get('cos_stripped_prefixed')} "
        "(filler must not pass content validator)."
    )
    if report.get("warning"):
        report["reading"].append(f"WARNING: {report['warning']}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Job1 aboutness A/B/C null")
    ap.add_argument(
        "--family",
        choices=("jina", "nomic", "both"),
        default=None,
        help="Embed family (default: PRIME_EMBED_FAMILY or jina)",
    )
    a = ap.parse_args()
    fam = a.family or resolve_family()

    if fam == "both" or a.family == "both":
        report = {
            "ok": True,
            "protocol": "A/B/C dual family",
            "jina": run_family("jina"),
            "nomic": run_family("nomic"),
        }
        # headline comparison
        jf = report["jina"].get("floor_C_prefixed")
        nf = report["nomic"].get("floor_C_prefixed")
        jr = report["jina"].get("dynamic_range_A_minus_C", {}).get("stripped_prefixed")
        nr = report["nomic"].get("dynamic_range_A_minus_C", {}).get("stripped_prefixed")
        report["headline"] = {
            "jina_floor_C": jf,
            "nomic_floor_C": nf,
            "jina_range_A_minus_C": jr,
            "nomic_range_A_minus_C": nr,
            "jina_floor_lt_0_30": report["jina"].get("floor_lt_0_30"),
        }
        report["ok"] = report["jina"].get("ok") or report["nomic"].get("ok")
    else:
        report = run_family(fam)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
