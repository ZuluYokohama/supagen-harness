#!/usr/bin/env python3
"""Adversarial twin cos vs lexical overlap (Claude follow-up cut)."""
from __future__ import annotations

import json
import re
from pathlib import Path

SUM = Path(__file__).resolve().parent.parent / "state" / "bakeoff_30_summary.json"


def tokens(t: str) -> list[str]:
    t = re.sub(r"[^a-z0-9\s:]", " ", t.lower())
    stop = {"attacks", "and", "to", "the", "a", "an", "of", "is", "for", "with", "or", "then"}
    return [w for w in t.split() if w and w not in stop]


def jaccard(a: str, b: str) -> float:
    A, B = set(tokens(a)), set(tokens(b))
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def dice(a: str, b: str) -> float:
    A, B = set(tokens(a)), set(tokens(b))
    if not A and not B:
        return 1.0
    return 2 * len(A & B) / (len(A) + len(B)) if (A or B) else 0.0


def trigram_jaccard(a: str, b: str) -> float:
    def tri(s: str) -> set[str]:
        s = re.sub(r"\s+", " ", s.lower().strip())
        return {s[i : i + 3] for i in range(max(0, len(s) - 2))}

    Ta, Tb = tri(a), tri(b)
    u = Ta | Tb
    return len(Ta & Tb) / len(u) if u else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> int:
    s = json.loads(SUM.read_text(encoding="utf-8"))
    items = s["jina"]["items_meta"]
    pairs = s["jina"]["adversarial_separation"]["pairs"]
    nomic_map = {
        (p["a"], p["b"]): p["cos"]
        for p in s["nomic"]["adversarial_separation"]["pairs"]
    }

    rows = []
    print(f"{'pair':<32} {'jina':>6} {'nomic':>6} {'jac':>6} {'dice':>6} {'trig':>6}")
    for pr in pairs:
        ta = items[pr["a"]]["text"]
        tb = items[pr["b"]]["text"]
        jac = jaccard(ta, tb)
        dic = dice(ta, tb)
        tjac = trigram_jaccard(ta, tb)
        nc = nomic_map[(pr["a"], pr["b"])]
        shared = sorted(set(tokens(ta)) & set(tokens(tb)))
        rows.append(
            {
                "a": pr["a"],
                "b": pr["b"],
                "jina_cos": pr["cos"],
                "nomic_cos": nc,
                "jaccard": round(jac, 4),
                "dice": round(dic, 4),
                "trigram_jaccard": round(tjac, 4),
                "shared_tokens": shared,
            }
        )
        print(
            f"{pr['a']+'/'+pr['b']:<32} {pr['cos']:6.3f} {nc:6.3f} "
            f"{jac:6.3f} {dic:6.3f} {tjac:6.3f}"
        )

    cos = [r["jina_cos"] for r in rows]
    jac = [r["jaccard"] for r in rows]
    dic = [r["dice"] for r in rows]
    tri = [r["trigram_jaccard"] for r in rows]
    print()
    print("pearson jina_cos ~ jaccard :", round(pearson(cos, jac), 4))
    print("pearson jina_cos ~ dice    :", round(pearson(cos, dic), 4))
    print("pearson jina_cos ~ trigram :", round(pearson(cos, tri), 4))
    print(
        "jina adversarial cos min/max/spread:",
        min(cos),
        max(cos),
        round(max(cos) - min(cos), 4),
    )
    print("jaccard min/max:", min(jac), max(jac))

    # rank extremes with shared tokens
    by_cos = sorted(rows, key=lambda r: r["jina_cos"])
    print("\n--- best separation (lowest cos) ---")
    for r in by_cos[:3]:
        print(
            f"  {r['a']}/{r['b']}: cos={r['jina_cos']} jac={r['jaccard']} "
            f"shared={r['shared_tokens']}"
        )
    print("--- worst glue (highest cos) ---")
    for r in by_cos[-3:]:
        print(
            f"  {r['a']}/{r['b']}: cos={r['jina_cos']} jac={r['jaccard']} "
            f"shared={r['shared_tokens']}"
        )

    out = {
        "source": str(SUM),
        "n_pairs": len(rows),
        "pearson": {
            "jina_cos_jaccard": round(pearson(cos, jac), 4),
            "jina_cos_dice": round(pearson(cos, dic), 4),
            "jina_cos_trigram": round(pearson(cos, tri), 4),
        },
        "reading": (
            "If pearson ~ high, failures predictable from surface form; "
            "if low, arbitrary topic glue / polarity blindness."
        ),
        "pairs": rows,
    }
    out_path = SUM.parent / "bakeoff_adv_lexical.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
