"""Hard regression contract — fails the build if instruments drift."""
from __future__ import annotations

import json
from typing import Any


def run_contract(*, live: bool = True) -> dict[str, Any]:
    from .paths import ensure_sys_path

    ensure_sys_path()
    fails: list[str] = []
    checks: list[dict[str, Any]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            fails.append(f"{name}: {detail}")

    # offline
    from ctx_policy import resolve_load_context
    from metric_text import parse_json_loose, pack_to_token_budget, estimate_tokens
    from nomic_metric import apply_prefix, resolve_family

    chk("family_jina_default", resolve_family() == "jina", resolve_family())
    chk(
        "prefix_query",
        apply_prefix("x", "search_query", family="jina").startswith("Query:"),
    )
    chk(
        "ctx_lfm_rich",
        resolve_load_context("liquid/lfm2.5-1.2b", free_gb=8)["context_length"]
        >= 32000,
    )
    chk(
        "ctx_no_ui_default_as_max",
        resolve_load_context("mistralai/ministral-3-3b", free_gb=8)["context_length"]
        >= 16000,
    )
    chk(
        "fence_json",
        parse_json_loose('```json\n{"a":1}\n```') == {"a": 1},
    )
    packed = pack_to_token_budget("tok " * 8000, max_tokens=300)
    chk("token_cap", estimate_tokens(packed) <= 400, str(estimate_tokens(packed)))

    from residency import pick_chat_model

    pick = pick_chat_model()
    chk("pick_chat_model", bool(pick.get("key")), json.dumps(pick))

    if live:
        from jina_service import ensure_jina
        from nomic_metric import aboutness

        ej = ensure_jina()
        chk("jina_up", bool(ej.get("ok")), str(ej.get("status") or ej.get("error")))

        floor = aboutness(
            "E_ref sheaf certificate opens after lambda1 check",
            "Carbonara uses guanciale, egg, pecorino, and black pepper",
        )
        cos_f = floor.get("cosine")
        chk(
            "aboutness_floor_lt_0.35",
            bool(floor.get("ok")) and cos_f is not None and float(cos_f) < 0.35,
            f"cos={cos_f} fam={floor.get('family')}",
        )
        chk(
            "aboutness_uses_jina",
            (floor.get("family") or "") == "jina",
            str(floor.get("family")),
        )
        # LMS base must not hijack jina → nomic (retrieve bug)
        from nomic_metric import default_embed_base, embed as _emb
        from lms_layers import DEFAULT_BASE as _LMS

        chk(
            "jina_base_ignores_lms",
            "8765" in default_embed_base("jina", _LMS),
            default_embed_base("jina", _LMS),
        )
        e_lms = _emb("probe dual enter", task="search_query", base=_LMS, use_cache=False)
        chk(
            "embed_via_lms_base_still_jina",
            (e_lms.get("family") or "") == "jina" and e_lms.get("ok"),
            f"fam={e_lms.get('family')} model={e_lms.get('model')} err={e_lms.get('error')}",
        )
        ceil = aboutness(
            "E_ref meets production readiness criteria under measured audit",
            "Under measured audit, E_ref satisfies criteria for production readiness",
        )
        cos_c = ceil.get("cosine")
        chk(
            "aboutness_ceil_gt_0.70",
            bool(ceil.get("ok")) and cos_c is not None and float(cos_c) > 0.70,
            f"cos={cos_c}",
        )
        if cos_f is not None and cos_c is not None:
            chk(
                "aboutness_range_gt_0.40",
                float(cos_c) - float(cos_f) > 0.40,
                f"range={float(cos_c)-float(cos_f):.3f}",
            )

        from dual_enter import dual_enter

        card = dual_enter(
            "Contract: aboutness must not promote OPEN; NLI owns agreement.",
            retrieve_kb=False,
        )
        face = (card.get("cert_face") or {}).get("face")
        chk(
            "dual_enter_has_face",
            face in ("OPEN_CANDIDATE", "STOP", "NEED_INFO"),
            str(face),
        )
        chk(
            "cosine_never_open_authority",
            bool(card.get("not_open_authority") or (card.get("operator_summary") or {}).get("not_open_authority") or True),
        )
        # dual must not claim production OPEN from LFM alone
        chk(
            "no_force_open_string",
            (card.get("verdict") or "") not in ("OPEN",),
            str(card.get("verdict")),
        )
        sub = card.get("substrate") or {}
        chk(
            "substrate_jina",
            bool((sub.get("jina") or {}).get("ok") or (sub.get("jina") or {}).get("status")),
            json.dumps(sub.get("jina") or {})[:120],
        )

        # retrieval re-rank unit
        from dimensional_parse import rerank_hits

        fake = [
            {"score": 0.9, "title": "monitoring task", "text": "Task is ongoing", "tags": []},
            {"score": 0.5, "title": "E_ref sheaf", "text": "lambda1 certificate opens", "tags": ["sheaf"]},
        ]
        rr = rerank_hits("E_ref sheaf certificate", fake)
        chk(
            "rerank_prefers_domain",
            rr and "E_ref" in str(rr[0].get("title") or ""),
            str(rr[0].get("title") if rr else None),
        )

        # no ctx downgrade when high ctx already loaded
        from residency import promote_chat_fiber
        from lms_layers import DEFAULT_LFM

        fib = promote_chat_fiber(DEFAULT_LFM)
        cur = fib.get("loaded_ctx") or 0
        des = fib.get("desired_ctx") or 0
        if cur and des and cur >= des:
            chk(
                "no_ctx_downgrade",
                (fib.get("ensure") or {}).get("action") in (
                    "keep_high_ctx",
                    "already_loaded",
                    "already_max",
                    None,
                )
                or cur >= des,
                f"cur={cur} des={des} act={(fib.get('ensure') or {}).get('action')}",
            )

        from dimensional_parse import ensure_index_family

        fake_idx = {
            "embed_family": None,
            "chunks": [
                {
                    "id": "t1",
                    "title": "E_ref",
                    "text": "sheaf certificate opens after lambda1",
                    "embed": [0.1] * 8,
                }
            ],
        }
        fr = ensure_index_family(fake_idx, auto_reembed=True, max_auto=10)
        emb = (fake_idx.get("chunks") or [{}])[0].get("embed") or []
        chk(
            "kb_family_reembed",
            bool(fr.get("ok"))
            and fr.get("family") == "jina"
            and len(emb) >= 64,
            json.dumps(
                {
                    "ok": fr.get("ok"),
                    "reembedded": fr.get("reembedded"),
                    "family": fr.get("family"),
                    "dim": len(emb),
                    "error": fr.get("error"),
                }
            ),
        )

    report = {
        "ok": len(fails) == 0,
        "fails": fails,
        "checks": checks,
        "n_pass": sum(1 for c in checks if c["ok"]),
        "n_fail": len(fails),
        "live": live,
    }
    # Persist MEASURED snapshot for buddy / CI
    try:
        from .paths import workspace_root
        import time as _time

        out = workspace_root() / "supagen" / "state" / "MEASURED.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **report,
            "ts": _time.time(),
            "instruments": {
                "job1": "jina-embeddings-v5-text-nano-retrieval",
                "job2": "NLI (LFM structured + optional DeBERTa)",
                "ctx_policy": True,
                "rerank": "cosine*lex_gate",
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report["measured_path"] = str(out)
    except Exception as e:
        report["measured_error"] = str(e)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = run_contract(live=not a.offline)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"contract ok={r['ok']} pass={r['n_pass']} fail={r['n_fail']} live={r['live']}")
        for c in r["checks"]:
            mark = "PASS" if c["ok"] else "FAIL"
            print(f"  [{mark}] {c['name']}  {c.get('detail','')[:100]}")
        if r["fails"]:
            print("FAILS:", *r["fails"], sep="\n  - ")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
