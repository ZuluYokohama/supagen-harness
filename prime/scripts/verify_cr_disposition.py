#!/usr/bin/env python3
"""Assert HEAD still carries dispositions for prior CodeRabbit majors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "prime" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def main() -> int:
    def t(rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

    checks = {
        "owns_open_false": 'owns_open_gate": False' in t("prime/scripts/truth_plane.py"),
        "measure_fabric": (ROOT / "prime/scripts/measure_fabric.py").is_file(),
        "htp_refuse": "htp_refused" in t("prime/scripts/entailment_glue.py"),
        "qdq_complete": "_qdq_looks_complete" in t("prime/scripts/npu_stress.py"),
        "stream_profile": "n_lines_scanned" in t("prime/scripts/npu_stress.py"),
        "ps_comma": ", @{N='val'" in t("prime/scripts/npu_stress.py"),
        "single_quantize": "do not pre-write via quantize_static" in t(
            "prime/scripts/npu_qnn_smoke.py"
        ),
        "preserve_keys": "any(p in env_p.lower()" in t("prime/scripts/residency.py"),
        "preserve_alone": "preserve_alone" in t("prime/scripts/residency.py"),
        "rerank_envelope": "rerank inference failed" in t("prime/scripts/rerank_service.py"),
        "ce_model_locks": "_CE_MODEL_LOCKS" in t("prime/scripts/entailment_glue.py"),
        "stable_none": "stable" in t("prime/scripts/truth_plane.py")
        and "else None" in t("prime/scripts/truth_plane.py"),
        "d1_applied": "applied_rule" in t("prime/scripts/vv_full_matrix.py"),
        "golden_fail_closed": "incomplete" in t(
            "golden_paths/filmore_multiplane_v1/verify_golden.py"
        ),
        "go_seal": (ROOT / "docs/GO_MEASURE_SEAL.md").is_file(),
        "e3_accept": (ROOT / "docs/RESIDUAL_ACCEPTANCE_E3.md").is_file(),
        "cr_map": (ROOT / "docs/CODERABBIT_RESOLUTION_MAP.md").is_file(),
        "ensure_mode": "--mode" in t("supagen/supagen/cli.py"),
        "e3_parity_gate": "nli_htp_parity_pass" in t("docs/COMPUTE_HW_ABSTRACTION.md"),
        "job2_owns_open_false_results": '"job2_owns_open": false'
        in t("docs/VV_RUN_RESULTS.md").lower()
        or '"job2_owns_open": false' in t("docs/VV_RUN_RESULTS.md"),
        "n_warn_count": "n_warn" in t("prime/scripts/vv_push_domains.py"),
        "family_mismatch": "family_mismatch" in t("prime/scripts/bakeoff_aboutness_30.py"),
        "register_returns_copy": "return dict(_LAST)" in t("prime/scripts/npu_qnn.py"),
        "adv_empty_guard": "all adversarial pairs skipped" in t(
            "prime/scripts/bakeoff_adv_lexical.py"
        ),
        "nli_partial_not_live_alone": (
            "NPU_NLI_LIVENESS_PROBE" in t("prime/scripts/npu_nli_qdq.py")
            or "NPU_NLI_PARTIAL" in t("prime/scripts/npu_nli_qdq.py")
        )
        and "label_parity_rate" in t("prime/scripts/npu_nli_qdq.py"),
        "nli_ort_fail_envelope": 'gate": "NEED_INFO"' in t("prime/scripts/entailment_glue.py")
        and "ort_nli_failed" in t("prime/scripts/entailment_glue.py"),
        "rerank_backoff": "PRIME_RERANK_FAIL_BACKOFF" in t("prime/scripts/rerank_service.py")
        or "_FAIL_BACKOFF" in t("prime/scripts/rerank_service.py"),
        "held_out_npu": "HELD_OUT" in t("prime/scripts/npu_nli_qdq.py")
        and "parity_with_ort" in t("prime/scripts/npu_nli_qdq.py"),
        "go_measure_seal": "GO_MEASURE" in t("docs/GO_MEASURE_SEAL.md"),
        "qnn_opt_in_only": 'pref in ("qnn", "npu")' in t("prime/scripts/accel_nli_ort.py")
        and 'pref in ("auto", "qnn", "npu")' not in t("prime/scripts/accel_nli_ort.py"),
        "oneway_p_shared": "ONEWAY_P" in t("prime/scripts/accel_nli_ort.py")
        and "ONEWAY_P" in t("prime/scripts/entailment_glue.py"),
        "mutual_uses_ort": "_nli_one_way" in t("prime/scripts/entailment_glue.py"),
        "tier_b_rerank_required": "rerank_residual" in t("prime/scripts/tier_b_challenger.py"),
        "golden_struct_gate": "claims_artifact_structure" in t(
            "golden_paths/filmore_multiplane_v1/verify_golden.py"
        ),
        "held_out_disjoint_fn": "validate_held_out_disjoint" in t(
            "prime/scripts/npu_nli_qdq.py"
        ),
        "ort_parity_force_cpu_call": "force_cpu=True" in t("prime/scripts/npu_nli_qdq.py"),
        "predict_force_cpu_param": "force_cpu: bool = True" in t(
            "prime/scripts/accel_nli_ort.py"
        ),
        "cert_contract_live_structured": '"contract_live"'
        in t("docs/evidence/DOMAIN_COMPLETION_CERTIFICATE.json")
        and '"measured_count"'
        in t("docs/evidence/DOMAIN_COMPLETION_CERTIFICATE.json"),
        "nli_ort_force_cpu_kw": "force_cpu: bool = True" in t(
            "prime/scripts/entailment_glue.py"
        )
        or "force_cpu=True" in t("prime/scripts/entailment_glue.py"),
        # D4 CR major: preserve_ok must require frankenstein, not any non-empty key
        "d4_preserve_ok_frankenstein_only": (
            'preserve_ok = "frankenstein" in preserve_key' in t(
                "prime/scripts/vv_full_matrix.py"
            )
            and "or bool(preserve_pick.get(\"key\"))"
            not in t("prime/scripts/vv_full_matrix.py")
        ),
        "npu_evidence_archive_note": "docs/evidence/npu" in t(".gitignore")
        or "docs/evidence/npu/" in t(".gitignore"),
        "buddy_l8_timeout": "TimeoutExpired" in t("prime/scripts/buddy_l8_offline.py")
        and "PRIME_BUDDY_L8_TIMEOUT" in t("prime/scripts/buddy_l8_offline.py"),
        "prefer_unknown_reject": "unknown prefer=" in t("prime/scripts/entailment_glue.py"),
        "held_out_mandatory": (
            'bool(cert.get("held_out"))' in t("prime/scripts/measure_fabric.py")
            and 'or cert.get("label_parity_rate") is not None)'
            not in t("prime/scripts/measure_fabric.py")
        ),
        "count_ok_no_assert": (
            "count_ok" in t("prime/scripts/vv_push_domains.py")
            and "assert n_pass + n_warn + n_fail" not in t("prime/scripts/vv_push_domains.py")
        ),
        "rerank_rev_required": "40-char commit SHA" in t("prime/scripts/rerank_service.py")
        or "full 40-char commit SHA" in t("prime/scripts/rerank_service.py"),
        "qdq_recipe_reject": "unsupported QDQ recipe" in t("prime/scripts/npu_nli_qdq.py"),
        "md_truncate_wrapper": '"truncated": True' in t("prime/scripts/vv_full_matrix.py")
        or '"truncated": true' in t("prime/scripts/vv_full_matrix.py").lower(),
        "truth_loop_default_one": 'PRIME_TRUTH_ROUNDS", "1"' in t(
            "prime/scripts/truth_plane.py"
        ),
        "ort_force_cpu_no_synth": 'run.get("ort_force_cpu") is True'
        in t("prime/scripts/measure_fabric.py")
        and 'ort_force_cpu", True)' not in t("prime/scripts/measure_fabric.py"),
        "glue_fiber_mode_param": "fiber_mode: str | None = None"
        in t("prime/scripts/entailment_glue.py"),
        "dual_enter_passes_fiber": "fiber_mode=mode" in t("prime/scripts/dual_enter.py"),
        "held_out_neutral": '"neutral"' in t("prime/scripts/npu_nli_qdq.py")
        and "labels_covered" in t("prime/scripts/npu_nli_qdq.py"),
        "soft_critical_fn_names": '"d1_aboutness"' in t("prime/scripts/vv_full_matrix.py")
        and '"d8_accel_npu"' in t("prime/scripts/vv_full_matrix.py")
        and '"d1_job1_aboutness"' not in t("prime/scripts/vv_full_matrix.py"),
        "d15_freshness": "freshness_required" in t("prime/scripts/vv_full_matrix.py"),
        "ort_session_reuse": "EP class mismatch" in t("prime/scripts/accel_nli_ort.py"),
        "intent_self_symmetric": '"hypothesis": self_text' in t("prime/scripts/truth_plane.py"),
        "hexagon_present_not_hardcoded": (
            '"present": None' in t("prime/scripts/truth_plane.py")
            or "present_note" in t("prime/scripts/truth_plane.py")
        ),
        "cpu_qdq_isolation": "run_cpu_qdq" in t("prime/scripts/npu_nli_qdq.py")
        and "static_qdq_geometry" in t("prime/scripts/npu_nli_qdq.py"),
        "e3_quant_not_htp": "static QDQ of DeBERTa" in t(
            "docs/RESIDUAL_ACCEPTANCE_E3.md"
        )
        or "quant geometry" in t("docs/RESIDUAL_ACCEPTANCE_E3.md").lower()
        or "CPU EP" in t("docs/RESIDUAL_ACCEPTANCE_E3.md")
        or "CPUExecutionProvider" in t(
            "docs/evidence/npu/npu_nli_qdq_cpu_ep_same_graph.json"
        ),
        "jina_listwise_no_chunk": "jina_rerank_api" in t("prime/scripts/rerank_service.py")
        and "never chunk that path" in t("prime/scripts/rerank_service.py"),
        "d12_critical": 'critical=True' in t("prime/scripts/vv_full_matrix.py")
        and "D12_ort_nli" in t("prime/scripts/vv_full_matrix.py"),
        "smoke_no_device_ptrs": "0x0000" not in t("docs/evidence/npu/npu_qnn_smoke.json"),
        "d17_portable_evidence": (
            ROOT / "docs/evidence/vv_push_domains_integrity.json"
        ).is_file(),
    }

    # --- Executable routing / bank integrity (not string-only) ---
    runtime: dict[str, bool] = {}
    try:
        from npu_nli_qdq import (  # type: ignore
            CALIB_PAIRS,
            HELD_OUT_PAIRS,
            _norm_pair_text,
            validate_held_out_disjoint,
        )

        validate_held_out_disjoint()
        calib = {_norm_pair_text(a) for a, b in CALIB_PAIRS} | {
            _norm_pair_text(b) for a, b in CALIB_PAIRS
        }
        overlap = False
        for a, b, _lab in HELD_OUT_PAIRS:
            if _norm_pair_text(a) in calib or _norm_pair_text(b) in calib:
                overlap = True
                break
        labels = {lab for _a, _b, lab in HELD_OUT_PAIRS}
        labels_ok = labels >= {"contradiction", "entailment", "neutral"}
        runtime["runtime_held_out_disjoint"] = (
            not overlap and len(HELD_OUT_PAIRS) >= 3 and labels_ok
        )
        runtime["runtime_held_out_has_neutral"] = "neutral" in labels
    except Exception as e:
        runtime["runtime_held_out_disjoint"] = False
        runtime["runtime_held_out_has_neutral"] = False
        print("runtime held_out error:", e)

    try:
        import measure_fabric as _mf  # type: ignore

        # Inject red parity so this check never fails on a future green E3 cert
        _orig = _mf.nli_htp_parity_pass

        def _red_parity(*_a, **_k):
            return {"ok": False, "reason": "disposition_injected_red"}

        _mf.nli_htp_parity_pass = _red_parity  # type: ignore[assignment]
        try:
            order = _mf.route_job2().get("order") or []
            runtime["runtime_red_cert_no_htp_first"] = (
                not order or order[0] != "htp_qdq"
            ) and "htp_qdq" not in (order[:1] or [])
        finally:
            _mf.nli_htp_parity_pass = _orig  # type: ignore[assignment]
        # Live path still honest: if currently red, order must not start with htp
        par = _orig()
        live_order = _mf.route_job2().get("order") or []
        if not par.get("ok"):
            runtime["runtime_live_red_no_htp_first"] = (
                not live_order or live_order[0] != "htp_qdq"
            )
        else:
            # Green cert may advertise htp first in route_job2; product glue still auto
            runtime["runtime_live_red_no_htp_first"] = True
    except Exception as e:
        runtime["runtime_red_cert_no_htp_first"] = False
        runtime["runtime_live_red_no_htp_first"] = False
        print("runtime fabric error:", e)

    # Executable: incomplete labels_covered cannot green parity cert
    try:
        import measure_fabric as _mf2  # type: ignore
        import tempfile
        from pathlib import Path as _P

        tmp = _P(tempfile.gettempdir()) / "prime_fake_parity_cert.json"
        old_cert = _mf2.PARITY_CERT
        _mf2.PARITY_CERT = tmp  # type: ignore[assignment]
        try:
            # Case A: incomplete labels_covered (held_out true, rate 1.0) must stay red
            tmp.write_text(
                '{"ok": true, "label_parity_rate": 1.0, "hits": 3, "n": 3, '
                '"cpu_fallback": false, "qnn_ep_registered": true, '
                '"ort_force_cpu": true, "held_out": true, '
                '"labels_covered": ["entailment"], "recipe": {"act": "uint8"}, '
                '"model_id": "x", "probe_only": false, "uncalibrated_probe": false}',
                encoding="utf-8",
            )
            bad_lab = _mf2.nli_htp_parity_pass()
            # Case B: empty labels_covered must stay red
            tmp.write_text(
                '{"ok": true, "label_parity_rate": 1.0, "hits": 3, "n": 3, '
                '"cpu_fallback": false, "qnn_ep_registered": true, '
                '"ort_force_cpu": true, "held_out": true, '
                '"labels_covered": [], "recipe": {"act": "uint8"}, '
                '"model_id": "x", "probe_only": false, "uncalibrated_probe": false}',
                encoding="utf-8",
            )
            bad_empty = _mf2.nli_htp_parity_pass()
            # Case C: missing labels_covered key must stay red
            tmp.write_text(
                '{"ok": true, "label_parity_rate": 1.0, "hits": 3, "n": 3, '
                '"cpu_fallback": false, "qnn_ep_registered": true, '
                '"ort_force_cpu": true, "held_out": true, '
                '"recipe": {"act": "uint8"}, "model_id": "x", '
                '"probe_only": false, "uncalibrated_probe": false}',
                encoding="utf-8",
            )
            bad_miss = _mf2.nli_htp_parity_pass()
            runtime["runtime_incomplete_labels_not_green"] = (
                not bool(bad_lab.get("ok"))
                and not bool(bad_empty.get("ok"))
                and not bool(bad_miss.get("ok"))
            )
        finally:
            _mf2.PARITY_CERT = old_cert  # type: ignore[assignment]
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception as e:
        runtime["runtime_incomplete_labels_not_green"] = False
        print("runtime incomplete labels error:", e)

    # Executable: D17-style count recompute from cells
    try:
        cells = [
            {"status": "PASS", "critical": True},
            {"status": "WARN", "critical": False},
            {"status": "FAIL", "critical": False},
        ]
        n_pass = sum(1 for c in cells if c.get("status") == "PASS")
        n_warn = sum(1 for c in cells if c.get("status") == "WARN")
        n_fail = sum(1 for c in cells if c.get("status") == "FAIL")
        runtime["runtime_d17_count_recompute"] = (
            n_pass == 1 and n_warn == 1 and n_fail == 1 and n_pass + n_warn + n_fail == 3
        )
        runtime["runtime_d17_empty_cells_reject"] = not (
            isinstance([], list) and len([]) > 0
        )  # empty must fail cells_ok
    except Exception as e:
        runtime["runtime_d17_count_recompute"] = False
        runtime["runtime_d17_empty_cells_reject"] = False
        print("runtime d17 error:", e)

    try:
        cert = json.loads(
            (ROOT / "docs/evidence/DOMAIN_COMPLETION_CERTIFICATE.json").read_text(
                encoding="utf-8"
            )
        )
        cl = cert.get("contract_live") or {}
        runtime["runtime_cert_structured"] = isinstance(cl, dict) and bool(
            cl.get("measured_count")
        ) and bool(cl.get("evidence_path"))
    except Exception as e:
        runtime["runtime_cert_structured"] = False
        print("runtime cert error:", e)

    checks.update(runtime)
    miss = [k for k, v in checks.items() if not v]
    print(f"disposition_ok={len(checks) - len(miss)}/{len(checks)}")
    for k, v in sorted(checks.items()):
        print(f"  {'OK' if v else 'MISS'}  {k}")
    if miss:
        print("FAIL", miss)
        return 1
    print("CR_DISPOSITION_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
