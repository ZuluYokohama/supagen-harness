#!/usr/bin/env python3
"""Assert HEAD still carries dispositions for prior CodeRabbit majors."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    }
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
