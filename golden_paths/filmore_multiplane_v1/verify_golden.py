#!/usr/bin/env python3
"""Verify Filmore multi-plane golden sealed claim against sandbox extract.

Usage (from PRIMEdEV-1 root):
  python golden_paths/filmore_multiplane_v1/verify_golden.py

Optional:
  set GOLDEN_SANDBOX_ROOT to the SandBox folder if not at the default extract path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERT_PATH = HERE / "certificate.json"

DEFAULT_SANDBOX = Path(
    r"C:\PRIMEdEV-1\123abc\_sandbox_extract\SandBox"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cert = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    # Prefer portable roots: GOLDEN_SANDBOX_ROOT, monorepo extract, legacy absolute.
    # CI/buddy without field dumps: schema-only PASS when sandbox missing.
    # Force schema-only: GOLDEN_SANDBOX_ROOT=0 or GOLDEN_SCHEMA_ONLY=1
    repo = HERE.parents[1]  # …/golden_paths/filmore → repo root
    schema_only = os.environ.get("GOLDEN_SCHEMA_ONLY", "").strip() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("GOLDEN_SANDBOX_ROOT", "").strip() in ("0", "none", "schema")
    env_root = (os.environ.get("GOLDEN_SANDBOX_ROOT") or "").strip()
    if schema_only:
        sandbox = Path("__no_sandbox__")
    elif env_root and env_root not in ("0", "none", "schema"):
        sandbox = Path(env_root)
    else:
        candidates = [
            repo / "123abc" / "_sandbox_extract" / "SandBox",
            repo / "_sandbox_extract" / "SandBox",
            DEFAULT_SANDBOX,
        ]
        sandbox = next((p for p in candidates if p.is_dir()), Path("__no_sandbox__"))
    rel = cert["claims_artifact"]["relative_path"]
    claims_path = sandbox / rel.replace("\\", "/")
    # Windows path as stored
    if not claims_path.is_file():
        claims_path = sandbox / Path(rel)

    checks = []
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        checks.append({"check": name, "pass": passed, "detail": detail})
        if not passed:
            ok = False

    check("certificate_present", CERT_PATH.is_file())

    # Buddy/CI kits without field sandbox: verify certificate schema only (soft PASS)
    if not sandbox.is_dir():
        planes = {p["id"] for p in cert.get("planes", [])}
        for need in ("tool_magpi", "wits_surface", "decoder_rt"):
            check(f"plane_declared_{need}", need in planes)
        check("golden_claim_id_set", bool(cert.get("golden_claim_id")))
        check("claims_artifact_meta", bool(cert.get("claims_artifact")))
        print(
            json.dumps(
                {
                    "ok": ok,
                    "skipped_sandbox": True,
                    "sandbox": str(sandbox),
                    "checks": checks,
                    "note": "Sandbox extract not present — schema-only golden (set GOLDEN_SANDBOX_ROOT for full seal)",
                },
                indent=2,
            )
        )
        if ok:
            print("GOLDEN VERIFY OK (schema-only; no sandbox)")
            return 0
        print("GOLDEN VERIFY FAIL")
        return 1

    check("sandbox_root_exists", sandbox.is_dir(), str(sandbox))
    check("claims_json_exists", claims_path.is_file(), str(claims_path))

    if claims_path.is_file():
        digest = sha256_file(claims_path)
        expected = cert["claims_artifact"]["sha256"]
        check("claims_sha256", digest == expected, f"got={digest}")

        data = json.loads(claims_path.read_text(encoding="utf-8"))
        ids = [c.get("id") for c in data.get("claims", [])]
        required = cert["claims_artifact"]["required_ids"]
        check("claims_count", len(ids) == cert["claims_artifact"]["n_claims"], str(len(ids)))
        missing = [i for i in required if i not in ids]
        check("required_claim_ids", not missing, str(missing))
        golden = cert["golden_claim_id"]
        check("golden_claim_present", golden in ids, golden)
        # All listed claims HIGH in this sealed pack
        highs = [
            c.get("id")
            for c in data.get("claims", [])
            if str(c.get("confidence", "")).upper() == "HIGH"
        ]
        check("all_required_high", set(required).issubset(set(highs)), str(highs))

        # Continuity: golden claim text must assert multi-plane (decoder + magpi)
        golden_row = next(c for c in data["claims"] if c["id"] == golden)
        text = (golden_row.get("claim") or "").lower()
        check(
            "golden_claim_multiplane_language",
            "decoder" in text and ("mag-pi" in text or "magpi" in text or "tool" in text),
            text[:120],
        )

    # Plane matrix presence in certificate (schema guard)
    planes = {p["id"] for p in cert.get("planes", [])}
    for need in ("tool_magpi", "wits_surface", "decoder_rt"):
        check(f"plane_declared_{need}", need in planes)

    # Optional: release certificate mention
    release_md = sandbox / "Post_Run_SSI_Reports" / "SEND" / "RELEASE_CERTIFICATE.md"
    if release_md.is_file():
        body = release_md.read_text(encoding="utf-8", errors="replace")
        chain = cert.get("release_chain_sha256", "")
        check("release_chain_in_cert_md", chain in body, chain[:16])
    else:
        checks.append(
            {
                "check": "release_chain_in_cert_md",
                "pass": None,
                "detail": "optional; RELEASE_CERTIFICATE.md not found",
            }
        )

    print(json.dumps({"ok": ok, "claims_path": str(claims_path), "checks": checks}, indent=2))
    if ok:
        print("GOLDEN VERIFY OK")
        return 0
    print("GOLDEN VERIFY FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
