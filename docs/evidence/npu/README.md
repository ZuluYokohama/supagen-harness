# NPU evidence archive (tracked)

**Run ID:** `npu-htp-2026-08-06` · host ASUS Zenbook A14 · Snapdragon X Plus  
**Packages:** `onnxruntime==1.24.4` + `onnxruntime-qnn==2.4.0`

Slim copies of measured Hexagon HTP proofs. Full ONNX/QDQ weights and full `htp_profile.csv` stay local under `prime/state/npu/` (gitignored). This directory is the **tracked** archive so V&V docs can cite proofs without shipping multi‑MB profiles.

| File | Meaning |
|------|---------|
| `npu_qnn_smoke.json` | Plugin register + tiny QDQ HTP smoke (`NPU_PATH_LIVE`) |
| `npu_stress_report.json` | Sustained ~45s HTP load, providers, rates |
| `htp_profile_head.csv` | First lines of QNN HTP profile (HVX / accelerator / mm\* cycles) |
| `npu_nli_qdq_report.json` | DeBERTa QDQ **UINT8** on HTP: session OK, logits collapse ~neutral 0.51 |
| `npu_nli_qdq_uint16_report.json` | DeBERTa QDQ **act=UINT16 w=UINT8** (early): session OK ~32ms; **labels inverted** (0/3) |
| `npu_nli_qdq_uint16_expanded_calib_report.json` | Expanded CALIB (18) UINT16/UINT8 HTP: parity **1/4** |
| `npu_nli_qdq_uint16_contra_heavy_report.json` | Contra-heavy CALIB 32 UINT16/UINT8 HTP: held-out **1/4**, all 3 labels covered |
| `npu_nli_qdq_cpu_ep_same_graph.json` | Isolation: same QDQ on **CPU EP** **1/4** + `qdq_sha256` / held-out digest |
| `nli_htp_parity_cert.json` | E3 parity certificate consumed by `measure_fabric` (currently red) |
| `../vv_push_domains_integrity.json` | Portable D17 integrity (count_ok, n_warn, cells) |
| `nli_eval_qdq_vs_cpu.json` | **65-pair nli_eval_v1:** ORT fp32 con_high **0.952** vs QDQ CPU/HTP con_high **0.048** |

**Design law (earned):** NPU is **measure fabric only**; **Job2 product gate is CPU ORT/CE permanently**.  
HTP work is limited to measurement experiments (latency, profile, research). It does **not** replace or approximate the CPU gate. Verifier cannot be low-bit QDQ (con_high 0.95 → 0.05 on `nli_eval_v1`).

**E3 residual (accepted for GO_MEASURE):** label-parity **FAIL** (held-out **measured** 1/4 = 0.25).  
CPU-EP on the same QDQ graph matches HTP miss pattern → residual is **quant geometry**, not Hexagon routing.  
FP32 ORT CPU: **con_high 0.952**. Product: `measure_fabric` red cert → HTP refused as gate.

**Retention:** keep slim JSON/CSV heads in-repo; regenerate from local state when re-measuring. Full profile: `prime/state/npu/htp_profile.csv` (ignored).

Regenerate: `python prime/scripts/npu_qnn_smoke.py` · `npu_stress.py` · `npu_nli_qdq.py --act uint8|uint16`

**Authority:** Task Manager is not the NPU oracle on this Windows image. Use these artifacts + full local profile.
