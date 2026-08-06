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
| `npu_nli_qdq_uint16_expanded_calib_report.json` | **Expanded balanced CALIB (18 pairs)** UINT16/UINT8: HTP ~35ms; parity **1/4 (0.25)** — still red |

**E3 residual (accepted for GO_MEASURE):** label-parity **still FAIL** (must be ≥0.9 for green).  
Progress: 0/3 invert → **1/4** after balanced calib (entail hit; contra/neutral still wrong).  
ORT CPU on same pairs: **PASS**. Product path: `measure_fabric` + red cert → HTP refused.  
See `docs/RESIDUAL_ACCEPTANCE_E3.md`. Next: more calib / stance distill / QAI Hub — **not** GPU llama hunt for law.

**Retention:** keep slim JSON/CSV heads in-repo; regenerate from local state when re-measuring. Full profile: `prime/state/npu/htp_profile.csv` (ignored).

Regenerate: `python prime/scripts/npu_qnn_smoke.py` · `npu_stress.py` · `npu_nli_qdq.py --act uint8|uint16`

**Authority:** Task Manager is not the NPU oracle on this Windows image. Use these artifacts + full local profile.
