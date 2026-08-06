# NPU evidence archive (tracked)

Slim copies of measured Hexagon HTP proofs. Full ONNX/QDQ weights stay local (`prime/state/ort_models/`, gitignored).

| File | Meaning |
|------|---------|
| `npu_qnn_smoke.json` | Plugin register + tiny QDQ HTP smoke (`NPU_PATH_LIVE`) |
| `npu_stress_report.json` | Sustained ~45s HTP load, providers, rates |
| `htp_profile_head.csv` | First lines of QNN HTP profile (HVX / accelerator / mm\* cycles) |
| `npu_nli_qdq_report.json` | DeBERTa QDQ on HTP: session OK, **label parity residual** |

Regenerate: `python prime/scripts/npu_qnn_smoke.py` · `npu_stress.py` · `npu_nli_qdq.py`

**Authority:** Task Manager is not the NPU oracle on this Windows image. Use these artifacts + full `prime/state/npu/htp_profile.csv` locally.
