# Local mode scouts

| Preset | Model | Footprint | Speed | Vision |
|--------|--------|-----------|-------|--------|
| [`lfm_scout_v1/`](lfm_scout_v1/) | LFM2.5-1.2B Instruct Q8 | ~1.2 GB | Fast | No (this file) |
| [`bonsai_scout_v1/`](bonsai_scout_v1/) | Bonsai 27B Q1_0 | ~3.6 GB | Slow | When mmproj ready |

**Both:** explore only · `may_certify_open: false` · certify via `harness/certify/v1`

```bash
python harness/local_mode/lfm_scout_v1/smoke_local.py
python harness/local_mode/bonsai_scout_v1/smoke_local.py
```
