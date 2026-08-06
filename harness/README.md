# multiplane-harness

Agentic multi-plane control surface: LLMs search; law certifies.  
**Plugin root:** this folder — install via `docs/PLUGIN_INSTALL.md`.

## Slash commands (use these)

| Command | Purpose |
|---------|---------|
| **`/harness-pipeline`** | Main: pack ± live scout → OPEN\|STOP |
| **`/harness-smoke`** | Offline golden + certify + pipeline smoke |
| **`/harness-ingest`** | Dump + `.emz` → inventory pack (± pipeline) |
| **`/harness-certify`** | Bundle JSON → external gate only |

## CLI (same thing without slash)

```bash
cd C:\PRIMEdEV-1

python harness/smoke_offline.py
python harness/pipeline/v1/pipeline.py --pack filmore_magpi
python harness/pipeline/v1/pipeline.py --pack frozen_lakes_surface
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout lfm
python harness/ingest/v1/ingest.py --sandbox-filmore --run-pipeline
python harness/certify/v1/certify.py --demo filmore
```

## Layout

| Path | Role |
|------|------|
| `pipeline/v1/` | scout → bundle → certify |
| `ingest/v1/` | dump + emz plane inventory |
| `certify/v1/` | OPEN\|STOP gate |
| `local_mode/lfm_scout_v1` | Fast scout (LFM 1.2B) |
| `local_mode/bonsai_scout_v1` | Deep scout (Bonsai 27B) |
| `commands/` | Slash recipes |
| `skills/multiplane-harness/` | Agent skill |
| `docs/FIELD_RUNBOOK.md` | On-tour dual-track |
| `docs/PLUGIN_INSTALL.md` | How to enable plugin |

## Law

```
explore (scout) → draft → certify (external) → OPEN | STOP
```

Residue never forced. Vision plane: wait for Bonsai mmproj download.
