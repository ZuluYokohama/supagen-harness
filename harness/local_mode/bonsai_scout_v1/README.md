# bonsai_scout_v1 — local explore-only mode

**Decision-tree node:** local LLM branch after GGUF lands.  
**Authority:** EXPLORE only. Certify is external (never this model).

## Model on disk

```
...\lmstudio-community\Bonsai-27B-GGUF\Bonsai-27B-Q1_0.gguf   (~3.6 GB, ready)
...\downloading_mmproj-*.part                                 (vision — wait for complete)
```

## Activate

1. Open LM Studio → load `Bonsai-27B-Q1_0.gguf`
2. Paste system prompt from `system_scout_explore_only.txt` (or load as preset)
3. Start **Local Server** (default `http://127.0.0.1:1234`)
4. Smoke:

```bash
python harness/local_mode/bonsai_scout_v1/smoke_local.py
```

## Files

| File | Role |
|------|------|
| `preset.json` | Paths, sampling regimes, authority flags |
| `system_scout_explore_only.txt` | G1/value + multi-plane scout discipline |
| `smoke_local.py` | Server + chat smoke; fails soft if LMS off |

## Law

```
local Bonsai  →  draft planes / relations / DRAFT claims
external gate →  OPEN | STOP | residue
```

Sampling regimes in `preset.json`: `scout_default` | `tight_draft` | `boundary_explore`.
