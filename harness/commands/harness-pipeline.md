---
name: harness-pipeline
description: Run multi-plane scout→bundle→certify pipeline (pack and/or live local scout)
---

# /harness-pipeline

Run the multi-plane harness pipeline under design law: explore ≠ certify.

## Resolve root

Prefer workspace containing `harness/pipeline/v1/pipeline.py` (PRIMEdEV-1 or plugin root `${GROK_PLUGIN_ROOT}` / `${CLAUDE_PLUGIN_ROOT}`).

## Default (offline pack → OPEN|STOP)

```bash
python harness/pipeline/v1/pipeline.py --pack filmore_magpi
```

Optional second pack:

```bash
python harness/pipeline/v1/pipeline.py --pack frozen_lakes_surface
```

## With existing scout markdown

```bash
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --scout path/to/scout.md
```

## Live local scout (LM Studio server :1234 required)

```bash
# fast
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout lfm
# deep / slow
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout bonsai
```

## Report

Print `BUNDLE_VERDICT`, opened/stopped claim ids, paths under `harness/pipeline/v1/out/`.  
Do not claim domain discovery. OPEN means structural multi-plane cover passed the gate only.
