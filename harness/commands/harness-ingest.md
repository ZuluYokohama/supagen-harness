---
name: harness-ingest
description: Ingest tool dump folder and/or ACQ .emz into plane inventory pack; optional pipeline
---

# /harness-ingest

User drops **dump files** and/or **database/ACQ packages** (`.emz`).

## SandBox demo

```bash
python harness/ingest/v1/ingest.py --sandbox-filmore --run-pipeline
```

## Custom paths

```bash
python harness/ingest/v1/ingest.py --dump path/to/MicroPulse_CSVs --emz path/to/well.emz --run-pipeline
```

Inventory-only (no pipeline):

```bash
python harness/ingest/v1/ingest.py --dump path/to/dumps --emz path/to/export.emz
```

## Law

Ingest packs are **DRAFT cover** by default (certify will STOP inventory claim until upgraded with multi-plane HIGH claims + relation_summary).  
Report plane list and paths under `harness/ingest/v1/out/`.
