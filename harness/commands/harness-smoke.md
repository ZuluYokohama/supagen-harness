---
name: harness-smoke
description: Offline harness smoke (golden Q + certify + pipeline) and optional local scout smokes
---

# /harness-smoke

## Offline (no LMS)

```bash
python harness/smoke_offline.py
```

Expect: `HARNESS OFFLINE SMOKE OK`

## Local scouts (LM Studio :1234, model loaded)

```bash
python harness/local_mode/lfm_scout_v1/smoke_local.py
python harness/local_mode/bonsai_scout_v1/smoke_local.py
```

## Certify only

```bash
python harness/certify/v1/certify.py --demo filmore
```

Report pass/fail with exact error text. Never force OPEN.
