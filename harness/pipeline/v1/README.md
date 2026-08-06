# Pipeline v1 — scout → bundle → certify

```
[optional live scout: LFM | Bonsai]
            │
            ▼
   evidence pack (planes + HIGH claims)
   + scout DRAFT claims (never auto-OPEN)
            │
            ▼
        bundle.json
            │
            ▼
      certify.py gate
            │
            ▼
   certificate.json + summary.md
```

## Commands

```bash
# pack only (offline, no LMS)
python harness/pipeline/v1/pipeline.py --pack filmore_magpi

# pack + prior scout markdown
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --scout harness/local_mode/lfm_scout_v1/runs/scout_XXXX.md

# live LFM scout then certify (LMS :1234)
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout lfm

# live Bonsai (slow)
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout bonsai
```

Outputs under `harness/pipeline/v1/out/`.

## Law

- Pack / structured evidence may request **HIGH** (can OPEN if cover glues).
- Scout-parsed claims are always **DRAFT** → STOP until a human/pack upgrades with evidence.
- External gate only OPENs.
