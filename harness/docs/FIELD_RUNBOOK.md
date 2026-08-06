# Field runbook (item 6 — you execute on tour)

Dual-track: MWD day-rate floor + harness when idle.

## Minimum kit

1. LM Studio + **LFM2.5-1.2B** loaded (fast)  
2. Optional: Bonsai for deep night pass  
3. Tool dump folder + ACQ `.emz` for the run  

## Loop

```bat
:: 1) Ingest
python C:\PRIMEdEV-1\harness\ingest\v1\ingest.py --dump D:\path\dumps --emz D:\path\well.emz

:: 2) Optional: upgrade pack manually or use sealed pack templates
::    (filmore_magpi / frozen_lakes_surface are workshop demos)

:: 3) Pipeline with live scout
python C:\PRIMEdEV-1\harness\pipeline\v1\pipeline.py --pack filmore_magpi --live-scout lfm

:: 4) Read summary under harness\pipeline\v1\out\
```

## Slash (in Grok/Claude with multiplane-harness plugin)

- `/harness-ingest`  
- `/harness-pipeline`  
- `/harness-certify`  
- `/harness-smoke`  

## Law on location

- Scout may draft.  
- Only certify OPENs.  
- Incomplete cover = residue, not failure of nerve.
