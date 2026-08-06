# Certify v1 — external OPEN | STOP gate

**Authority:** only this layer may OPEN.  
Local Bonsai / cloud LLMs = explore + draft bundles.  
This tool = continuity / cover certification (not mystical truth).

## Law

```
scout (LMS Bonsai, etc.)  →  claim bundle JSON
certify.py                →  OPEN | STOP per claim + seal
ship / merge / client     →  only on OPEN artifacts
```

## Run demo (Filmore multi-plane)

```bash
python harness/certify/v1/certify.py --demo filmore
```

Expect: golden claim **OPEN**, draft single-plane story **STOP** → `BUNDLE_VERDICT=MIXED`.

## Bundle shape

See `schema_plane_claim.json`. Minimal idea:

- `planes[]` — id, modality, present, state, evidence_refs  
- `claims[]` — required_planes, text, confidence_requested, relation_summary  
- tag `allow_dead_plane` when DEAD is *evidence* (e.g. decoder dead)

## Wire to local scout

1. Scout drafts a bundle (or human/scripts build it).  
2. `python harness/certify/v1/certify.py bundle.json -o cert.json`  
3. Only then treat claims as sealed.

v1 is **structural**. Semantic audit (λ₁, domain physics) can plug in later without changing the authority split.
