# lfm_scout_v1 — fast local explore scout

**Regime:** `scout_fast`  
**Model:** Liquid LFM2.5-1.2B-Instruct Q8_0 (~1.2 GB)  
**Authority:** EXPLORE only → external `certify/v1`

## Pairing

| Preset | Use when |
|--------|----------|
| **lfm_scout_v1** | Interactive loops, draft JSON, always-on dual-track |
| **bonsai_scout_v1** | Deep / slow multi-plane narrative, long context, binary-structure experiments |

Same law. Same certify gate. Different search geometry.

## Activate

1. LM Studio → load `LFM2.5-1.2B-Instruct-Q8_0.gguf` (id often `liquid/lfm2.5-1.2b`)
2. Server `http://127.0.0.1:1234`
3. Smoke / turn:

```bash
python harness/local_mode/lfm_scout_v1/smoke_local.py
python harness/local_mode/lfm_scout_v1/scout_turn.py --certify-demo
```

## Path

```
C:\LM_STUDIO_MODELS\00.LLM HF MODELS4 CODING-RESEARCH-TESTING-USE-RESEARCH-TESTING-USE-1JUN26\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf
```
