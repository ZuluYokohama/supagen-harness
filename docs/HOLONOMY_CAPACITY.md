# Holonomy identity floor (v3) — capacity sheet

**Gate:** mutual entailment both ways (DeBERTa). **Threshold:** p ≥ 0.80.  
**Cosine is diagnostic only** — never gates agreement or identity.

| Model | Quant | identity p | cos≥0.75 would say | median cos | Gate | Modes (8 seeds) |
|-------|-------|------------|--------------------|------------|------|-----------------|
| **frankenstein-2.0-i1** | Q4_K_S 7.2B | **~0.88** (chain flat through d4) | 1.00 | ~0.89 | **PASS** | mostly closed |
| **liquid/lfm2.5-1.2b** | Q8 1.2B | **0.29** | **0.00** | ~0.66 | **FAIL** | dropped×4, inverted×1, closed×2 |
| **gemma-4-12b** Q4_K_M | 12B | **0.38** | — | — | **FAIL** | channel/meta pollution |

## Reading (Claude's accounting)

- **p** = fraction of identity round-trips that **mutually entail** (same claim both ways).  
- Cosine can look “ok-ish” (LFM median ~0.66) while **p collapses** — same failure class as granite rewords scoring 0.598 cos while DeBERTa said entailed both ways… inverted here: LFM often **drops** content (one-way entailment).  
- LFM was **not** a strong holonomy subject. Use it as **fast scout fiber**, not multi-hop identity carrier.  
- Frankenstein remains the multi-hop / identity-chain subject on this kit.  
- Gemma-12B capacity did **not** buy higher p.

## Reproduce

```bash
python prime/scripts/run_lfm_identity_floor.py
python prime/scripts/run_identity_chain.py --model frankenstein-2.0-i1
# prior gemma: prime/scripts/run_gemma12b_floor.py (needs llama-server side load)
```

Artifacts: `prime/state/holonomy_v3_lfm12b_identity_floor.json`,  
`holonomy_v3_frankenstein_identity_chain.json`, `holonomy_v3_gemma12b_floor.json`.
