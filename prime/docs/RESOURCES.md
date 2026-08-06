# Resource plane — within reason

## This kit (detected)

| Resource | Reality |
|----------|---------|
| CPU | Snapdragon X Plus · 8 Oryon cores |
| RAM | ~16 GB total — leave headroom for OS + Grok + LM Studio |
| GPU | Qualcomm Adreno X1-45 (no CUDA) |
| NPU | Qualcomm Hexagon (OK in PnP) |
| In-process torch | **CPU-only** build |

## Utilization policy

| Layer | How we use it |
|-------|----------------|
| **CPU** | Parallel measures (`measure_parallel` / `measure all`) up to `max_workers` (typically 2–4) |
| **RAM** | Cap workers when free &lt; 3 GB; never cold-load Bonsai on low free |
| **GPU** | Via **LM Studio** GPU offload (LMS settings), not PyTorch CUDA |
| **NPU** | Via OS / LM Studio / vendor stack if LMS routes it — Prime does not pin QNN kernels directly |

## LM model routing

| Situation | Model |
|-----------|--------|
| Default scout | `liquid/lfm2.5-1.2b` or `ibm/granite-4-h-tiny` |
| Deep scout | `prism-ml/bonsai-27b` only if free RAM ≳ 8 GB **or** already loaded in LMS |
| Embeddings | nomic embed if needed later — still RAM-aware |

## MCP

- `resource_status` — snapshot + plan  
- `measure_parallel` — fan-out CPU instruments  
- `meta_loop` — includes resource plan at META_META  

## Env overrides

| Env | Effect |
|-----|--------|
| `PRIME_MAX_WORKERS` | Force worker count 1–8 |
| `PRIME_STATE_DIR` | Session state path |

## Within reason

Long **correct** graph &gt; 100% util. Saturating Adreno while thrashing 16 GB is a regression, not a flex.
