# LM Studio as projection algebra

## Default: one LFM, orthogonal operators

**Preferred:** `liquid/lfm2.5-1.2b` + nomic embed only.  
Roles SCOUT / FALSIFY / GLUE / VERDICT = dimensional projections via abstraction, not multi-model RAM burn.

See also: [LM Studio Developer Docs](https://lmstudio.ai/docs/developer) map in `LM_STUDIO_DEVELOPER_MAP.md`.

## Not normal thinking

LMS is not “a local ChatGPT.” On this stack it is a **local multi-fiber inference + residency controller**:

| LMS capability | Prime reading |
|----------------|---------------|
| Multiple models loaded | Parallel **fibers** over the same enter-event |
| `/api/v1/chat` | Project event through one fiber (native) |
| `/v1/embeddings` | **Metric** on language stalks (768-d nomic) |
| `/api/v1/models/load\|unload` | Which fibers are **resident** under 16GB RAM |
| Stateful chat / response_id | Persistent fiber trajectory across enters |
| Tool-trained models (Granite etc.) | Future: fibers that call Prime tools |
| MCP via LMS (`/api/v1/chat`, `/v1/responses`) | Future: LMS as MCP host *or* client into Prime |
| Streaming load/prompt events | Observability of projection cost |

Grok/Prime = **meta operator** (graph, law, OPEN\|STOP).  
LMS = **dimensional projection engine** (local, private, resource-bounded).

## Every enter

```
user enter  →  base event E
             ├─ fiber LFM      (fast, cheap stalk)
             ├─ fiber Granite  (tool-ready mid stalk)
             ├─ fiber Bonsai   (deep stalk — swap in only when needed)
             └─ embed(E) + embed(fiber_i) → cosine glue
votes(OPEN|STOP|NEED_INFO) + agreement → regime coherent_fiber | dispersed_streams
```

MCP: **`enter_projection`** — MEASURE only. Never OPEN authority.

## Simplest use via MCP

```
meta_loop(intent)
restrict(...)
enter_projection(prompt=intent)     # multi-fiber
project_align()                     # human ↔ domains (code/rplc/eref)
measure_parallel()                  # smoke+rplc+…
audit(OPEN|STOP)
```

Residency:

```
lm_load(model="liquid/lfm2.5-1.2b", context_length=4096)
lm_unload(instance_id="ibm/granite-4-h-tiny")
lm_models()   # native catalog + roster recommendation
```

## Layered stack (now default)

See **`LMS_LAYERED_GATES.md`**. Default enter path is L0–L7:

residency consolidate → native `/api/v1/chat` fiber → JSON roles → nomic glue → policy demotion.

`previous_response_id` chaining is **live-verified**. Instance spam is **L1-gated**.

## What we are *not* claiming yet

- LMS-hosted remote MCP calling Prime tools (path documented; not default)
- `/v1/responses` custom tools loop as primary path
- That multi-fiber agreement is truth

## Hardware map (this kit)

| Resource | LMS role |
|----------|----------|
| Adreno GPU / Hexagon NPU | LMS runtime offload if enabled in LMS settings |
| RAM 16GB | load/unload is the real control surface |
| CPU | Prime parallel measures + projection fan-out |

Keep **LFM + nomic embed** resident. Swap Granite/Bonsai.
