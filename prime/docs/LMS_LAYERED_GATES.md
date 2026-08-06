# LMS layered gates — hyper-optimized abstraction

Source: [LM Studio Developer Docs](https://lmstudio.ai/docs/developer) + live probes on this kit.

## Why layers

LMS ships three families of endpoints (native `/api/v1`, OpenAI `/v1`, Anthropic `/v1/messages`).  
Without gates you get: **instance spam**, free-text role drift, multi-model RAM thrash, and accidental “OPEN” from a 1.2B model.

Prime wraps LMS as an **eight-layer stack**. Every activity has a layer, a cost measure, and a stop condition.

```
L0.5 LOCAL HOME  ~/.lmstudio settings · server-logs · model.yaml · backends
L0   TRANSPORT   HTTP · auth · timeouts · error taxonomy
L1   RESIDENCY   catalog · load/unload · dedupe instances · RAM gate
L2   INFERENCE   /api/v1/chat · /v1/responses · /v1/embeddings
L3   STATE FIBER store=true · previous_response_id (OFF if pack large)
L4   STRUCTURE   JSON-gated SCOUT → FALSIFY → GLUE → VERDICT
L5   METRIC      nomic ABOUTNESS (retrieve) + NLI AGREEMENT (glue) · tok/s
L6   POLICY      FATAL demotes OPEN_CANDIDATE · never production OPEN
L7   ORCHESTRATION enter_projection · deep_loop · dimensional handoff
```

Design law: **restrict → measure → audit → OPEN|STOP**. Residue never forced.  
**LMS is MEASURE only.** Production OPEN stays in Prime audit + domain measures.

---

## LMS development elements → gates

| LMS surface | Docs | Gate layer | Our use |
|-------------|------|------------|---------|
| `GET /api/v1/models` | List + `loaded_instances` + config | L1 | Catalog, duplicate detect, ctx per instance |
| `POST /api/v1/models/load` | `model`, `context_length` | L1 | Load only if missing; **never** re-load → `:2,:3` spam |
| `POST /api/v1/models/unload` | `instance_id` | L1 | Drop deep models + extras |
| `POST /api/v1/chat` | `input`, `system_prompt`, `store`, `previous_response_id`, `integrations`, `context_length`, `reasoning` | L2–L4 | Primary fiber; MCP-ready |
| Chat output blocks | `message` \| `tool_call` \| `reasoning` \| `invalid_tool_call` | L2 parse | Structured tool observation |
| Chat `stats` | tokens, tok/s, TTFT, `model_load_time_seconds` | L5 cost | Latency / thrash signal |
| `POST /v1/embeddings` | nomic + task prefixes | L2 / Job1 | **Aboutness/retrieval only** — not claim agreement |
| NLI (LFM or cross-encoder) | entail/contradict/neutral | L5 Job2 | **Agreement glue** — cosine has no contradiction channel |
| `POST /v1/responses` | Open Responses, `previous_response_id`, tools | L2 alt | Rich stateful / tools path |
| `POST /v1/chat/completions` | OpenAI chat | L2 fallback | Compatibility only |
| MCP integrations | `ephemeral_mcp` + `plugin` `mcp/<label>` | L2 optional | Future: model as client of Prime tools |
| Auth `LM_API_TOKEN` | Bearer on REST | L0 | Optional |

### Local home (`~/.lmstudio`) — this kit

| Path | What we use |
|------|-------------|
| `settings.json` | default ctx **4096**, guardrails **4GB high**, JIT TTL 3600s, unload-previous |
| `.internal/http-server-config.json` | port 1234, bind 127.0.0.1, JIT load, **verbose full** logs |
| `server-logs/YYYY-MM/*.log` | ERROR/WARN: **Context size exceeded**, GPU offload unsupported |
| `hub/models/*/model.yaml` | LFM minRAM ~0.95GB, temp 0.1 top_p 0.1; Bonsai min ~3.8GB |
| `extensions/backends/llama.cpp-win-arm64-*` | engine **2.27.1**, ARM64, **no GPU offload** |
| `mcp.json` | **empty** — no LMS plugin MCPs |
| `.internal/api-prediction-history/` | response_id → pack index (audit trail) |
| `downloadsFolder` | `C:\LM_STUDIO_MODELS\...` |

Observer: `scripts/lms_home.py` · MCP `lms_layers(action=home|policy|logs)`.

### Live-verified on this kit

- Stateful chain: `previous_response_id` **works** on `/api/v1/chat` (codeword probe).
- **Disable chain** when pack > `chain_max_input_chars` (logs: Context size exceeded × many).
- JSON-only LFM output works for gated roles.
- Substrate: `liquid/lfm2.5-1.2b` (ctx 4096) + nomic embed — **CPU-only** engine.
- Re-`load` without check creates `lfm:2`, `:3` — **L1 consolidates**.

---

## Gate matrix

| Fail point | Gate | Effect |
|------------|------|--------|
| LMS down | L0 **STOP** | No ops; report unreachable |
| Free RAM &lt; 1.5 GB before load | L1 **STOP** | Refuse load |
| Duplicate instances | L1 unload extras | One instance per key |
| Deep model loaded without allow | L1 unload | Keep LFM+embed only |
| Chat HTTP error | L2 **STOP** | Role fails |
| JSON unparseable | L4 retry once → **NEED_INFO** | No silent free-text OPEN |
| Missing schema keys | L4 **NEED_INFO** | Partial ops |
| FALSIFY `fatal: true` | L6 demote | OPEN_CANDIDATE → **STOP** |
| mean cosine &lt; 0.12 | L6 soft | OPEN_CANDIDATE → **NEED_INFO** |
| Partial role failure | L6 | Cannot claim structured_ops |
| Any LMS verdict | L6 | **Candidate only** — not production OPEN |

---

## Code entry points

```python
from lms_layers import layered_enter, layer_matrix, l1_ensure_substrate, l0_health

layer_matrix()                 # documentation dict
l0_health()                    # LMS up?
l1_ensure_substrate()          # one LFM + embed; unload thrash
layered_enter("intent text")   # full L0–L7 pass
```

Compat façade (existing imports unchanged):

```python
from lm_studio_client import enter_projection, LMStudio
from lfm_ops import lfm_role_pass   # defaults to layered path
```

CLI:

```bash
python prime/scripts/lms_layers.py matrix
python prime/scripts/lms_layers.py ensure
python prime/scripts/lms_layers.py ops "your intent"
```

---

## Residency policy (Snapdragon ~16 GB)

| Always on | Deep only (explicit) | Never default |
|-----------|----------------------|---------------|
| LFM 1.2B (1 instance, ctx 4096) | Bonsai 27B | Multi-model fanout |
| nomic embed | Granite tool mid | Unbounded ctx |

Unload extras after every `ensure`. Structure &gt; parameters.

---

## Live kit check (2026-08-05)

| Check | Result |
|-------|--------|
| L0.5 home | settings ctx=4096; logs ~1.9MB today; CPU-only arm64 |
| Log errors | **Context size exceeded** dominant; bad field probes fixed |
| L0 health | PASS (~30ms) |
| L1 ensure | unloaded `lfm:2`, `lfm:3`; substrate LFM+embed only |
| L3 chain | works small; **auto-off** on large dimensional packs |
| L4 JSON roles | SCOUT/FALSIFY/GLUE/VERDICT schema-gated |
| L5 glue | mean cosine ~0.70 on test enter |
| L6 policy | OPEN_CANDIDATE = measure only; FATAL demotes |
| Free RAM at ensure | ~2.2 GB (tight; Bonsai 3.8GB min blocked) |

## Future (gated, not free)

1. Ephemeral MCP on `/api/v1/chat` → Prime measure tools (model as **client of law**).
2. `/v1/responses` tools loop for tool-trained models only when free RAM allows.
3. Stream SSE events for deep_loop tick observability.
4. Session store persistence of `response_id` across Grok turns.

---

## Not claimed

- LFM equals frontier open chat  
- VERDICT is production OPEN  
- Multi-fiber vote is truth  
- MCP-in-LMS loop fully wired (path documented; default is role-ops without remote MCP)
