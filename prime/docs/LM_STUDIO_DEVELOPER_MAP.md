# LM Studio developer docs → Prime orthogonal use

Source: [LM Studio Developer Docs](https://lmstudio.ai/docs/developer)

## What LMS ships (mainstream stack)

| Layer | Docs promise | Typical app |
|-------|--------------|-------------|
| **lmstudio-js / lmstudio-python** | SDK clients | Chat UIs, notebooks |
| **REST `/api/v1/*`** | Stateful chat, MCP, load/unload | Local agents |
| **OpenAI compat** | `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings` | Drop-in cloud clients |
| **Anthropic compat** | `/v1/messages` | Claude-shaped clients |
| **lms CLI / llmster** | Headless daemon, CI | Servers without GUI |

### `/api/v1/chat` (the interesting one)

From [Chat docs](https://lmstudio.ai/docs/developer/rest/chat):

- `model`, `input`, `system_prompt`
- `integrations`: **ephemeral MCP** or **mcp.json plugins** (`mcp/<label>`)
- `store` + `response_id` / previous response append → **stateful fiber**
- `context_length` per request (RAM control)
- `stream`, sampling knobs, `reasoning`
- Output blocks: `message` | `tool_call` | `reasoning` | `invalid_tool_call`
- Stats: tokens, tok/s, TTFT, optional model_load_time

### MCP via API

From [MCP docs](https://lmstudio.ai/docs/developer/core/mcp):

- Model calls **tools during chat**
- Ephemeral MCP: `server_url` + `allowed_tools` per request
- Preconfigured: LMS `mcp.json` plugins
- Requires 0.4.0+ server settings flags for per-request / mcp.json MCPs

### OpenAI surface we already use

- `/v1/embeddings` — nomic 768-d metric  
- `/v1/chat/completions` — fallback single-fiber  
- `/v1/models` — simple list  

### Native surface we use

- `/api/v1/models` — loaded_instances, size, tool_use  
- `/api/v1/models/load|unload` — residency  
- `/api/v1/chat` — LFM role ops  

---

## Orthogonal use (first / outside the lines)

Mainstream: **chat with a local model.**

Ours:

```
LFM 1.2B + nomic embed  =  substrate fiber (small, resident)
Prime MCP               =  law + graph + domain measures
system_prompt roles     =  orthogonal abstract operators
enter_projection        =  SCOUT → FALSIFY → GLUE → VERDICT on ONE model
embeddings              =  glue metric between roles and human text
OPEN|STOP               =  audit authority stays in Prime, never LMS alone
```

**Why this can rival frontier ops without frontier weights:**

| Frontier default | Our move |
|------------------|----------|
| Bigger model ≈ better answers | Better **restriction maps** (roles + law) |
| Multi-agent multi-model zoo | **One fiber, multi-operator algebra** |
| Chat history as product | Stateful `response_id` as fiber memory |
| Tools as agent toys | Tools as **domain measures** under OPEN\|STOP |
| Embeddings for RAG spam | Embeddings for **projection glue** |

Structure is the accelerator. Parameters are just the carrier wave.

---

## Simplest production path (this kit)

1. Keep loaded: `liquid/lfm2.5-1.2b` + `text-embedding-nomic-embed-text-v1.5`  
2. Unload Granite/Bonsai unless experimental  
3. On each enter: `enter_projection` (lfm_ops)  
4. Then: `project_align` + `measure_parallel` (smoke/rplc)  
5. `audit` — LFM VERDICT is **candidate only**  

### Future (when you flip LMS MCP settings)

Point ephemeral MCP at a **HTTP bridge** for Prime tools, or register Prime in LMS `mcp.json`, so LFM can call `measure` / `project_align` mid-chat — model as **client of law**, not free narrator.

---

## Local home (this machine)

See **`LM_STUDIO_HOME.md`** and `scripts/lms_home.py`.

Reads `~/.lmstudio` (settings, http-server-config, model.yaml, backends, **server-logs**).  
Log-proven on kit: context overflow, no GPU offload, empty mcp.json.

## Layered gates (hyper-optimized)

See **`LMS_LAYERED_GATES.md`** and `scripts/lms_layers.py`.

```
L0 TRANSPORT → L1 RESIDENCY → L2 INFERENCE → L3 STATE FIBER
→ L4 JSON ROLES → L5 EMBED METRIC → L6 POLICY → L7 ORCHESTRATION
```

| Fail | Gate |
|------|------|
| LMS down / RAM &lt; 1.5G before load | **STOP** |
| Duplicate LFM instances | unload extras (L1) |
| JSON role unparseable | retry → **NEED_INFO** |
| FALSIFY fatal | demote OPEN_CANDIDATE → **STOP** |
| LMS verdict | **candidate measure only** |

```bash
python prime/scripts/lms_layers.py ensure
python prime/scripts/lms_layers.py ops "intent"
```

MCP: `lms_ensure`, `lms_layers`, `enter_projection` (defaults to layered).

---

## Not claimed

- LFM “equals GPT-5” on open chat  
- VERDICT from LFM is production OPEN  
- MCP-in-LMS loop is fully wired today (path is documented; default is role-ops without remote MCP)
