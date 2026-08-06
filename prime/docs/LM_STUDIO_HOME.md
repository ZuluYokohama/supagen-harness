# LM Studio local home — gate inputs

Prime reads **this machine’s** LMS install, not generic docs only.

Root: `%USERPROFILE%\.lmstudio` (override `LM_STUDIO_HOME`)

## Map

```
.lmstudio/
  settings.json              # default ctx, guardrails, JIT TTL, unload-previous
  mcp.json                   # LMS-side MCP plugins (empty on this kit)
  server-logs/YYYY-MM/*.log  # full request/response TRACE when verbose
  hub/models/*/model.yaml    # minMemory, sampling, tool_use, reasoning
  extensions/backends/*      # llama.cpp-win-arm64 engine versions
  extensions/plugins/*       # rag-v1, js-code-sandbox, …
  .internal/
    http-server-config.json  # port, bind, JIT, logging mode
    model-data.json          # lastLoaded timestamps
    backend-preferences-v1.json
    api-prediction-history/  # response_id → pack index
    gguf-metadata-cache.json
  bin/lms.exe
  conversations/
```

## This kit (observed)

| Fact | Implication for gates |
|------|------------------------|
| defaultContextLength = **4096** | Never default-load 32k on 16GB Snapdragon |
| modelLoadingGuardrails = **high ~4.3GB** | Bonsai 3.8GB min + OS = thrash; block deep unless free |
| JIT load + unload previous | L1 ensure must not fight JIT thrash |
| llama.cpp-win-arm64 **2.27.1** | Logs: **no GPU offload** → CPU-only engine |
| server log: **Context size exceeded ×18** | Cap dimensional packs; disable fiber chain on large inputs |
| mcp.json empty | LMS plugins not wired; use Prime MCP / ephemeral_mcp |
| LFM model.yaml temp **0.1** top_p **0.1** | Align structured roles with hub defaults |
| Granite min ~**4.5GB**, Bonsai ~**3.8GB** | DEEP_ONLY residency |
| logSensitiveData / verbose full | Useful for audit; do not paste logs into public OPEN claims |

## CLI / MCP

```bash
python prime/scripts/lms_home.py policy
python prime/scripts/lms_home.py logs
python prime/scripts/lms_layers.py home
```

MCP: `lms_layers(action="home"|"policy"|"logs")`

## Wired into stack

`lms_layers.l4_ops_pass` starts with **L0.5**:

- pack_budget_chars (2800 when context thrash seen)
- chain_max_input_chars (2200) → fiber chain OFF if larger
- cpu_only_engine flag
- recommendations from logs + model.yaml

See `LMS_LAYERED_GATES.md`.
