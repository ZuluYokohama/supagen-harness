---
name: prime
description: Start Prime session — boots session MCP, plots compute graph, runs LFM modality exchange, OPEN|STOP law. One door for lineage/rplc/multiplane/sheaf + local LLM work.
argument-hint: "[intent or status|halt|exchange]"
---

# /prime

**One door (optional).** Preferred mode is **no slash at all**:
the UserPromptSubmit hook runs `runner.py` on every enter (human:Grok only).

`/prime` still works for explicit sessions / status / halt.

## Design law

```
restrict → measure → audit → OPEN | STOP
residue never forced
```

## What this command does

1. Ensure **prime-session MCP** is live (`.mcp.json`).
2. `meta_loop` / `session_start` + compute graph.
3. **`exchange`** — **mandatory** modality pass (LFM does work):
   - LFM roles: SCOUT → FALSIFY → GLUE → VERDICT  
   - Bilateral project: human ↔ code/rplc/eref/field  
   - Optional parallel smoke/rplc  
4. Continue graph: CODE only if not fatal / not all_frustrated; then AUDIT.
5. OPEN only via `audit` — never from LFM alone.

## Args

| Args | Action |
|------|--------|
| intent text | Start/continue + **exchange** on that text |
| `exchange …` | Force exchange on remainder as prompt |
| `deep <path-to-pdf>` | Ingest deep research PDF; long-horizon verify loop (hours ok) |
| `deep status` / `deep tick` | Progress / one step |
| `status` | session_status + graph_show (no LFM) |
| `halt` | cert_write + HALT |

## Process (mandatory every enter after start)

```
META_META / META
RESTRICT          (once per arc, refresh if intent shifts)
exchange          ← LFM + projection + domain measures  ★ THE TIE-IN
[CODE]            only if exchange not fatal/frustrated
MEASURE/CONDITION as needed
AUDIT → OPEN|STOP
CERT
```

**If you skip `exchange`, you skipped the local LLM doing work.** That is a process bug.

## MCP tools (modality / LLM)

| Tool | Role |
|------|------|
| **`exchange`** | Full human↔domain↔LFM pass (use this) |
| `enter_projection` | LFM roles only |
| `project_human` / `project_domain` / `project_align` | Language stalks |
| `lm_models` / `lm_load` / `lm_unload` / `lm_embed` / `lm_scout` | LMS control |
| `measure` / `measure_parallel` | smoke, rplc, eref, project |
| `meta_loop`, `restrict`, `audit`, `cert_write`, `ask_need` | law |

## What is NOT lost

- Multiplane harness: measure/harness paths + `/harness-*` when field packs needed  
- Bonsai/Granite: optional load; default fiber is **LFM only**  
- Embeddings: glue metric inside exchange  
- Explore ≠ certify: LFM verdict is **candidate**; OPEN is external  

## Deep research (hours-scale)

Load a report and loop until every work item is OPEN, STOP, or residue:

```text
/prime deep "C:\path\to\report.pdf"
```

Under the hood: `deep_ingest` → repeated `deep_tick` (LFM verify per section/op item) → `FINAL_BRIEF.md`.  
Never force-OPEN report claims without evidence. Scheduler can fire `deep tick` for multi-hour runs.

CLI:

```bash
python prime/scripts/deep_loop.py ingest --path "report.pdf"
python prime/scripts/deep_loop.py run --sleep 0.5 --max-hours 12
python prime/scripts/deep_loop.py status
```

## Accuracy

Ambiguity → `ask_need`. Do not invent instruments (e.g. backbone P2).

## User intent

$ARGUMENTS
