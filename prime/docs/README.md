# Prime — one door

Higher-domain operator stack: **compute graph + session MCP + OPEN|STOP**.

Not another dozen slash commands. **`/prime`** is the product surface.

## Install (Grok)

```bash
grok plugin install C:\PRIMEdEV-1\prime --trust
# reload plugins / new session
```

Or copy to `~/.grok/installed-plugins/prime` and enable.

MCP is declared in `.mcp.json` (`prime-session`).

## Pure path (preferred)

Human : Grok only — no slash required.

- Hook: `~/.grok/hooks/prime-runner.json` → `runner.py` on every enter  
- Law: `PRIMEdEV-1/AGENTS.md`  
- See `docs/PURE_RUNNER.md`

## Explicit path

```
/prime certify rplc residue batch under design law
```

Agent should:

1. `meta_loop` / session start  
2. show graph  
3. `restrict`  
4. `measure` (smoke, rplc, lm_models, …)  
5. `condition_pulse` if coding  
6. `audit` OPEN|STOP  
7. `cert_write`  

## Local LM Studio

Detected default: `http://127.0.0.1:1234` (native `/api/v1` + OpenAI `/v1`).  
Substrate: **LFM 1.2B + nomic embed** only by default.

**Layered gates (L0–L7):** see `LMS_LAYERED_GATES.md` — residency, stateful fiber, JSON roles, policy.  
Scouts are **measures**, not OPEN proofs.

## Design law

Residue never forced. Force-OPEN is blocked in `audit` without restrict + measures.

## Language = bilateral projection

See `LANGUAGE_PROJECTION.md`. Human intent and domain speak (code/rplc/eref/field) are both projections; `project_*` + `measure(project)` glue them before CODE/OPEN.

```
/prime operationalize bilateral language projection on rplc + eref
```

## Layout

```
prime/
  commands/prime.md
  agents/prime-operator.md
  skills/prime/SKILL.md
  scripts/
    mcp_server.py
    session_store.py
    compute_graph.py
    measures.py
    condition_gate.py
  state/   # session.json at runtime
  docs/
```
