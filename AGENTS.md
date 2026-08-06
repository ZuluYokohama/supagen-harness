# PRIMEdEV — pure runner process (human : Grok only)

## Surface

You talk to **Grok CLI only**. Do not send the user shopping plugins or slash zoos.

**Buddy / install path:** `supagen/` — `.\supagen\install.ps1` then `supagen ensure` / `supagen e2e`.

Under the hood (silent):

1. **UserPromptSubmit hook** runs `prime/scripts/runner.py` → local fiber + **jina aboutness** (auto `:8765`) + NLI face.
2. That card is injected as `additionalContext` (PRIME RUNNER CARD).
3. You act with tools/files under design law.
4. Chat loads use **ctx_policy** (not LMS UI 4k/8k).

## Design law

```
restrict → measure → audit → OPEN | STOP
residue never forced
```

## Every user enter

1. Read **PRIME RUNNER CARD** if present in context (from hook).
2. If card missing and task is non-trivial, call MCP `exchange` or run:
   `python -m supagen enter "<user text>"`  
   (or `python prime/scripts/runner.py --prompt "..."` from monorepo root)
3. Honor card:
   - `fatal=true` or `all_frustrated=true` → no CODE; STOP or ask user.
   - LFM `VERDICT` is **candidate only** — never OPEN from LFM alone.
4. Do real work (edit/test/rplc/derive) only when card allows.
5. OPEN only after real measures survive audit.

## Local LLM

- Default fiber: `liquid/lfm2.5-1.2b` (or already-loaded small chat) via LMS `:1234`.
- Job1 aboutness: **jina** on `:8765` (auto-ensure; not LMS embeddings UI).
- Job2 agreement: NLI (never cosine). Cosine never promotes OPEN.
- Orthogonal roles: SCOUT → FALSIFY → GLUE → VERDICT.
- Loads use **ctx_policy** (32k daily / 128k LFM when RAM allows).

## Domains in this tree

| Path | Role |
|------|------|
| `prime/` | Runner + MCP session |
| `123abc/` / rplc-sheaf | OPEN\|STOP ALU |
| `topology-sees-sequence/` | E_ref sequence prior; P2 parked |
| `harness/` | Multiplane field packs |
| `LINEAGE_CHARTER.md` | North star |

## Forbidden UX

- Listing plugin command menus unless user asks.
- Force-OPEN.
- Inventing backbone energy / P2 instruments.
- Skipping the runner when LMS is up.

## Manual runner

```bash
python C:\PRIMEdEV-1\prime\scripts\runner.py --prompt "your enter"
# or heavy domain measures:
set PRIME_RUNNER_HEAVY=1
```

## Deep research long loop (PDF → verify for hours)

```bash
python C:\PRIMEdEV-1\prime\scripts\deep_loop.py ingest --path "C:\path\to\report.pdf"
python C:\PRIMEdEV-1\prime\scripts\deep_loop.py run --max-hours 12 --sleep 0.3
python C:\PRIMEdEV-1\prime\scripts\deep_loop.py status
# brief: prime/state/deep/FINAL_BRIEF.md
```

Loop until all items OPEN/STOP/residue. Residue never forced. Grok may continue other work; deep_loop is the waiter.
