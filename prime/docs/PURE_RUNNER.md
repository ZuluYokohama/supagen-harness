# Pure runner — human : Grok CLI only

## What you installed

| Piece | Path |
|-------|------|
| Runner script | `prime/scripts/runner.py` |
| Global hook | `~/.grok/hooks/prime-runner.json` |
| Workspace law | `PRIMEdEV-1/AGENTS.md` |
| Rule | `PRIMEdEV-1/.grok/rules/00-prime-runner.md` |
| Last card | `prime/state/runner/last_exchange.md` |

## Flow

```
you hit enter
  → UserPromptSubmit hook
  → runner.py → LFM roles + embed + projection
  → additionalContext: PRIME RUNNER CARD
  → Grok sees card + your text
  → Grok acts (tools) under OPEN|STOP
  → you read one reply
```

No `/prime` required. Plugins stay installed but **invisible**.

## Env

| Var | Meaning |
|-----|---------|
| `PRIME_STATE_DIR` | Default `prime/state/runner` |
| `PRIME_RUNNER_HEAVY=1` | Include parallel smoke/rplc in every enter (slower) |

Default enter is **light**: LFM ops + bilateral projection only (fast).

## Trust

Project hooks need folder trust once: `/hooks-trust` or open Hooks UI.

## Offline / LMS down

Runner fail-opens: card says FAIL; Grok continues with law but without local LFM.
