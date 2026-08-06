# Modality exchange — where the LLM work went

## Honest status (before `exchange`)

We built:

- LM Studio client (native v1 + embeddings)
- LFM orthogonal ops (SCOUT/FALSIFY/GLUE/VERDICT)
- Bilateral language projection
- Harness LFM scout (explore ≠ certify)
- Resource plane for Snapdragon + LMS

But those lived as **optional tools**. `/prime` docs listed only `lm_scout` / `meta_loop`.  
Nothing **forced** a local-LLM pass on every enter. Grok could “think” without LFM.

That felt like the modality fell out of the full context. It did — at the **binding** layer, not the capability layer.

## Fix: `exchange` tool

One MCP call = full cross-modal pass:

```
human enter
    → LFM role algebra (measure)
    → domain language stalks + align
    → smoke/rplc parallel (optional, default on)
    → card (verdict candidate, glue, parts)
```

**OPEN authority stays in `audit`.** LFM never OPENs alone.

## Operator contract

After `meta_loop` / session start, **every user enter → `exchange(prompt=…)`**.

| Skip | Failure mode |
|------|----------------|
| Skip exchange | Local LLM idle; Grok monologue |
| Skip projection | No bilateral glue |
| Skip audit | No certificate honesty |

## Tie-in map (full conversation)

| Thread | In exchange |
|--------|-------------|
| LFM / LMS as projection | `enter_projection` / lfm_ops |
| Language both sides | `project_*` |
| E_ref / rplc / field | domains + measure modes |
| Residue never forced | fatal → STOP candidate |
| Multiplane scouts | harness still `/harness-*` or future measure mode |
| Bonsai deep | optional load — not default |

## Commands

User still types **`/prime …`**.  
Agent must call **`exchange`**. That is the LLM tie-in.
