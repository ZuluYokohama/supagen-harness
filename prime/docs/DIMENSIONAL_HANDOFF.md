# Dimensional handoff — Grok parse ↔ LFM fiber

## Best form to pass (not everything)

| Form | Who uses it | Why |
|------|-------------|-----|
| **Full PDF text** | Index builder only | Too big / noisy for LFM |
| **Chunks + embed vectors** | Retriever (machine) | Metric space for glue |
| **Top-k scored packs** | **LFM roles** | Small, high-signal language stalks |
| **Slim retrieval JSON** | **Grok** | Plan/code without drowning context |
| **Cosine scores** | Both | Dimensional “how aligned” without raw R^768 in chat |

**Vectors stay on disk** (`dimensional_index.json`).  
**Language packs** move to LFM.  
**Structured previews + scores** move to Grok.

## Pipeline

```
PDF
 → extract text
 → chunk (page/section windows)
 → embed (nomic 768-d)     [local LMS]
 → index on disk
 → query = work item + goal
 → retrieve top-k by cosine
 → pack_for_lfm  → LFM SCOUT/FALSIFY/GLUE/VERDICT
 → pack_for_grok → Grok plans / codes / audits
 → measures (tests, rplc) → OPEN|STOP
```

## MCP / CLI

```
doc_parse(path, query)     # build/load index + retrieve
deep_ingest / deep_tick    # long loop uses retrieval packs automatically
```

```bash
python prime/scripts/deep_loop.py ingest --path report.pdf
# builds dimensional_index.json during ingest
python prime/scripts/deep_loop.py run
```

## Maximal capability = division of labor

| Actor | Does |
|-------|------|
| **Grok** | Parse intent, use `doc_parse`/`grok_pack`, tools, code, final OPEN |
| **Embed** | Geometry of language (distance, retrieval, role glue) |
| **LFM** | Cheap multi-role measure on **retrieved packs only** |
| **Human** | Goals and ambiguous NEED_INFO |

No raw “stuff the whole PDF into the 1.2B model.”  
That is the premium path.
