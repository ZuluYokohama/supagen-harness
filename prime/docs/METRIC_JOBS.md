# Two jobs, two instruments

## Canonical enter

```bash
python prime/scripts/dual_enter.py "your intent"
# MCP: exchange / enter_projection (default mode=dual)
```

`dual_enter` = KB aboutness retrieve → LFM roles → NLI agreement → **cert_face**.  
Cosine never promotes OPEN. Runner shows face + NLI, not mean_cos as glue.

## Job 1 — retrieval / aboutness → **jina** (default) / nomic (fallback)

"Which chunks bear on S08?"

| | **jina-v5-nano (default)** | **nomic v1.5 (fallback)** |
|--|--|--|
| Model | `jina-embeddings-v5-text-nano-retrieval` | `text-embedding-nomic-embed-text-v1.5` |
| Prefixes | `Query:` / `Document:` | `search_query:` / `search_document:` |
| Backend | `PRIME_JINA_BASE` (:8765 llama-server `--embedding`) | LM Studio :1234 |
| C-floor (pasta) | **~0.10** | ~0.47 |
| A−C range | **~0.83** | ~0.45 |

- Code: `nomic_metric.py` (multi-family), `dimensional_parse` retrieve
- Env: `PRIME_EMBED_FAMILY=jina|nomic`, `PRIME_JINA_BASE`, `PRIME_EMBED_FALLBACK=1`
- **Seamless jina:** every `embed()` calls `jina_service.ensure_jina()` (auto-starts
  llama-server `--embedding` on `:8765` if down). Manual:  
  `python prime/scripts/jina_service.py ensure`  
  (LMS types jina as `llm` — do **not** use LMS embeddings UI for jina.)
- Embed “system” channel = prefixes `Query:` / `Document:` (not chat system_prompt).
- Chat load context: `ctx_policy.py` — **not** LMS UI 4096/8192. Daily 1–3B → **32k**;
  LFM alone → up to **128k**; 12B+ → 4–8k. Reload if loaded_ctx ≪ policy.

**Not for:** "does this claim agree with the human?"

Probe: aboutness collapses contradiction (structural). DeBERTa / NLI stays Job 2.

## Job 2 — agreement / glue → **NLI**

"Does the domain stalk *agree* with human intent?"

- Primary: LFM structured NLI (`entailment_glue.py`) — **reason first, label last**
- Do **not** say “prefer neutral when unsure” (that was the measured prior bias)
- Reject enum-copy labels (`entailment|neutral`)
- Optional: DeBERTa-MNLI only if reason-first null fails
- Code/field: prefer **symbol jaccard** over wordpiece cosine

### NLI: smoke vs real null

```bash
python prime/scripts/null_nli.py                    # 3-pair smoke
python prime/eval_nli/score_nli_eval.py             # 65-pair real null
```

| Test | Result | Caveat |
|------|--------|--------|
| 3-pair smoke (reason→label) | 3/3 | **smoke only** |
| nli_eval_v1 con_high | 14/21 ≈ 67% | n=21; 95% CI ~46–83% **straddles 70% bar** |
| nli_eval_v1 ent_low | 1/12 ≈ 8% | n=12; **one pair**; CI huge |
| raw post-fix stream | contradiction 24%, entailment 13% | independent of eval |
| gold provenance | **model-written** | `label_source: model` |

**Not a settled “NOT EARNED”.** Underpowered cells + model golds.  
Session review of 12 `ent_low`: **disagree with 4 golds** (strict NLI) → eval needs repair.  
See `eval_nli/ent_low_human_review.md`, `nli_eval_v1_reviewed.jsonl`.

Production AGREEMENT still **provisional** until: human-labeled discriminating cells + adequate n + clear bar.

## Logogram

The **certificate** is the Arrival/semasiographic object: written whole, verifies whole, closes or not.  
Nomic is a chart. NLI is a channel. Certificate is closure.

## Null test (required before trusting aboutness numbers)

```bash
python prime/scripts/start_jina_embed.py   # once; leaves :8765 up
python prime/scripts/null_aboutness.py              # default family=jina
python prime/scripts/null_aboutness.py --family both
```

Live kit (2026-08-05, stripped + family prefixes):

| pair | jina Q/D | nomic pref |
|------|----------|------------|
| A paraphrase ceiling | ~0.93 | ~0.92 |
| B claim vs negation | ~0.62 | ~0.81 |
| C E_ref vs pasta (floor) | **~0.10** | ~0.47 |
| **dynamic range A−C** | **~0.83** | ~0.45 |
| envelope-only (legacy) | — | C ~0.83 contamination |

**Rule:** never embed raw JSON envelopes. Use `metric_text.strip_envelope`.  
**Rule:** placeholder/filler/law-core-only GLUE fail L4 content check.  
**Rule:** B≈A on cosine → NLI owns agreement.

## Content validators

`metric_text.validate_role_payload` rejects:
- `...` / ellipsis / empty
- filler ("task is ongoing", "monitoring required", …)
- GLUE `shared` that is **only** open/stop/measure/audit (prompt echo)

## Rebuild note

Indexes embedded **before** prefixes are slightly off-distribution. Rebuild:

```bash
python prime/scripts/kb_index.py build --root ...
python prime/scripts/null_aboutness.py
```
