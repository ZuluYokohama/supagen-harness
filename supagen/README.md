# supagen — super agent harness

**One package:** Prime dual-metric stack (jina aboutness + DeBERTa/NLI agreement + LM Studio fibers) **and** multiplane OPEN|STOP field harness.

Design law:

```
restrict → measure → audit → OPEN | STOP
residue never forced
```

Cosine never promotes OPEN. Jina is Job1 (aboutness). Agreement is Job2 (NLI). Certificates close as measures, not vibes.

---

## Buddy install (replicate from a link)

### From this monorepo (today)

```powershell
# Windows
cd C:\PRIMEdEV-1
.\supagen\install.ps1
```

```bash
# Unix
cd /path/to/PRIMEdEV-1
bash supagen/install.sh
```

### From GitHub

```bash
git clone https://github.com/<YOU>/<REPO>.git
cd <REPO>
# Windows:
.\install.ps1
# Unix:
bash install.sh
```

Sets `SUPAGEN_ROOT`, `pip install -e supagen`, runs `supagen verify` (offline contract + smoke + harness).

Acceptance after LMS is up:

```bash
python -m supagen ensure
python -m supagen verify --live   # contract 21 gates + e2e
```

### Optional

```bash
pip install -e "./supagen[nli]"   # DeBERTa holonomy judge
```

---

## After install — 60 second path

```bash
# 1) LM Studio: start local server :1234; load liquid/lfm2.5-1.2b OR mistralai/ministral-3-3b
# 2) Substrate (auto-starts jina on :8765 — NOT the LMS embeddings UI)
supagen ensure

# 3) Full E2E
supagen e2e --live

# 4) Operator enter
supagen enter "measure whether this claim is about E_ref or pasta"

# 5) Field harness (offline packs always work)
supagen harness smoke
supagen harness pipeline --pack filmore_magpi
```

---

## CLI

| Command | What |
|---------|------|
| `supagen smoke` | Offline only |
| `supagen e2e` | Offline + live if LMS up |
| `supagen e2e --live` | Require LMS |
| `supagen ensure` | **Seamless** jina + chat fiber @ **ctx_policy** (not UI 4k/8k) |
| `supagen status` | jina + loaded ctx vs policy |
| `supagen doctor` | Failures + fixes |
| `supagen contract` | **Hard CI gates** (buddy must pass) |
| `supagen serve` | Watchdog loop: keep jina `:8765` alive |
| `supagen reindex-kb` | Rebuild KB with current Job1 family (jina) |
| `supagen query "…"` | Hybrid retrieve (jina cos × lex + BM25) |
| `supagen aboutness` | A/B/C null (jina default) |
| `supagen enter "…"` | dual_enter (aboutness + agreement face) |
| `supagen harness smoke\|pipeline\|certify\|ingest` | multiplane field |

---

## Hardening (no sandbagging)

| Upgrade | Behavior |
|---------|----------|
| **Every dual_enter** | `seamless_substrate`: jina ensure + unload heavies + promote fiber ctx |
| **JSON fences** | Ministral-style ` ```json ` stripped before NLI/parse |
| **Token pack budget** | KB packs capped (~3.5k tok) to stop context thrash |
| **Jina watchdog** | Dead port + live PID → auto restart |
| **ctx_policy** | Free≥5GB LFM→128k; 1–3B→32k; not LMS UI 4k/8k |

## Architecture (what must stay seamless)

```
┌─────────────────────────────────────────────────────────┐
│  supagen CLI / agent hooks                              │
├───────────────┬─────────────────────┬───────────────────┤
│ Job1 ABOUTNESS│ Job2 AGREEMENT      │ Chat fiber        │
│ jina :8765    │ DeBERTa / LFM NLI   │ LMS :1234         │
│ auto-ensure   │ never cosine OPEN   │ ctx_policy load   │
│ Query:/Doc:   │                     │ 32k daily 1–3B    │
└───────────────┴─────────────────────┴───────────────────┘
         │                    │
         ▼                    ▼
   multiplane harness    rplc-sheaf / cert face
   pack → scout → certify → OPEN|STOP
```

### Why jina is not in the LMS “embeddings” list

LMS types `jina-embeddings-v5-text-nano-retrieval` as **`llm`**. `/v1/embeddings` on `:1234` will not serve it (silent nomic remap or 400). Supagen runs the **same GGUF** via `llama-server --embedding` on **`:8765`**, auto-started by `supagen ensure` / every `embed()`.

### Why models were loading at 8192

Legacy defaults used LMS UI `defaultContextLength` (4k/8k). **`ctx_policy`** now loads by model size + free RAM (e.g. LFM→128k when free≥5GB, Ministral→32k, 12B→4–8k).

---

## Layout

| Path | Role |
|------|------|
| `supagen/` | This installable CLI package |
| `prime/scripts/` | dual_enter, jina_service, ctx_policy, nomic_metric, LMS layers |
| `harness/` | multiplane packs, scouts, certify |
| `123abc/` | rplc-sheaf ALU (optional domain measure) |

Env:

| Var | Meaning |
|-----|---------|
| `SUPAGEN_ROOT` / `PRIMEDEV_ROOT` | Monorepo root |
| `PRIME_EMBED_FAMILY` | `jina` (default) \| `nomic` |
| `PRIME_JINA_BASE` | default `http://127.0.0.1:8765` |
| `PRIME_JINA_AUTO_START` | `1` (default) |
| `PRIME_LOAD_CTX` | force load context |
| `LM_STUDIO_BASE` | default `http://127.0.0.1:1234` |

---

## Law reminder

- **ACCEPTED ≠ production OPEN** without domain audit.
- Residue never forced.
- Explore (scout) ≠ certify (external gate).

---

## Troubleshooting

```bash
supagen doctor
# jina log:  prime/state/jina_embed.log
# reload chat to policy ctx when free RAM recovers:
supagen ensure --model liquid/lfm2.5-1.2b
```
