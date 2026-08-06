# Buddy install — replicate this kit

## One door

```powershell
git clone <THIS_REPO_URL>
cd <repo-root>
.\install.ps1                 # sets SUPAGEN_ROOT, pip install -e supagen, smoke+contract offline
python -m supagen ensure      # jina :8765 + LMS fiber @ ctx_policy
python -m supagen contract    # hard live gates
python -m supagen e2e --live
```

Unix: `bash install.sh`

Full docs: [`supagen/README.md`](supagen/README.md) · monorepo: [`README.md`](README.md)

### Acceptance checklist

| Step | Command | Expect |
|------|---------|--------|
| Offline | `python -m supagen verify` | rc=0 |
| Live | `python -m supagen verify --live` | contract 21/21, e2e PASS |
| Doctor | `python -m supagen doctor` | `import_nomic_metric: true`, jina ok, KB family=jina |
| Enter | `python -m supagen enter "…"` | cert_face + not_open_authority |
| Query | `python -m supagen query "dual enter aboutness"` | top hit METRIC / dual_enter docs |

## What you get

| Layer | Always-on behavior |
|-------|-------------------|
| **Job1 aboutness** | jina auto-starts on `:8765` (LMS cannot host it as embedder) |
| **Job2 agreement** | NLI / dual_enter face — cosine never OPEN |
| **Chat fiber** | LMS load via **ctx_policy** (32k daily 1–3B when RAM allows; not UI 4k/8k) |
| **Field harness** | `supagen harness smoke` / pipeline packs / OPEN\|STOP certify |

## Smoke contract (must pass)

```text
python -m supagen smoke          # offline packs + imports
python -m supagen e2e --live     # + jina floor/ceiling + dual_enter
python -m supagen doctor         # if anything red
```

Measured green on build machine (2026-08-06): offline PASS; live aboutness floor≈0.23 ceiling≈0.94; dual_enter **required** PASS (NEED_INFO face); LFM promoted to **128k** after heavies unloaded; jina auto-restart on dead port.
