# PRIMEdEV / supagen

**Super agent harness:** dual-metric Prime stack (jina aboutness + NLI agreement + LM Studio fibers) and multiplane OPEN|STOP field harness.

Design law: `restrict → measure → audit → OPEN | STOP` — residue never forced. Cosine never promotes OPEN.

## Buddy install (replicate)

```powershell
git clone https://github.com/ZuluYokohama/supagen-harness.git
cd <repo-root>
.\install.ps1
# → pip install -e supagen + bootstrap .pth (no PYTHONPATH) + offline verify
```

```bash
git clone https://github.com/ZuluYokohama/supagen-harness.git
cd <repo-root>
bash install.sh
```

Then (LM Studio local server on `:1234`):

```bash
python -m supagen ensure
python -m supagen verify --live   # 21-gate contract + e2e + harness
python -m supagen enter "your intent"
python -m supagen query "dual enter aboutness"
```

Full CLI: [`supagen/README.md`](supagen/README.md) · one-page: [`BUDDY_INSTALL.md`](BUDDY_INSTALL.md)

## Layout

| Path | Role |
|------|------|
| `supagen/` | Installable CLI package |
| `prime/scripts/` | dual_enter, jina_service, ctx_policy, residency, metrics |
| `harness/` | multiplane packs / scouts / certify |
| `123abc/` | rplc-sheaf ALU (optional domain measure) |

## Requirements

- Python ≥ 3.11  
- LM Studio (for live chat fiber)  
- jina GGUF auto-discovered under LMS downloads folder (or `PRIME_JINA_GGUF=`)  
- Optional: `pip install -e "./supagen[nli]"` for DeBERTa holonomy judge  

## Publish (maintainer)

```powershell
cd <repo-root>
git init -b main   # if needed
git config --global --add safe.directory <repo-root>   # if "dubious ownership"
git add .
git commit -m "supagen 0.3.1 buddy-ready"
# create GH repo, then:
git remote add origin https://github.com/ZuluYokohama/supagen-harness.git
git push -u origin main
```

## Law

Local models **measure**. Production OPEN only after domain audit. Residue never forced.
