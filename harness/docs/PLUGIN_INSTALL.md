# Install multiplane-harness plugin

Plugin root: `C:\PRIMEdEV-1\harness`  
Manifest: `.claude-plugin/plugin.json`

## Grok CLI (done on this machine 2026-08-05)

```bash
grok plugin validate C:\PRIMEdEV-1\harness
grok plugin install C:\PRIMEdEV-1\harness --trust
grok plugin enable multiplane-harness
```

Then **new session** or Plugins tab → **`r` reload**.

Installed id: `c--primedev-1-harness-ef69127f` (local → `C:\PRIMEdEV-1\harness`).  
Enabled in `~/.grok/config.toml` under `[plugins].enabled`.

## Commands after install

| Command | Purpose |
|---------|---------|
| **`/harness-pipeline`** | Main: pack ± scout → certify |
| **`/harness-smoke`** | Offline + optional LMS smoke |
| **`/harness-ingest`** | Dump + `.emz` → inventory pack |
| **`/harness-certify`** | Bundle JSON → OPEN\|STOP |

If names collide, use qualified form e.g. `/multiplane-harness:harness-pipeline`.

## Without plugin UI

```bash
cd C:\PRIMEdEV-1
# preferred: unified package
.\supagen\install.ps1
supagen harness smoke
supagen ensure
supagen harness pipeline --pack filmore_magpi

# or raw:
python harness/smoke_offline.py
python harness/pipeline/v1/pipeline.py --pack filmore_magpi
python harness/pipeline/v1/pipeline.py --pack filmore_magpi --live-scout lfm
```

See `supagen/README.md` for buddy GH install + seamless jina/ctx.

## Related

- rplc-sheaf: numeric OPEN/STOP ALU  
- Lineage: `C:\PRIMEdEV-1\LINEAGE_CHARTER.md`
