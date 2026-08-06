# Monorepo root install — buddy entrypoint
#   git clone <repo> ; cd <repo> ; .\install.ps1
$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$env:SUPAGEN_ROOT = (Resolve-Path $Root).Path
$env:PRIMEDEV_ROOT = $env:SUPAGEN_ROOT
Write-Host "SUPAGEN_ROOT=$env:SUPAGEN_ROOT"
& "$Root\supagen\install.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
Write-Host "=== full verify ==="
& python -m supagen contract --offline
if ($LASTEXITCODE -ne 0) { Write-Host "offline contract FAIL" -ForegroundColor Red; exit 1 }
Write-Host "Offline package OK. For live LMS:"
Write-Host "  python -m supagen ensure"
Write-Host "  python -m supagen contract"
Write-Host "  python -m supagen e2e --live"
