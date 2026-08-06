# Supagen buddy install (Windows)
#   irm https://raw.githubusercontent.com/<org>/supagen/main/install.ps1 | iex
#   or:  .\install.ps1
#   or monorepo:  .\supagen\install.ps1

$ErrorActionPreference = "Stop"

function Find-Root {
  if ($env:SUPAGEN_ROOT -and (Test-Path $env:SUPAGEN_ROOT)) { return (Resolve-Path $env:SUPAGEN_ROOT).Path }
  $here = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
  $cands = @(
    $here,
    (Join-Path $here ".."),
    "C:\PRIMEdEV-1",
    (Join-Path $here "supagen")
  )
  foreach ($c in $cands) {
    $p = Resolve-Path $c -ErrorAction SilentlyContinue
    if (-not $p) { continue }
    $root = $p.Path
    if (Test-Path (Join-Path $root "supagen\pyproject.toml")) { return $root }
    if (Test-Path (Join-Path $root "pyproject.toml")) {
      $name = (Get-Content (Join-Path $root "pyproject.toml") -Raw)
      if ($name -match 'name\s*=\s*"supagen"') { return (Split-Path $root -Parent) }
    }
  }
  return $here
}

$Root = Find-Root
# Prefer monorepo root (parent of supagen/) when install.ps1 lives in supagen/
if ((Split-Path $Root -Leaf) -eq "supagen" -and (Test-Path (Join-Path (Split-Path $Root -Parent) "prime"))) {
  $Root = (Resolve-Path (Split-Path $Root -Parent)).Path
}
$Supagen = if (Test-Path (Join-Path $Root "supagen\pyproject.toml")) { Join-Path $Root "supagen" } else { $Root }
Write-Host "SUPAGEN_ROOT=$Root"
Write-Host "package=$Supagen"

$env:SUPAGEN_ROOT = $Root
$env:PRIMEDEV_ROOT = $Root
try {
  [Environment]::SetEnvironmentVariable("SUPAGEN_ROOT", $Root, "User")
  [Environment]::SetEnvironmentVariable("PRIMEDEV_ROOT", $Root, "User")
  Write-Host "User env SUPAGEN_ROOT/PRIMEDEV_ROOT persisted"
} catch {
  Write-Host "User env set skipped: $_"
}

# Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "python not on PATH" }

Write-Host "== pip install -e supagen =="
& python -m pip install -U pip setuptools wheel
& python -m pip install -e $Supagen
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "== bootstrap import paths (.pth) =="
& python -m supagen bootstrap
if ($LASTEXITCODE -ne 0) {
  Write-Host "bootstrap WARN — using ensure_sys_path fallback" -ForegroundColor Yellow
}

Write-Host "== package verify (offline) =="
& python -m supagen verify
if ($LASTEXITCODE -ne 0) {
  Write-Host "verify FAIL — run: python -m supagen doctor" -ForegroundColor Red
}

# Put user Scripts on PATH for this session + permanently (user scope)
$pyScripts = Join-Path $env:APPDATA "Python\Python314\Scripts"
if (-not (Test-Path $pyScripts)) {
  $pyScripts = python -c "import sysconfig; print(sysconfig.get_path('scripts'))" 2>$null
}
if ($pyScripts -and (Test-Path $pyScripts)) {
  $env:Path = "$pyScripts;$env:Path"
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if ($userPath -notlike "*$pyScripts*") {
    [Environment]::SetEnvironmentVariable("Path", "$pyScripts;$userPath", "User")
    Write-Host "Added to User PATH: $pyScripts"
  }
}

Write-Host ""
Write-Host "Install OK. Use:  python -m supagen <cmd>   (or supagen after new terminal)"
Write-Host "Next:"
Write-Host "  1) Start LM Studio local server (:1234); load LFM 1.2B or Ministral 3B"
Write-Host "  2) python -m supagen ensure     # jina :8765 auto + policy ctx"
Write-Host "  3) python -m supagen e2e --live"
Write-Host "  4) python -m supagen enter `"your intent`""
Write-Host "  5) python -m supagen harness smoke"
Write-Host ""
Write-Host "Optional NLI judge:  pip install -e `"$Supagen[nli]`""
