#!/usr/bin/env bash
# Supagen buddy install (Unix)
#   curl -fsSL https://raw.githubusercontent.com/<org>/supagen/main/install.sh | bash
set -euo pipefail

find_root() {
  if [[ -n "${SUPAGEN_ROOT:-}" && -d "$SUPAGEN_ROOT" ]]; then
    echo "$SUPAGEN_ROOT"; return
  fi
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
  for c in "$here" "$here/.." "$here/../.." "$HOME/PRIMEdEV-1"; do
    if [[ -f "$c/supagen/pyproject.toml" ]]; then echo "$(cd "$c" && pwd)"; return; fi
    if [[ -f "$c/pyproject.toml" ]] && grep -q 'name = "supagen"' "$c/pyproject.toml" 2>/dev/null; then
      echo "$(cd "$c/.." && pwd)"; return
    fi
  done
  echo "$here"
}

ROOT="$(find_root)"
if [[ -f "$ROOT/supagen/pyproject.toml" ]]; then
  PKG="$ROOT/supagen"
else
  PKG="$ROOT"
fi
export SUPAGEN_ROOT="$ROOT"
export PRIMEDEV_ROOT="$ROOT"
echo "SUPAGEN_ROOT=$ROOT"
echo "package=$PKG"

python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e "$PKG"

echo "== bootstrap import paths =="
python3 -m supagen bootstrap || echo "bootstrap warn"

echo "== offline verify =="
python3 -m supagen verify || echo "verify had failures — run: python3 -m supagen doctor"

cat <<EOF

Install OK. Next:
  1) Start LM Studio local server (:1234)
  2) supagen ensure
  3) supagen e2e --live
  4) supagen enter "your intent"
  5) supagen harness smoke

Optional NLI: pip install -e "${PKG}[nli]"
EOF
