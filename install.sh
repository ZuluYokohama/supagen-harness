#!/usr/bin/env bash
# Monorepo root install — buddy entrypoint
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUPAGEN_ROOT="$ROOT"
export PRIMEDEV_ROOT="$ROOT"
echo "SUPAGEN_ROOT=$SUPAGEN_ROOT"
bash "$ROOT/supagen/install.sh"
python3 -m supagen contract --offline
echo "Offline package OK. For live LMS: python3 -m supagen ensure && python3 -m supagen contract"
