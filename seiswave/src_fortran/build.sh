#!/bin/bash
# Cross-platform build wrapper for CPS cps_core extension.
# Delegates to build.py for all the heavy lifting.
# Usage: bash build.sh [--clean]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/build.py" "$@"
