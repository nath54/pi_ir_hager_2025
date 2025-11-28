#!/usr/bin/env bash
set -euo pipefail

# Build script with optional "minimal" argument
# Usage: 
#   ./build.sh         - builds with main.c (full version with semihosting)
#   ./build.sh minimal - builds with main_minimal_test.c (LED-only version)

if [ "${1:-}" = "minimal" ]; then
    echo "Building MINIMAL version (main_minimal_test.c - LED-only, no semihosting)"
    MAIN_SRC=main_minimal_test.c
else
    echo "Building NORMAL version (main.c - with semihosting debug)"
    MAIN_SRC=main.c
fi

# Clean and build
make clean
make all MAIN_SRC="$MAIN_SRC"

echo ""
echo "======================================"
echo "Build complete: build/blink.elf"
echo "Source used: $MAIN_SRC"
echo "======================================"
