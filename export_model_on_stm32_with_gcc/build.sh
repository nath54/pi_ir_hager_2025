#!/usr/bin/env bash
set -euo pipefail

# Build script with optional "minimal" argument
# Usage: 
#   ./build.sh         - builds with main.c (full version with semihosting)
#   ./build.sh minimal - builds with main_minimal_test.c (LED-only version)
#   ./build.sh debug   - builds with main.c (full version with semihosting, debug flags)
#   ./build.sh minimal debug - builds with main_minimal_test.c (LED-only, debug flags)

# Default to main.c
MAIN_SRC="main.c"
MAKE_FLAGS=""
BUILD_TYPE="RELEASE (Fast, No Semihosting)"

# Parse arguments
for arg in "$@"
do
    if [ "$arg" == "debug" ]; then
        MAKE_FLAGS="$MAKE_FLAGS DEBUG=1"
        BUILD_TYPE="DEBUG (Semihosting Enabled)"
    fi
    if [ "$arg" == "int" ]; then
        MAKE_FLAGS="$MAKE_FLAGS INT_QUANTIZATION=1"
        BUILD_TYPE="$BUILD_TYPE (INT8 Quantized)"
    fi
done

echo "Building $BUILD_TYPE"
echo "Source: $MAIN_SRC"

# Clean and build
make clean
make MAIN_SRC=$MAIN_SRC $MAKE_FLAGS

echo ""
echo "======================================"
echo "Build complete: build/blink.elf"
echo "Source used: $MAIN_SRC"
echo "======================================"
