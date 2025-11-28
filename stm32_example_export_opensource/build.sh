#!/usr/bin/env bash
set -euo pipefail

cd /home/nathan/stm32
make clean && make all

echo "Build complete: build/blink.elf"


