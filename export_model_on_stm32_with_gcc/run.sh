#!/usr/bin/env bash
set -euo pipefail

# Start OpenOCD with semihosting enabled
# This will show debug printf output in the terminal
# Press Ctrl+C to stop

openocd \
	-f interface/stlink.cfg \
	-f target/stm32h7x.cfg \
	-c "init" \
	-c "arm semihosting enable" \
	-c "reset run"


