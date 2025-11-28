#!/usr/bin/env bash
set -euo pipefail

# Start OpenOCD, reset and run the target, then exit (assumes already flashed).
# Adjust interface/target cfgs if needed.

openocd \
	-f interface/stlink.cfg \
	-f target/stm32h7x.cfg \
	-c "init; reset run; exit"


