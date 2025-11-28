#!/usr/bin/env bash
set -euo pipefail

# Halt the core via OpenOCD and leave it halted.
openocd \
	-f interface/stlink.cfg \
	-f target/stm32h7x.cfg \
	-c "init; halt; exit"


