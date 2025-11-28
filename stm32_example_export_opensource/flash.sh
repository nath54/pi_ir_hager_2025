#!/usr/bin/env bash
set -euo pipefail

cd /home/nathan/stm32

BIN="build/blink.bin"
ELF="build/blink.elf"
if [[ ! -f "$BIN" || ! -f "$ELF" ]]; then
	echo "Build outputs not found: $BIN or $ELF"
	echo "Run ./build.sh first."
	exit 1
fi

# You can force OpenOCD by running: USE_OPENOCD=1 ./flash.sh
USE_OPENOCD=${USE_OPENOCD:-0}

flash_with_stlink() {
	# Try connect-under-reset first unconditionally, then fall back to normal
	st-flash --connect-under-reset write "$BIN" 0x08000000 || \
	st-flash write "$BIN" 0x08000000
}

flash_with_openocd() {
	openocd \
		-f interface/stlink.cfg \
		-f target/stm32h7x.cfg \
		-c "adapter speed 1000; reset_config srst_only srst_nogate connect_assert_srst; init; reset halt; program $ELF verify; reset run; exit"
}

if [[ "$USE_OPENOCD" == "1" ]]; then
	flash_with_openocd
else
	set +e
	flash_with_stlink
	RC=$?
	set -e
	if [[ $RC -ne 0 ]]; then
		echo "st-flash failed, falling back to OpenOCD..."
		flash_with_openocd
	fi
fi

echo "Flash complete."


