#!/usr/bin/env bash
set -euo pipefail

# Default to Release build, can override with: BUILD_TYPE=Debug ./flash.sh
BUILD_TYPE=${BUILD_TYPE:-Release}
ELF="build/${BUILD_TYPE}/test.elf"
BIN="build/${BUILD_TYPE}/test.bin"

if [[ ! -f "$ELF" ]]; then
	echo "Build output not found: $ELF"
	echo "Run: cmake --build --preset ${BUILD_TYPE}"
	exit 1
fi

# Convert ELF to binary for st-flash
echo "Converting ELF to binary..."
arm-none-eabi-objcopy -O binary "$ELF" "$BIN"
echo "Binary size: $(stat -c%s "$BIN") bytes"

# You can force OpenOCD by running: USE_OPENOCD=1 ./flash.sh
USE_OPENOCD=${USE_OPENOCD:-0}

flash_with_stlink() {
	# STM32U5 flash base address is 0x08000000
	st-flash --connect-under-reset write "$BIN" 0x08000000 || \
	st-flash write "$BIN" 0x08000000
}

flash_with_openocd() {
	# STM32U5 with TrustZone needs special handling
	openocd \
		-f interface/stlink.cfg \
		-f target/stm32u5x.cfg \
		-c "init" \
		-c "reset halt" \
		-c "flash write_image erase $ELF" \
		-c "verify_image $ELF" \
		-c "reset run" \
		-c "exit"
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
