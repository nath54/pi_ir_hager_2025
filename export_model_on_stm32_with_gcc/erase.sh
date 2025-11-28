#!/usr/bin/env bash
set -euo pipefail

# Mass erase internal flash and reset the board.
# Tries st-flash (connect-under-reset) first, then OpenOCD fallback.

erase_with_stlink() {
	# st-flash mass erase (use connect-under-reset if supported)
	if st-flash --help 2>&1 | grep -q -- "connect-under-reset"; then
		st-flash --connect-under-reset erase
	else
		st-flash erase
	fi
}

erase_with_openocd() {
	openocd \
		-f interface/stlink.cfg \
		-f target/stm32h7x.cfg \
		-c "adapter speed 1000; reset_config srst_only srst_nogate connect_assert_srst; init; reset halt; stm32h7x mass_erase 0; reset run; exit"
}

set +e
erase_with_stlink
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
	echo "st-flash erase failed, falling back to OpenOCD..."
	erase_with_openocd
fi

echo "Mass erase complete and board reset."



