/* Minimal linker script for STM32H723ZG.
 * Flash assumed 1024K at 0x08000000 (xG suffix), RAM uses DTCM at 0x20000000 (128K).
 * Adjust sizes if your exact device differs.
 */

MEMORY
{
	rom (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
	ram (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

INCLUDE libopencm3/lib/cortex-m-generic.ld


