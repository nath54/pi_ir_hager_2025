/* Linker script for STM32U545RE (NUCLEO-U545RE-Q)
 * Cortex-M33, 512KB Flash, 272KB SRAM
 *
 * Memory layout:
 * - Flash: 512KB at 0x08000000
 * - SRAM1: 192KB at 0x20000000
 * - SRAM2: 64KB at 0x20030000
 * - SRAM3: 16KB at 0x20040000 (backup SRAM)
 *
 * For simplicity, we use a contiguous 256KB RAM view at 0x20000000
 * The remaining 16KB is backup SRAM which has different retention properties.
 */

_stack_size = 0x4000;  /* 16KB stack */

MEMORY
{
    rom (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
    ram (rwx) : ORIGIN = 0x20000000, LENGTH = 256K
}

/* Include libopencm3 generic linker script */
INCLUDE libopencm3/lib/cortex-m-generic.ld
