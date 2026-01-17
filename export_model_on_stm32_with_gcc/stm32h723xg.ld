/* Linker script for STM32H723ZG (NUCLEO-H723ZG)
 * Cortex-M7, 1MB Flash, 560KB SRAM total
 *
 * Memory layout:
 * - Flash:   1MB at 0x08000000
 * - ITCM:    64KB at 0x00000000 (fast instruction memory)
 * - DTCM:   128KB at 0x20000000 (fast data memory, stack)
 * - AXI-SRAM: 320KB at 0x24000000 (main RAM for data/bss)
 * - SRAM1:   32KB at 0x30000000 (D2 domain)
 * - SRAM2:   16KB at 0x38000000 (D3 domain)
 *
 * This script uses AXI-SRAM (RAM_D1) as main RAM as it's larger
 * and holds the main application data including AI network buffers.
 */

_stack_size = 0x4000;  /* 16KB stack */

MEMORY
{
    rom (rx)    : ORIGIN = 0x08000000, LENGTH = 1024K
    ram (rwx)   : ORIGIN = 0x24000000, LENGTH = 320K   /* AXI-SRAM (RAM_D1) */
}

/* Provide symbols for libopencm3 startup */
_stack = ORIGIN(ram) + LENGTH(ram);

/* Include libopencm3 generic linker script */
INCLUDE libopencm3/lib/cortex-m-generic.ld
