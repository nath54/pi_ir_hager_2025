/**
 * @file init_config.c
 * @brief System initialization implementation
 *
 * Implements device-agnostic initialization for clock, cache, MPU, GPIO, UART.
 */

#include "init_config.h"
#include "device_config.h"

// =============================================================================
// BUILD OPTIONS
// =============================================================================

// SAFE_MODE: Skip advanced init that may hang (PLL, cache, MPU)
// Set via make SAFE_MODE=1
#ifdef SAFE_MODE
    #define SKIP_ADVANCED_INIT  1
#else
    #define SKIP_ADVANCED_INIT  0
#endif

// Maximum timeout iterations (prevents infinite hangs)
#define INIT_TIMEOUT_MAX  100000UL

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

volatile uint32_t system_millis = 0;

// Timeout for clock configuration (rough estimate)
#define CLOCK_TIMEOUT  10000000UL

// =============================================================================
// SYSTICK INTERRUPT HANDLER
// =============================================================================

void sys_tick_handler(void) {
    system_millis++;
}

// =============================================================================
// FPU ENABLE
// =============================================================================

void fpu_enable(void) {
#if DEVICE_HAS_FPU
    // Enable CP10 and CP11 full access
    SCB_CPACR |= (0xF << 20);
    __asm__ volatile("dsb");
    __asm__ volatile("isb");
#endif
}

// =============================================================================
// MPU CONFIGURATION
// =============================================================================

void mpu_config(void) {
#if SKIP_ADVANCED_INIT
    // Safe mode: just disable MPU, don't configure
    MPU_CTRL = 0;
    return;
#endif

#if defined(DEVICE_STM32H723ZG)
    // Disable MPU first
    MPU_CTRL = 0;
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

    // Enable MPU with privileged default mode
    // This allows background region access for better compatibility
    MPU_CTRL = MPU_CTRL_ENABLE | MPU_CTRL_PRIVDEFENA;
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

#elif defined(DEVICE_STM32U545RE)
    // U5 series MPU configuration
    // Disable MPU
    MPU_CTRL = 0;
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

    // Enable with privileged default
    MPU_CTRL = MPU_CTRL_ENABLE | MPU_CTRL_PRIVDEFENA;
    __asm__ volatile("dsb");
    __asm__ volatile("isb");
#endif
}

// =============================================================================
// CACHE CONFIGURATION
// =============================================================================

void cache_enable(void) {
#if SKIP_ADVANCED_INIT
    // Safe mode: skip cache enable entirely
    return;
#endif

#if DEVICE_HAS_ICACHE
    // Invalidate I-cache before enabling
    __asm__ volatile("dsb");
    __asm__ volatile("isb");
    SCB_ICIALLU = 0;  // Invalidate entire I-cache
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

    // Enable I-cache
    SCB_CCR |= SCB_CCR_IC;
    __asm__ volatile("dsb");
    __asm__ volatile("isb");
#endif

#if DEVICE_HAS_DCACHE
    // D-cache requires proper invalidation before enabling
    // Get cache size info from CCSIDR register
    __asm__ volatile("dsb");

    // Select data cache (level 1, data)
    // SCB_CSSELR = 0 selects D-cache
    volatile uint32_t *csselr = (volatile uint32_t *)0xE000ED84;
    *csselr = 0;
    __asm__ volatile("dsb");

    // Read CCSIDR for cache geometry
    volatile uint32_t *ccsidr = (volatile uint32_t *)0xE000ED80;
    uint32_t cache_info = *ccsidr;

    uint32_t line_size = 4 << ((cache_info & 0x7) + 2);  // Line size in bytes
    uint32_t ways = ((cache_info >> 3) & 0x3FF) + 1;
    uint32_t sets = ((cache_info >> 13) & 0x7FFF) + 1;

    // Calculate bit positions for set/way
    uint32_t way_shift = __builtin_clz(ways - 1);

    // Invalidate all sets and ways
    for (uint32_t set = 0; set < sets; set++) {
        for (uint32_t way = 0; way < ways; way++) {
            uint32_t value = (way << way_shift) | (set << 5);
            SCB_DCISW = value;  // Invalidate by set/way
        }
    }
    __asm__ volatile("dsb");

    // Enable D-cache
    SCB_CCR |= SCB_CCR_DC;
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

    (void)line_size;  // Suppress unused warning
#endif
}

// =============================================================================
// CLOCK CONFIGURATION - STM32H723ZG
// =============================================================================

#if defined(DEVICE_STM32H723ZG)

bool clock_config(void) {
    uint32_t timeout;

#if SKIP_ADVANCED_INIT
    // Safe mode: just run on HSI, no PLL
    return true;
#endif

    // Wait for HSI to be ready (with timeout)
    timeout = INIT_TIMEOUT_MAX;
    while (!(RCC_CR & RCC_CR_HSIRDY)) {
        if (--timeout == 0) return false;  // Timeout - continue anyway
    }

#if USE_PLL
    // Set flash latency before increasing clock
    flash_set_ws(FLASH_WS);

    // Configure voltage scaling
    PWR_D3CR = (PWR_D3CR & ~(PWR_D3CR_VOS_MASK << PWR_D3CR_VOS_SHIFT)) |
               (VOS_SCALE << PWR_D3CR_VOS_SHIFT);

    timeout = CLOCK_TIMEOUT;
    while (!(PWR_D3CR & PWR_D3CR_VOSRDY)) {
        if (--timeout == 0) return false;
    }

    // Disable PLL before configuration
    RCC_CR &= ~RCC_CR_PLL1ON;
    timeout = CLOCK_TIMEOUT;
    while (RCC_CR & RCC_CR_PLL1RDY) {
        if (--timeout == 0) return false;
    }

#if USE_HSE
    // Enable HSE
    RCC_CR |= RCC_CR_HSEON;
    timeout = CLOCK_TIMEOUT;
    while (!(RCC_CR & RCC_CR_HSERDY)) {
        if (--timeout == 0) return false;
    }

    // Configure PLL source (HSE) and divider M
    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~0x3) | RCC_PLLCKSELR_PLLSRC_HSE;
#else
    // Configure PLL source (HSI) and divider M
    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~0x3) | RCC_PLLCKSELR_PLLSRC_HSI;
#endif

    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~(0x3F << RCC_PLLCKSELR_DIVM1_SHIFT)) |
                     RCC_PLLCKSELR_DIVM1(PLL_M);

    // Configure PLL multiplier N and divider P
    RCC_PLL1DIVR = RCC_PLLNDIVR_DIVN(PLL_N) | RCC_PLLNDIVR_DIVP(PLL_P);

    // Configure PLL input range and VCO
    RCC_PLLCFGR = (RCC_PLLCFGR & ~(0x3 << RCC_PLLCFGR_PLL1RGE_SHIFT)) |
                   (PLL_RGE << RCC_PLLCFGR_PLL1RGE_SHIFT);
    RCC_PLLCFGR &= ~RCC_PLLCFGR_PLL1VCO_MED;  // Wide VCO range
    RCC_PLLCFGR |= RCC_PLLCFGR_DIVP1EN;       // Enable P output

    // Enable PLL and wait
    RCC_CR |= RCC_CR_PLL1ON;
    timeout = CLOCK_TIMEOUT;
    while (!(RCC_CR & RCC_CR_PLL1RDY)) {
        if (--timeout == 0) return false;
    }

    // Configure bus dividers
    RCC_D1CFGR = RCC_D1CFGR_D1HPRE(RCC_D1CFGR_D1HPRE_BYP) |
                 RCC_D1CFGR_D1PPRE(RCC_D1CFGR_D1PPRE_DIV2) |
                 RCC_D1CFGR_D1CPRE(RCC_D1CFGR_D1CPRE_BYP);
    RCC_D2CFGR = RCC_D2CFGR_D2PPRE1(RCC_D2CFGR_D2PPRE_DIV2) |
                 RCC_D2CFGR_D2PPRE2(RCC_D2CFGR_D2PPRE_DIV2);
    RCC_D3CFGR = RCC_D3CFGR_D3PPRE(RCC_D3CFGR_D3PPRE_DIV2);

    // Switch to PLL
    RCC_CFGR = (RCC_CFGR & ~(RCC_CFGR_SW_MASK << RCC_CFGR_SW_SHIFT)) |
               (RCC_CFGR_SW_PLL1 << RCC_CFGR_SW_SHIFT);

    timeout = CLOCK_TIMEOUT;
    while (((RCC_CFGR >> RCC_CFGR_SWS_SHIFT) & RCC_CFGR_SWS_MASK) != RCC_CFGR_SWS_PLL1) {
        if (--timeout == 0) return false;
    }
#endif // USE_PLL

    return true;
}

// =============================================================================
// CLOCK CONFIGURATION - STM32U545RE
// =============================================================================

#elif defined(DEVICE_STM32U545RE)

bool clock_config(void) {
    // U5 clock configuration will be implemented when hardware is available
    // For now, run at default HSI16 frequency

    // TODO: Implement PLL configuration for 80/160 MHz modes

    return true;
}

#endif // Device selection

// =============================================================================
// GPIO LED INITIALIZATION
// =============================================================================

void gpio_led_init(void) {
    // Enable clocks for ALL LED GPIOs - always enable explicitly
    // (The preprocessor conditionals don't work with libopencm3 RCC macros)
    rcc_periph_clock_enable(LED1_RCC);  // GPIOB for LED1/LED3
    rcc_periph_clock_enable(LED2_RCC);  // GPIOE for LED2 (yellow)
    rcc_periph_clock_enable(LED3_RCC);  // Already enabled if same as LED1

    // Configure ALL LED pins as outputs
    gpio_mode_setup(LED1_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED1_PIN);
    gpio_clear(LED1_PORT, LED1_PIN);

    gpio_mode_setup(LED2_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED2_PIN);
    gpio_clear(LED2_PORT, LED2_PIN);

    gpio_mode_setup(LED3_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED3_PIN);
    gpio_clear(LED3_PORT, LED3_PIN);
}

// =============================================================================
// SYSTICK INITIALIZATION
// =============================================================================

void systick_init(void) {
    uint32_t reload = (SYSCLK_FREQ_HZ / 1000u) - 1u;
    systick_set_reload(reload);
    systick_set_clocksource(STK_CSR_CLKSOURCE_AHB);
    systick_clear();
    systick_interrupt_enable();
    systick_counter_enable();
}

// =============================================================================
// UART INITIALIZATION (DEBUG)
// =============================================================================

#ifdef DEBUG_UART

void uart_init(void) {
    rcc_periph_clock_enable(DEBUG_USART_PORT_RCC);
    rcc_periph_clock_enable(DEBUG_USART_RCC);

    // Configure TX and RX pins
    gpio_mode_setup(DEBUG_USART_PORT, GPIO_MODE_AF, GPIO_PUPD_NONE,
                    DEBUG_USART_TX_PIN | DEBUG_USART_RX_PIN);
    gpio_set_af(DEBUG_USART_PORT, DEBUG_USART_AF,
                DEBUG_USART_TX_PIN | DEBUG_USART_RX_PIN);

    // Configure USART
    usart_set_baudrate(DEBUG_USART, 115200);
    usart_set_databits(DEBUG_USART, 8);
    usart_set_stopbits(DEBUG_USART, USART_STOPBITS_1);
    usart_set_mode(DEBUG_USART, USART_MODE_TX_RX);
    usart_set_parity(DEBUG_USART, USART_PARITY_NONE);
    usart_set_flow_control(DEBUG_USART, USART_FLOWCONTROL_NONE);
    usart_enable(DEBUG_USART);
}

#else

void uart_init(void) {
    // No-op when debug disabled
}

#endif

// =============================================================================
// MASTER INITIALIZATION
// =============================================================================

void system_init(void) {
    // 1. Enable FPU first
    fpu_enable();

    // 2. Configure MPU
    mpu_config();

    // 3. Setup LEDs early for error indication
    gpio_led_init();

    // 4. Configure clock (with error indication via red LED)
    if (!clock_config()) {
        // Clock config failed - blink red LED and continue at default frequency
        for (int i = 0; i < 10; i++) {
            gpio_toggle(LED3_PORT, LED3_PIN);
            for (volatile int j = 0; j < 500000; j++) __asm__("nop");
        }
    }

    // 5. Enable caches (after clock config)
    cache_enable();

    // 6. Setup SysTick
    systick_init();

    // 7. Setup UART (if debug enabled)
    uart_init();
}
