// =============================================================================
// STM32H723ZG AI Inference Firmware
// =============================================================================
// Features:
// - Configurable CPU frequency via TARGET_SYSCLK_MHZ
// - Debug mode with UART logging (DEBUG build)
// - Release mode optimized for performance (default)
// - Signal output on GPIO (PE2-PE9: data, PE10: strobe)
// =============================================================================

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/stm32/flash.h>
#include <libopencm3/stm32/pwr.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/cm3/scb.h>
#include <libopencm3/cm3/mpu.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>

#include "network.h"
#include "network_data.h"

// =============================================================================
// CONFIGURATION
// =============================================================================

// Target CPU frequency in MHz (choose one: 64, 120, 240, 480)
// Note: Higher frequencies require proper voltage scaling and flash latency
#ifndef TARGET_SYSCLK_MHZ
#define TARGET_SYSCLK_MHZ  64  // Default: Use HSI without PLL (safest)
#endif

// Enable UART debug output (defined via Makefile DEBUG=1 flag)
// #define DEBUG_UART  // Uncomment to force enable, or use DEBUG build

#ifdef DEBUG_SEMIHOSTING
#define DEBUG_UART
#endif

// =============================================================================
// CLOCK CONFIGURATION - AUTO-CALCULATED FROM TARGET_SYSCLK_MHZ
// =============================================================================
// HSI frequency is fixed at 64 MHz
// PLL formula: SYSCLK = (HSI_FREQ / PLLM) * PLLN / PLLP
//
// Constraints:
// - PLL input (after /M): 2-16 MHz (ideally 4-8 MHz for stability)
// - VCO output: 192-836 MHz
// - SYSCLK max: 550 MHz (VOS0), 480 MHz (VOS1), 300 MHz (VOS2)
// =============================================================================

#define HSI_FREQ_MHZ       64
#define HSI_FREQ_HZ        (HSI_FREQ_MHZ * 1000000UL)
#define SYSCLK_FREQ_HZ     (TARGET_SYSCLK_MHZ * 1000000UL)

#if TARGET_SYSCLK_MHZ == 64
    // No PLL needed - run directly from HSI
    #define USE_PLL         0
    #define FLASH_LATENCY   FLASH_ACR_LATENCY_1WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_3
#elif TARGET_SYSCLK_MHZ == 120
    #define USE_PLL         1
    #define PLL_M           8   // 64/8 = 8 MHz
    #define PLL_N           60  // 8*60 = 480 MHz VCO
    #define PLL_P           4   // 480/4 = 120 MHz
    #define PLL_INPUT_RANGE RCC_PLLCFGR_PLLRGE_4_8MHZ
    #define FLASH_LATENCY   FLASH_ACR_LATENCY_1WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_3
#elif TARGET_SYSCLK_MHZ == 240
    #define USE_PLL         1
    #define PLL_M           8   // 64/8 = 8 MHz
    #define PLL_N           60  // 8*60 = 480 MHz VCO
    #define PLL_P           2   // 480/2 = 240 MHz
    #define PLL_INPUT_RANGE RCC_PLLCFGR_PLLRGE_4_8MHZ
    #define FLASH_LATENCY   FLASH_ACR_LATENCY_2WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_2
#elif TARGET_SYSCLK_MHZ == 480
    #define USE_PLL         1
    #define PLL_M           4   // 64/4 = 16 MHz
    #define PLL_N           60  // 16*60 = 960 MHz VCO
    #define PLL_P           2   // 960/2 = 480 MHz
    #define PLL_INPUT_RANGE RCC_PLLCFGR_PLLRGE_8_16MHZ
    #define FLASH_LATENCY   FLASH_ACR_LATENCY_4WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_1
#else
    #error "Unsupported TARGET_SYSCLK_MHZ. Choose 64, 120, 240, or 480."
#endif

// APB clocks = SYSCLK / 2 (max ~275 MHz for APB)
#define APB_FREQ_HZ        (SYSCLK_FREQ_HZ / 2)

// =============================================================================
// HARDWARE DEFINITIONS
// =============================================================================

// LED pins (NUCLEO-H723ZG)
#define LED1_PORT   GPIOB   // Green LED
#define LED1_PIN    GPIO0
#define LED2_PORT   GPIOE   // Yellow LED
#define LED2_PIN    GPIO1
#define LED3_PORT   GPIOB   // Red LED
#define LED3_PIN    GPIO14

// Signal output pins
#define SIGNAL_DATA_PORT    GPIOE
#define SIGNAL_DATA_PINS    (GPIO2 | GPIO3 | GPIO4 | GPIO5 | GPIO6 | GPIO7 | GPIO8 | GPIO9)
#define SIGNAL_STROBE_PIN   GPIO10

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

static volatile uint32_t system_millis = 0;

// ST.AI Network buffers
STAI_ALIGNED(STAI_NETWORK_CONTEXT_ALIGNMENT)
static stai_network network[STAI_NETWORK_CONTEXT_SIZE];

STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations[STAI_NETWORK_ACTIVATIONS_SIZE];

static float input_data[STAI_NETWORK_IN_1_SIZE];
static float output_data[STAI_NETWORK_OUT_1_SIZE];

// Input data generator macro
#define MODEL_INPUT_GEN(i) ((float)((system_millis * 13 + counter * 7 + i * 3) % 100) / 100.0f)

// =============================================================================
// DEBUG OUTPUT (conditional compilation)
// =============================================================================

#ifdef DEBUG_UART

static void usart_setup(void) {
    rcc_periph_clock_enable(RCC_GPIOD);
    rcc_periph_clock_enable(RCC_USART3);

    // USART3: PD8 (TX), PD9 (RX) - VCP on Nucleo board
    gpio_mode_setup(GPIOD, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO8 | GPIO9);
    gpio_set_af(GPIOD, GPIO_AF7, GPIO8 | GPIO9);

    usart_set_baudrate(USART3, 115200);
    usart_set_databits(USART3, 8);
    usart_set_stopbits(USART3, USART_STOPBITS_1);
    usart_set_mode(USART3, USART_MODE_TX_RX);
    usart_set_parity(USART3, USART_PARITY_NONE);
    usart_set_flow_control(USART3, USART_FLOWCONTROL_NONE);
    usart_enable(USART3);
}

static void debug_printf(const char *format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    for (int i = 0; buffer[i] != '\0'; i++) {
        usart_send_blocking(USART3, buffer[i]);
    }
}

#else

// Release mode: UART disabled, debug_printf does nothing
#define usart_setup()       ((void)0)
#define debug_printf(...)   ((void)0)

#endif

// =============================================================================
// SYSTEM FUNCTIONS
// =============================================================================

void sys_tick_handler(void) {
    system_millis++;
}

static void delay_ms(uint32_t ms) {
    uint32_t start = system_millis;
    while ((system_millis - start) < ms) {
        __asm__("wfi");
    }
}

static void mpu_config(void) {
    // Disable MPU to avoid memory access conflicts
    MPU_CTRL = 0;
}

static void systick_setup(void) {
    uint32_t reload = (SYSCLK_FREQ_HZ / 1000u) - 1u;
    systick_set_reload(reload);
    systick_set_clocksource(STK_CSR_CLKSOURCE_AHB);
    systick_clear();
    systick_interrupt_enable();
    systick_counter_enable();
}

// =============================================================================
// CLOCK CONFIGURATION
// =============================================================================

// Timeout value: ~1 second at 64MHz (rough estimate, actual timing depends on loop overhead)
#define CLOCK_TIMEOUT  10000000UL

static bool system_clock_config(void) {
    uint32_t timeout;

    // Wait for HSI to be ready (with timeout)
    timeout = CLOCK_TIMEOUT;
    while (!(RCC_CR & RCC_CR_HSIRDY)) {
        if (--timeout == 0) return false;
    }

#if USE_PLL
    // Set flash latency before increasing clock
    flash_set_ws(FLASH_LATENCY);

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

    // Configure PLL source (HSI) and divider M
    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~0x3) | RCC_PLLCKSELR_PLLSRC_HSI;
    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~(0x3F << RCC_PLLCKSELR_DIVM1_SHIFT)) |
                     RCC_PLLCKSELR_DIVM1(PLL_M);

    // Configure PLL multiplier N and divider P
    RCC_PLL1DIVR = RCC_PLLNDIVR_DIVN(PLL_N) | RCC_PLLNDIVR_DIVP(PLL_P);

    // Configure PLL input range and VCO
    RCC_PLLCFGR = (RCC_PLLCFGR & ~(0x3 << RCC_PLLCFGR_PLL1RGE_SHIFT)) |
                   (PLL_INPUT_RANGE << RCC_PLLCFGR_PLL1RGE_SHIFT);
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

    // Enable caches
    SCB_ICIALLU = 0;
    SCB_CCR |= SCB_CCR_IC;
    SCB_DCISW = 0;
    SCB_CCR |= SCB_CCR_DC;
#endif
    // If USE_PLL == 0, we stay on HSI at 64 MHz (default after reset)
    return true;
}

// =============================================================================
// GPIO SETUP
// =============================================================================

static void gpio_setup(void) {
    rcc_periph_clock_enable(RCC_GPIOB);
    rcc_periph_clock_enable(RCC_GPIOE);

    // LEDs
    gpio_mode_setup(LED1_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED1_PIN);
    gpio_mode_setup(LED2_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED2_PIN);
    gpio_mode_setup(LED3_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED3_PIN);
    gpio_clear(LED1_PORT, LED1_PIN);
    gpio_clear(LED2_PORT, LED2_PIN);
    gpio_clear(LED3_PORT, LED3_PIN);
}

static void signal_gpio_setup(void) {
    rcc_periph_clock_enable(RCC_GPIOE);
    // Data pins PE2-PE9 and strobe PE10
    gpio_mode_setup(SIGNAL_DATA_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE,
                    SIGNAL_DATA_PINS | SIGNAL_STROBE_PIN);
    gpio_clear(SIGNAL_DATA_PORT, SIGNAL_DATA_PINS | SIGNAL_STROBE_PIN);
}

static void emit_signal(uint8_t value) {
    // Set data on PE2-PE9 (shift by 2 for pin alignment)
    uint16_t mask = 0xFF << 2;
    gpio_clear(SIGNAL_DATA_PORT, mask);
    gpio_set(SIGNAL_DATA_PORT, ((uint16_t)value << 2) & mask);

    // Pulse strobe
    gpio_set(SIGNAL_DATA_PORT, SIGNAL_STROBE_PIN);
    for (volatile int i = 0; i < 100; i++) __asm__("nop");
    gpio_clear(SIGNAL_DATA_PORT, SIGNAL_STROBE_PIN);
}

// =============================================================================
// ERROR HANDLER
// =============================================================================

static void error_handler(const char* msg, stai_return_code code) {
    (void)msg;  // Used only in debug builds
    (void)code; // Used only in debug builds
    debug_printf("ERROR: %s (0x%X)\n", msg, code);
    for (;;) {
        gpio_toggle(LED3_PORT, LED3_PIN);
        delay_ms(100);
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(void) {
    // Enable FPU
    SCB_CPACR |= (0xF << 20);
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

    // System initialization
    mpu_config();
    gpio_setup();  // Setup LEDs first for error indication

    // Configure clock (with timeout protection)
    if (!system_clock_config()) {
        // Clock config failed - blink red LED rapidly and continue at 64MHz HSI
        for (int i = 0; i < 10; i++) {
            gpio_toggle(LED3_PORT, LED3_PIN);
            for (volatile int j = 0; j < 500000; j++) __asm__("nop");
        }
        // Continue anyway at default 64MHz HSI
    }

    signal_gpio_setup();
    usart_setup();
    systick_setup();

    // Startup message
    debug_printf("\n\n====================================\n");
    debug_printf("STM32 AI Inference - %d MHz\n", TARGET_SYSCLK_MHZ);
    debug_printf("====================================\n");

    // Initialize AI network
    debug_printf("Initializing network...\n");

    stai_return_code err = stai_runtime_init();
    if (err != STAI_SUCCESS) error_handler("Runtime Init", err);

    err = stai_network_init(network);
    if (err != STAI_SUCCESS) error_handler("Network Init", err);

    const stai_ptr act_ptrs[] = {(stai_ptr)activations};
    err = stai_network_set_activations(network, act_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Activations", err);

    const stai_ptr wgt_ptrs[] = {(stai_ptr)g_network_weights_array};
    err = stai_network_set_weights(network, wgt_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Weights", err);

    const stai_ptr in_ptrs[] = {(stai_ptr)input_data};
    err = stai_network_set_inputs(network, in_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Inputs", err);

    const stai_ptr out_ptrs[] = {(stai_ptr)output_data};
    err = stai_network_set_outputs(network, out_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Outputs", err);

    debug_printf("Network ready: %s\n", STAI_NETWORK_MODEL_NAME);

    // Main inference loop
    int counter = 0;
    for (;;) {
        // Clear LEDs
        gpio_clear(LED1_PORT, LED1_PIN);
        gpio_clear(LED2_PORT, LED2_PIN);
        gpio_clear(LED3_PORT, LED3_PIN);

        // Prepare input
        gpio_set(LED2_PORT, LED2_PIN);
        for (int i = 0; i < STAI_NETWORK_IN_1_SIZE; i++) {
            input_data[i] = MODEL_INPUT_GEN(i);
        }

        // Run inference
        gpio_clear(LED2_PORT, LED2_PIN);
        gpio_set(LED3_PORT, LED3_PIN);

        uint32_t t_start = system_millis;
        err = stai_network_run(network, STAI_MODE_SYNC);
        uint32_t t_elapsed = system_millis - t_start;
        (void)t_elapsed; // Used only in debug builds

        if (err != STAI_SUCCESS) {
            debug_printf("Inference FAILED (0x%X)\n", err);
            for (int i = 0; i < 6; i++) {
                gpio_toggle(LED3_PORT, LED3_PIN);
                delay_ms(50);
            }
        } else {
            // Output result
            #ifdef QUANTIZED_INT8
            uint8_t out_byte = (uint8_t)output_data[0];
            debug_printf("OK %lums | Out: %d | INT8\n", t_elapsed, (int8_t)output_data[0]);
            #else
            uint8_t out_byte = (uint8_t)(output_data[0] * 100);
            debug_printf("OK %lums | Out: %.4f | FP32\n", t_elapsed, (double)output_data[0]);
            #endif

            emit_signal(out_byte);

            gpio_clear(LED3_PORT, LED3_PIN);
            gpio_set(LED1_PORT, LED1_PIN);
            delay_ms(100);
        }

        #ifndef NO_SLEEP
        delay_ms(200);
        #endif

        counter++;
        if (system_millis > 10000 || counter > 10000) {
            system_millis = 0;
            counter = 0;
            debug_printf("--- COUNTER RESET ---\n");
        }

        gpio_clear(LED1_PORT, LED1_PIN);

        #ifndef NO_SLEEP
        delay_ms(400);
        #endif
    }
}
