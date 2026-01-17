/**
 * @file utils.c
 * @brief Utility function implementations
 */

#include "utils.h"
#include "device_config.h"
#include "init_config.h"
#include <stdio.h>
#include <string.h>

// =============================================================================
// LED CONTROL
// =============================================================================

// LED port/pin lookup tables for abstraction
static const uint32_t led_ports[] = {LED1_PORT, LED2_PORT, LED3_PORT};
static const uint16_t led_pins[]  = {LED1_PIN, LED2_PIN, LED3_PIN};

void led_set(led_t led) {
    if (led < LED_COUNT) {
        gpio_set(led_ports[led], led_pins[led]);
    }
}

void led_clear(led_t led) {
    if (led < LED_COUNT) {
        gpio_clear(led_ports[led], led_pins[led]);
    }
}

void led_toggle(led_t led) {
    if (led < LED_COUNT) {
        gpio_toggle(led_ports[led], led_pins[led]);
    }
}

void led_clear_all(void) {
    gpio_clear(LED1_PORT, LED1_PIN);
#if LED2_PORT != LED1_PORT || LED2_PIN != LED1_PIN
    gpio_clear(LED2_PORT, LED2_PIN);
#endif
#if (LED3_PORT != LED1_PORT || LED3_PIN != LED1_PIN) && \
    (LED3_PORT != LED2_PORT || LED3_PIN != LED2_PIN)
    gpio_clear(LED3_PORT, LED3_PIN);
#endif
}

// =============================================================================
// SIGNAL OUTPUT (OSCILLOSCOPE)
// =============================================================================

// Current signal state for alternating pattern
static bool signal_state = false;

void signal_gpio_init(void) {
    rcc_periph_clock_enable(SIGNAL_RCC);

    // Configure main signal pin as push-pull with maximum drive strength
    gpio_mode_setup(SIGNAL_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, SIGNAL_PIN);
    gpio_set_output_options(SIGNAL_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_100MHZ, SIGNAL_PIN);
    gpio_clear(SIGNAL_PORT, SIGNAL_PIN);

#if SIGNAL_DATA_PINS != 0
    // Configure data bus pins if available (H723ZG only)
    rcc_periph_clock_enable(SIGNAL_DATA_RCC);
    gpio_mode_setup(SIGNAL_DATA_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, SIGNAL_DATA_PINS);
    gpio_set_output_options(SIGNAL_DATA_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_100MHZ, SIGNAL_DATA_PINS);
    gpio_clear(SIGNAL_DATA_PORT, SIGNAL_DATA_PINS);
#endif
}

void signal_inference_start(void) {
    // Toggle state for alternating pattern
    // Creates: HIGH->inference->LOW->inference->HIGH->...
    // This makes it very easy to see inference boundaries on oscilloscope
    signal_state = !signal_state;

    if (signal_state) {
        gpio_set(SIGNAL_PORT, SIGNAL_PIN);
    } else {
        gpio_clear(SIGNAL_PORT, SIGNAL_PIN);
    }
}

void signal_inference_end(void) {
    // For the alternating pattern, we keep the signal as-is until next inference
    // The signal change happens ONLY at start of inference, creating clear edges
    //
    // Timeline example:
    //   |--LOW--|  inference_start (becomes HIGH) |--HIGH--|
    //   |--HIGH--|  inference_start (becomes LOW) |--LOW--|
    //
    // This creates clean square waves where:
    // - The HIGH/LOW period = inference time
    // - Each edge = start of new inference

    // No action needed - signal stays at current level
}

void signal_emit_data(uint8_t value) {
#if SIGNAL_DATA_PINS != 0
    // Set data on PE2-PE9 (shift by 2 for pin alignment)
    uint16_t mask = 0xFF << 2;
    gpio_clear(SIGNAL_DATA_PORT, mask);
    gpio_set(SIGNAL_DATA_PORT, ((uint16_t)value << 2) & mask);

    // Brief strobe pulse on main signal pin
    gpio_set(SIGNAL_PORT, SIGNAL_PIN);
    for (volatile int i = 0; i < 100; i++) __asm__("nop");
    gpio_clear(SIGNAL_PORT, SIGNAL_PIN);
#else
    // No data bus available - just indicate output occurred
    (void)value;
    gpio_set(SIGNAL_PORT, SIGNAL_PIN);
    for (volatile int i = 0; i < 100; i++) __asm__("nop");
    gpio_clear(SIGNAL_PORT, SIGNAL_PIN);
#endif
}

// =============================================================================
// DELAY FUNCTIONS
// =============================================================================

void delay_ms(uint32_t ms) {
    uint32_t start = system_millis;
    while ((system_millis - start) < ms) {
        __asm__("wfi");  // Wait for interrupt (power saving)
    }
}

uint32_t millis(void) {
    return system_millis;
}

// =============================================================================
// DEBUG OUTPUT
// =============================================================================

#ifdef DEBUG_UART

void debug_printf(const char *format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    for (int i = 0; buffer[i] != '\0'; i++) {
        usart_send_blocking(DEBUG_USART, buffer[i]);
    }
}

#endif

// =============================================================================
// ERROR HANDLING
// =============================================================================

void error_handler(const char* msg, uint32_t code) {
    // Debug output if available
#ifdef DEBUG_UART
    debug_printf("ERROR: %s (0x%lX)\n", msg, code);
#else
    (void)msg;
    (void)code;
#endif

    // Blink red LED in infinite loop
    for (;;) {
        led_toggle(LED_RED);
        delay_ms(100);
    }
}
