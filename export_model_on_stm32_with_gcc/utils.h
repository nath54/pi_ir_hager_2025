/**
 * @file utils.h
 * @brief Utility functions for LED control, signal output, and delays
 *
 * Provides device-abstracted utility functions.
 */

#ifndef UTILS_H
#define UTILS_H

#include "device_config.h"
#include "init_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>

// =============================================================================
// LED CONTROL FUNCTIONS
// =============================================================================

/**
 * @brief Turn on specified LED
 * @param led LED index (LED_GREEN, LED_YELLOW, LED_RED)
 */
void led_set(led_t led);

/**
 * @brief Turn off specified LED
 * @param led LED index
 */
void led_clear(led_t led);

/**
 * @brief Toggle specified LED
 * @param led LED index
 */
void led_toggle(led_t led);

/**
 * @brief Turn off all LEDs
 */
void led_clear_all(void);

// =============================================================================
// SIGNAL OUTPUT FUNCTIONS (for oscilloscope)
// =============================================================================

/**
 * @brief Initialize signal output GPIO
 *
 * Configures signal pin as push-pull output with maximum drive strength.
 */
void signal_gpio_init(void);

/**
 * @brief Signal start of inference (for oscilloscope timing)
 *
 * Uses alternating pattern: HIGH on odd inferences, LOW on even.
 * This creates clear transitions visible on oscilloscope.
 */
void signal_inference_start(void);

/**
 * @brief Signal end of inference
 *
 * Toggle signal state to mark end of inference period.
 */
void signal_inference_end(void);

/**
 * @brief Emit a data byte on parallel signal pins
 * @param value 8-bit value to output (PE2-PE9 on H723ZG)
 */
void signal_emit_data(uint8_t value);

// =============================================================================
// DELAY FUNCTIONS
// =============================================================================

/**
 * @brief Delay for specified milliseconds
 * @param ms Number of milliseconds to delay
 */
void delay_ms(uint32_t ms);

/**
 * @brief Get current system time in milliseconds
 * @return System uptime in ms
 */
uint32_t millis(void);

// =============================================================================
// DEBUG OUTPUT
// =============================================================================

#ifdef DEBUG_UART

/**
 * @brief Printf-style debug output (UART or semihosting)
 * @param format Printf format string
 * @param ... Variable arguments
 */
void debug_printf(const char *format, ...);

#else

// No-op macro when debug disabled
#define debug_printf(...) ((void)0)

#endif

// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * @brief Enter error state with LED indication
 * @param msg Error message (for debug output)
 * @param code Error code
 *
 * Blinks red LED and halts in infinite loop.
 */
void error_handler(const char* msg, uint32_t code);

#endif // UTILS_H
