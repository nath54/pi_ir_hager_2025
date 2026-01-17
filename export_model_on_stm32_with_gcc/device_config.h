/**
 * @file device_config.h
 * @brief Device-specific configuration for STM32 boards
 *
 * This header provides abstracted definitions for different STM32 development
 * boards. Select the target device via Makefile DEVICE= flag.
 *
 * Supported devices:
 * - DEVICE_STM32H723ZG (NUCLEO-H723ZG) - Cortex-M7 @ 550MHz
 * - DEVICE_STM32U545RE (NUCLEO-U545RE-Q) - Cortex-M33 @ 160MHz
 */

#ifndef DEVICE_CONFIG_H
#define DEVICE_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

// =============================================================================
// DEVICE DETECTION AND DEFAULTS
// =============================================================================

#if !defined(DEVICE_STM32H723ZG) && !defined(DEVICE_STM32U545RE)
    // Default to H723ZG if no device specified
    #define DEVICE_STM32H723ZG
#endif

// =============================================================================
// STM32H723ZG (NUCLEO-H723ZG) CONFIGURATION
// Cortex-M7 with FPU, D-cache, I-cache, 550MHz max
// =============================================================================
#if defined(DEVICE_STM32H723ZG)

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/stm32/flash.h>
#include <libopencm3/stm32/pwr.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/cm3/scb.h>
#include <libopencm3/cm3/mpu.h>

#define DEVICE_NAME             "NUCLEO-H723ZG"
#define DEVICE_CORE             "Cortex-M7"
#define DEVICE_MAX_SYSCLK_MHZ   550
#define DEVICE_HAS_DCACHE       1
#define DEVICE_HAS_ICACHE       1
#define DEVICE_HAS_FPU          1
#define DEVICE_HAS_HSE          1
#define DEVICE_HSE_MHZ          8

// LED definitions (active high on Nucleo-144)
// LED1 = Green (PB0), LED2 = Yellow (PE1), LED3 = Red (PB14)
#define LED1_PORT               GPIOB
#define LED1_PIN                GPIO0
#define LED1_RCC                RCC_GPIOB

#define LED2_PORT               GPIOE
#define LED2_PIN                GPIO1
#define LED2_RCC                RCC_GPIOE

#define LED3_PORT               GPIOB
#define LED3_PIN                GPIO14
#define LED3_RCC                RCC_GPIOB

// Signal output for oscilloscope (PE10 - strong push-pull)
#define SIGNAL_PORT             GPIOE
#define SIGNAL_PIN              GPIO10
#define SIGNAL_RCC              RCC_GPIOE

// Optional data bus for parallel output (PE2-PE9)
#define SIGNAL_DATA_PORT        GPIOE
#define SIGNAL_DATA_PINS        (GPIO2 | GPIO3 | GPIO4 | GPIO5 | GPIO6 | GPIO7 | GPIO8 | GPIO9)
#define SIGNAL_DATA_RCC         RCC_GPIOE

// USART3 for debug (VCP on Nucleo) - PD8=TX, PD9=RX
#define DEBUG_USART             USART3
#define DEBUG_USART_RCC         RCC_USART3
#define DEBUG_USART_PORT        GPIOD
#define DEBUG_USART_PORT_RCC    RCC_GPIOD
#define DEBUG_USART_TX_PIN      GPIO8
#define DEBUG_USART_RX_PIN      GPIO9
#define DEBUG_USART_AF          GPIO_AF7

// Flash/RAM for reference
#define DEVICE_FLASH_SIZE_KB    1024
#define DEVICE_RAM_SIZE_KB      564    // Total SRAM

// Clock configuration constants
#define HSI_FREQ_MHZ            64
#define HSI_FREQ_HZ             (HSI_FREQ_MHZ * 1000000UL)

// =============================================================================
// STM32U545RE (NUCLEO-U545RE-Q) CONFIGURATION
// Cortex-M33 with TrustZone, I-cache only, 160MHz max
// =============================================================================
#elif defined(DEVICE_STM32U545RE)

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/stm32/flash.h>
#include <libopencm3/stm32/pwr.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/cm3/scb.h>

#define DEVICE_NAME             "NUCLEO-U545RE-Q"
#define DEVICE_CORE             "Cortex-M33"
#define DEVICE_MAX_SYSCLK_MHZ   160
#define DEVICE_HAS_DCACHE       0    // No D-cache on Cortex-M33
#define DEVICE_HAS_ICACHE       1
#define DEVICE_HAS_FPU          1
#define DEVICE_HAS_HSE          0    // No HSE on this Nucleo (uses MSI/HSI)
#define DEVICE_HSI_MHZ          16

// LED definition (NUCLEO-64 has only one user LED on PA5)
// Note: PA5 is shared with Arduino D13 / SPI SCK
#define LED1_PORT               GPIOA
#define LED1_PIN                GPIO5
#define LED1_RCC                RCC_GPIOA

// LED2 and LED3 not available - map to same as LED1 for compatibility
#define LED2_PORT               LED1_PORT
#define LED2_PIN                LED1_PIN
#define LED2_RCC                LED1_RCC

#define LED3_PORT               LED1_PORT
#define LED3_PIN                LED1_PIN
#define LED3_RCC                LED1_RCC

// Signal output for oscilloscope (PA8 - available on Arduino D7)
#define SIGNAL_PORT             GPIOA
#define SIGNAL_PIN              GPIO8
#define SIGNAL_RCC              RCC_GPIOA

// Data bus not available on Nucleo-64
#define SIGNAL_DATA_PORT        GPIOA
#define SIGNAL_DATA_PINS        0
#define SIGNAL_DATA_RCC         RCC_GPIOA

// USART2 for debug (VCP on Nucleo-64) - PA2=TX, PA3=RX (directly on ST-LINK)
// Note: USART1 is available on PA9/PA10 for Arduino compatibility
#define DEBUG_USART             USART2
#define DEBUG_USART_RCC         RCC_USART2
#define DEBUG_USART_PORT        GPIOA
#define DEBUG_USART_PORT_RCC    RCC_GPIOA
#define DEBUG_USART_TX_PIN      GPIO2
#define DEBUG_USART_RX_PIN      GPIO3
#define DEBUG_USART_AF          GPIO_AF7

// Flash/RAM for reference
#define DEVICE_FLASH_SIZE_KB    512
#define DEVICE_RAM_SIZE_KB      272

// Clock configuration constants
#define HSI_FREQ_MHZ            16    // HSI16 on U5 series
#define HSI_FREQ_HZ             (HSI_FREQ_MHZ * 1000000UL)

#endif // Device selection

// =============================================================================
// COMMON DEFINITIONS
// =============================================================================

// LED indices for abstracted access
typedef enum {
    LED_GREEN = 0,
    LED_YELLOW = 1,
    LED_RED = 2,
    LED_COUNT
} led_t;

// Library selection (set via Makefile)
#if defined(USE_HAL_LIBRARY)
    #define LIB_NAME "ST HAL"
#else
    #define LIB_NAME "libopencm3"
#endif

#endif // DEVICE_CONFIG_H
