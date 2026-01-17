/**
 * @file init_config.h
 * @brief System initialization and configuration functions
 *
 * Provides device-agnostic initialization for clock, cache, MPU, GPIO, UART.
 * Uses device_config.h for device-specific parameters.
 */

#ifndef INIT_CONFIG_H
#define INIT_CONFIG_H

#include "device_config.h"
#include <stdbool.h>
#include <stdint.h>

// =============================================================================
// CLOCK CONFIGURATION MACROS
// =============================================================================

// Target CPU frequency in MHz (set via Makefile SYSCLK=xxx)
#ifndef TARGET_SYSCLK_MHZ
    #if defined(DEVICE_STM32H723ZG)
        #define TARGET_SYSCLK_MHZ  64   // Default: Safe HSI mode
    #elif defined(DEVICE_STM32U545RE)
        #define TARGET_SYSCLK_MHZ  16   // Default: HSI16 mode
    #endif
#endif

// Derived frequency values
#define SYSCLK_FREQ_HZ     (TARGET_SYSCLK_MHZ * 1000000UL)
#define APB_FREQ_HZ        (SYSCLK_FREQ_HZ / 2)

// =============================================================================
// CLOCK CONFIGURATION FOR STM32H723ZG
// =============================================================================
#if defined(DEVICE_STM32H723ZG)

#if TARGET_SYSCLK_MHZ == 64
    // No PLL needed - run directly from HSI
    #define USE_PLL         0
    #define USE_HSE         0
    #define FLASH_WS        FLASH_ACR_LATENCY_1WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_3
#elif TARGET_SYSCLK_MHZ == 120
    #define USE_PLL         1
    #define USE_HSE         0
    #define PLL_M           8   // 64/8 = 8 MHz
    #define PLL_N           60  // 8*60 = 480 MHz VCO
    #define PLL_P           4   // 480/4 = 120 MHz
    #define PLL_RGE         RCC_PLLCFGR_PLLRGE_4_8MHZ
    #define FLASH_WS        FLASH_ACR_LATENCY_1WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_3
#elif TARGET_SYSCLK_MHZ == 240
    #define USE_PLL         1
    #define USE_HSE         0
    #define PLL_M           8   // 64/8 = 8 MHz
    #define PLL_N           60  // 8*60 = 480 MHz VCO
    #define PLL_P           2   // 480/2 = 240 MHz
    #define PLL_RGE         RCC_PLLCFGR_PLLRGE_4_8MHZ
    #define FLASH_WS        FLASH_ACR_LATENCY_2WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_2
#elif TARGET_SYSCLK_MHZ == 480
    #define USE_PLL         1
    #define USE_HSE         0
    #define PLL_M           4   // 64/4 = 16 MHz
    #define PLL_N           60  // 16*60 = 960 MHz VCO
    #define PLL_P           2   // 960/2 = 480 MHz
    #define PLL_RGE         RCC_PLLCFGR_PLLRGE_8_16MHZ
    #define FLASH_WS        FLASH_ACR_LATENCY_4WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_1
#elif TARGET_SYSCLK_MHZ == 550
    // Maximum performance - requires HSE (8 MHz on Nucleo)
    #define USE_PLL         1
    #define USE_HSE         1
    #define PLL_M           4   // 8/4 = 2 MHz (HSE)
    #define PLL_N           275 // 2*275 = 550 MHz VCO
    #define PLL_P           1   // 550/1 = 550 MHz
    #define PLL_RGE         RCC_PLLCFGR_PLLRGE_1_2MHZ
    #define FLASH_WS        FLASH_ACR_LATENCY_3WS
    #define VOS_SCALE       PWR_D3CR_VOS_SCALE_0  // VOS0 for >480MHz
#else
    #error "Unsupported TARGET_SYSCLK_MHZ for H723ZG. Choose 64, 120, 240, 480, or 550."
#endif

// =============================================================================
// CLOCK CONFIGURATION FOR STM32U545RE
// =============================================================================
#elif defined(DEVICE_STM32U545RE)

#if TARGET_SYSCLK_MHZ == 16
    // No PLL - run directly from HSI16
    #define USE_PLL         0
    #define FLASH_WS        0   // 0 wait states at 16MHz
    #define VOS_RANGE       3   // Range 3 sufficient
#elif TARGET_SYSCLK_MHZ == 80
    #define USE_PLL         1
    #define PLL_M           2   // 16/2 = 8 MHz
    #define PLL_N           20  // 8*20 = 160 MHz VCO
    #define PLL_P           2   // 160/2 = 80 MHz
    #define FLASH_WS        3   // 3 wait states at 80MHz
    #define VOS_RANGE       2   // Range 2
#elif TARGET_SYSCLK_MHZ == 160
    #define USE_PLL         1
    #define PLL_M           2   // 16/2 = 8 MHz
    #define PLL_N           40  // 8*40 = 320 MHz VCO
    #define PLL_P           2   // 320/2 = 160 MHz
    #define FLASH_WS        4   // 4 wait states at 160MHz
    #define VOS_RANGE       1   // Range 1 for max performance
#else
    #error "Unsupported TARGET_SYSCLK_MHZ for U545RE. Choose 16, 80, or 160."
#endif

#endif // Device selection

// =============================================================================
// PUBLIC FUNCTION DECLARATIONS
// =============================================================================

/**
 * @brief Master system initialization
 *
 * Calls all initialization functions in correct order:
 * 1. FPU enable
 * 2. MPU configuration
 * 3. GPIO setup (LEDs)
 * 4. Clock configuration
 * 5. Cache enable
 * 6. SysTick setup
 * 7. UART setup (if debug enabled)
 */
void system_init(void);

/**
 * @brief Configure system clock to TARGET_SYSCLK_MHZ
 * @return true on success, false on timeout/failure
 */
bool clock_config(void);

/**
 * @brief Enable I-cache and D-cache (if available)
 *
 * Properly invalidates caches before enabling to ensure coherency.
 */
void cache_enable(void);

/**
 * @brief Configure MPU for optimal performance
 *
 * Enables MPU with privileged default mode.
 */
void mpu_config(void);

/**
 * @brief Enable FPU coprocessor
 */
void fpu_enable(void);

/**
 * @brief Configure SysTick for 1ms interrupts
 */
void systick_init(void);

/**
 * @brief Configure GPIO for LEDs
 */
void gpio_led_init(void);

/**
 * @brief Configure UART for debug output
 */
void uart_init(void);

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

// Updated by SysTick interrupt handler
extern volatile uint32_t system_millis;

#endif // INIT_CONFIG_H
