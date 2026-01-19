# STM32 NUCLEO-U545RE-Q Pin Documentation

This document details the pinout and access methods for the NUCLEO-U545RE-Q board, specifically formatted for the HAL project structure.

## 1. User LED (Green)

*   **Board Name:** LD2 (Green User LED)
*   **Location:** Next to the ST-LINK USB connector. Connected to **Arduino Pin D13**.
*   **STM32 Pin:** `PA5` (Port A, Pin 5)
*   **Function:** Visual feedback (blinking, status indication).
*   **C Code Access (HAL BSP):**
    *   **Header:** `#include "stm32u5xx_nucleo.h"`
    *   **Initialization:** `BSP_LED_Init(LED_GREEN);`
    *   **Toggle:** `BSP_LED_Toggle(LED_GREEN);`
    *   **On/Off:** `BSP_LED_On(LED_GREEN);` / `BSP_LED_Off(LED_GREEN);`

## 2. User Button (Blue)

*   **Board Name:** B1 (User Button)
*   **Location:** Blue button on the edge of the board.
*   **STM32 Pin:** `PC13` (Port C, Pin 13)
*   **Function:** User input (trigger, mode switch).
*   **C Code Access (HAL BSP):**
    *   **Header:** `#include "stm32u5xx_nucleo.h"`
    *   **Initialization:** `BSP_PB_Init(BUTTON_USER, BUTTON_MODE_GPIO);`
    *   **Read State:** `BSP_PB_GetState(BUTTON_USER);` (Returns `1` when pressed)

## 3. VCP UART (Virtual COM Port)

*   **Board Name:** Virtual COM Port (ST-LINK)
*   **Location:** Connected internally to the ST-LINK programmer (USB Micro-B connector).
*   **STM32 Pins:**
    *   **TX:** `PA9` (Arduino D8) *[Note: BSP Default]*
    *   **RX:** `PA10` (Arduino D2) *[Note: BSP Default]*
    *   *Note: Standard Nucleo-64 often uses PA2/PA3, but this BSP is configured for PA9/PA10 on USART1.*
*   **Function:** Serial console output (printf debugging).
*   **C Code Access (HAL BSP):**
    *   **Header:** `#include "stm32u5xx_nucleo.h"`
    *   **Initialization:** `BSP_COM_Init(COM1, ...)`
    *   **Usage:** Standard `printf` (retargeted to COM port).

## 4. Signal Pin (Oscilloscope Timing) - *GCC Project Legacy*

*   **Board Name:** **Arduino D7**
*   **Location:** CN5 connector, Pin 8.
*   **STM32 Pin:** `PA8` (Port A, Pin 8)
*   **Function:** Used in the GCC project to output a precise toggling signal at the start/end of inference for oscilloscope timing measurements.
*   **Signal Behavior:** Push-Pull, High Speed. Alternates 0V/3.3V with every inference flow.
*   **C Code Access (Manual HAL):**
    *   **Initialization:**
        ```c
        /* Enable Clock */
        __HAL_RCC_GPIOA_CLK_ENABLE();
        
        /* Configure GPIO */
        GPIO_InitTypeDef GPIO_InitStruct = {0};
        GPIO_InitStruct.Pin = GPIO_PIN_8;
        GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
        ```
    *   **Toggle:** `HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_8);`
    *   **Set High:** `HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);`
    *   **Set Low:** `HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);`
