/**
  ******************************************************************************
  * @file    ai_led.c
  * @brief   LED feedback implementation for AI inference
  * @details Uses BSP_LED functions. U545 has only LED_GREEN available.
  ******************************************************************************
  */

#include "ai_led.h"

/**
  * @brief  Toggle the user LED
  */
void ai_led_toggle(void)
{
  BSP_LED_Toggle(LED_GREEN);
}

/**
  * @brief  Turn LED on
  */
void ai_led_on(void)
{
  BSP_LED_On(LED_GREEN);
}

/**
  * @brief  Turn LED off
  */
void ai_led_off(void)
{
  BSP_LED_Off(LED_GREEN);
}

/**
  * @brief  Error blink pattern
  * @param  count Number of blinks indicating error code
  */
void ai_led_error_blink(int count)
{
  /* Short blinks = error code number */
  for (int i = 0; i < count; i++)
  {
    BSP_LED_On(LED_GREEN);
    HAL_Delay(100);  /* Short ON */
    BSP_LED_Off(LED_GREEN);
    HAL_Delay(100);  /* Short OFF */
  }
  /* Long ON to mark end of error code */
  BSP_LED_On(LED_GREEN);
  HAL_Delay(500);
  BSP_LED_Off(LED_GREEN);
  HAL_Delay(500);
}

/**
  * @brief  Startup blink to show system is alive
  */
void ai_led_startup_blink(void)
{
  /* Single long blink */
  BSP_LED_On(LED_GREEN);
  HAL_Delay(500);
  BSP_LED_Off(LED_GREEN);
  HAL_Delay(200);
}
