/**
  ******************************************************************************
  * @file    ai_led.h
  * @brief   LED feedback module for AI inference on STM32U545RE-Q
  * @details Single LED control (U545 has only one user LED)
  *          Behavior similar to yellow LED on H723ZG
  ******************************************************************************
  */

#ifndef __AI_LED_H
#define __AI_LED_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* LED toggle interval for inference feedback */
#define AI_LED_TOGGLE_INTERVAL  50  /* Toggle every N inferences */

/**
  * @brief  Toggle LED (called periodically during inference)
  */
void ai_led_toggle(void);

/**
  * @brief  LED on
  */
void ai_led_on(void);

/**
  * @brief  LED off
  */
void ai_led_off(void);

/**
  * @brief  Error blink pattern
  * @param  count Number of blinks (error code)
  */
void ai_led_error_blink(int count);

/**
  * @brief  Startup blink to indicate system is alive
  */
void ai_led_startup_blink(void);

#ifdef __cplusplus
}
#endif

#endif /* __AI_LED_H */
