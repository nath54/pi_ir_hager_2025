/**
  ******************************************************************************
  * @file    ai_signal.h
  * @brief   Signal output for oscilloscope timing measurements
  * @details Uses PA8 (Arduino D7) as a timing signal pin that toggles
  *          at each inference start, creating a square wave for measurement.
  ******************************************************************************
  */

#ifndef AI_SIGNAL_H
#define AI_SIGNAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* Signal pin configuration */
#define SIGNAL_PIN          GPIO_PIN_8
#define SIGNAL_PORT         GPIOA

/**
  * @brief  Initialize the signal output pin (PA8)
  * @retval None
  */
void ai_signal_init(void);

/**
  * @brief  Toggle signal at inference start (creates square wave)
  * @note   Call this at the START of each inference.
  *         Signal alternates: HIGH->LOW->HIGH->LOW...
  *         Each level duration = one inference period.
  * @retval None
  */
void ai_signal_inference_start(void);

/**
  * @brief  Called at inference end (no-op for alternating pattern)
  * @note   The signal stays at its current level until next inference.
  * @retval None
  */
void ai_signal_inference_end(void);

/**
  * @brief  Set signal high
  * @retval None
  */
void ai_signal_set_high(void);

/**
  * @brief  Set signal low
  * @retval None
  */
void ai_signal_set_low(void);

/**
  * @brief  Toggle signal
  * @retval None
  */
void ai_signal_toggle(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_SIGNAL_H */
