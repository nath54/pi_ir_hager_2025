/**
  ******************************************************************************
  * @file    ai_signal.c
  * @brief   Signal output for oscilloscope timing measurements
  * @details Uses PA8 (Arduino D7) as a timing signal pin.
  *          Signal toggles at each inference start, creating a square wave
  *          where each HIGH or LOW period = one inference duration.
  ******************************************************************************
  */

#include "ai_signal.h"

/* Current signal state for alternating pattern */
static uint8_t signal_state = 0;

/**
  * @brief  Initialize the signal output pin (PA8)
  */
void ai_signal_init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* Enable GPIOA clock */
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* Configure PA8 as push-pull output, high speed */
  GPIO_InitStruct.Pin = SIGNAL_PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(SIGNAL_PORT, &GPIO_InitStruct);

  /* Start with signal LOW */
  HAL_GPIO_WritePin(SIGNAL_PORT, SIGNAL_PIN, GPIO_PIN_RESET);
  signal_state = 0;
}

/**
  * @brief  Toggle signal at inference start
  * @note   Creates alternating pattern:
  *         HIGH->inference->LOW->inference->HIGH->...
  *         Each edge marks the start of a new inference.
  */
void ai_signal_inference_start(void)
{
  /* Toggle state */
  signal_state = !signal_state;

  if (signal_state)
  {
    HAL_GPIO_WritePin(SIGNAL_PORT, SIGNAL_PIN, GPIO_PIN_SET);
  }
  else
  {
    HAL_GPIO_WritePin(SIGNAL_PORT, SIGNAL_PIN, GPIO_PIN_RESET);
  }
}

/**
  * @brief  Called at inference end (no-op for alternating pattern)
  */
void ai_signal_inference_end(void)
{
  /* For alternating pattern, signal stays at current level until next inference */
  /* No action needed here */
}

/**
  * @brief  Set signal high
  */
void ai_signal_set_high(void)
{
  HAL_GPIO_WritePin(SIGNAL_PORT, SIGNAL_PIN, GPIO_PIN_SET);
  signal_state = 1;
}

/**
  * @brief  Set signal low
  */
void ai_signal_set_low(void)
{
  HAL_GPIO_WritePin(SIGNAL_PORT, SIGNAL_PIN, GPIO_PIN_RESET);
  signal_state = 0;
}

/**
  * @brief  Toggle signal
  */
void ai_signal_toggle(void)
{
  HAL_GPIO_TogglePin(SIGNAL_PORT, SIGNAL_PIN);
  signal_state = !signal_state;
}
