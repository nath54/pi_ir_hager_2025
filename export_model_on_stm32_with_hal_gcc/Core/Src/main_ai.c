/**
  ******************************************************************************
  * @file    main_ai.c
  * @brief   AI Inference main application for STM32U545RE-Q
  * @details Runs neural network inference with LED feedback
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "system_init.h"
#include "ai_network.h"
#include "ai_led.h"
#include "ai_signal.h"
#include <stdio.h>

/* Private variables ---------------------------------------------------------*/
__IO uint32_t BspButtonState = BUTTON_RELEASED;
static uint32_t inference_counter = 0;

/* Private function prototypes -----------------------------------------------*/
static void run_ai_inference_loop(void);

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the System Power */
  SystemPower_Config();

  /* Initialize BSP (LED, button, COM) */
  BSP_Init();

  /* Initialize signal output for oscilloscope (PA8) */
  ai_signal_init();

  /* Startup LED blink */
  ai_led_startup_blink();

  /* Print startup message */
  printf("\r\n========================================\r\n");
  printf("STM32 AI Inference - NUCLEO-U545RE-Q\r\n");
#ifdef QUANTIZED_INT8
  printf("Model Type: INT8 (Quantized)\r\n");
#else
  printf("Model Type: FLOAT32\r\n");
#endif
  printf("========================================\r\n\r\n");

  /* Initialize AI network */
  printf("Initializing AI network...\r\n");
  ai_error_t err = ai_network_init();

  if (err != AI_OK)
  {
    printf("ERROR: AI init failed (code %d)\r\n", err);
    ai_led_error_blink(err);
    /* Continue anyway - may recover */
  }
  else
  {
    printf("AI network ready!\r\n");
    printf("Input size: %lu elements\r\n", ai_network_get_input_size());
    printf("Output size: %lu elements\r\n\r\n", ai_network_get_output_size());
  }

  /* Run inference loop */
  run_ai_inference_loop();

  /* Should never reach here */
  return 0;
}

/**
  * @brief  Main AI inference loop
  */
static void run_ai_inference_loop(void)
{
  uint32_t tick_start;
  uint32_t tick_elapsed;

  while (1)
  {
    /* Prepare test input data */
    ai_network_prepare_test_input(HAL_GetTick() + inference_counter);

    /* Toggle LED every N inferences (like yellow on H723ZG) */
    if (inference_counter % AI_LED_TOGGLE_INTERVAL == 0)
    {
      ai_led_toggle();
    }

    /* Signal inference start (for oscilloscope) */
    ai_signal_inference_start();

    /* Run inference with timing */
    tick_start = HAL_GetTick();
    ai_error_t err = ai_network_run_inference();
    tick_elapsed = HAL_GetTick() - tick_start;

    /* Signal inference end */
    ai_signal_inference_end();

    if (err != AI_OK)
    {
      printf("Inference FAILED!\r\n");
      ai_led_error_blink(AI_ERROR_INFERENCE);
    }
    else
    {
      /* Print result */
#ifdef QUANTIZED_INT8
      int8_t* output = (int8_t*)ai_network_get_output();
      printf("OK %lums | Out[0]: %d | INT8\r\n", tick_elapsed, (int)output[0]);
#else
      float* output = (float*)ai_network_get_output();
      printf("OK %lums | Out[0]: %.4f | F32\r\n", tick_elapsed, (double)output[0]);
#endif
    }

    /* Increment counter */
    inference_counter++;
    if (inference_counter > 10000)
    {
      inference_counter = 0;
    }

    /* Small delay between inferences */
    HAL_Delay(100);
  }
}

/**
  * @brief BSP Push Button callback
  * @param Button Specifies the pressed button
  * @retval None
  */
void BSP_PB_Callback(Button_TypeDef Button)
{
  if (Button == BUTTON_USER)
  {
    BspButtonState = BUTTON_PRESSED;
  }
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
    ai_led_error_blink(10);
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  printf("Assert failed: file %s on line %lu\r\n", file, line);
}
#endif
