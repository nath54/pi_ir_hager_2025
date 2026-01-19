/**
  ******************************************************************************
  * @file    ai_network.h
  * @brief   AI Network module for ST.AI inference
  * @details Supports both INT8 and FLOAT32 models via QUANTIZED_INT8 macro
  ******************************************************************************
  */

#ifndef __AI_NETWORK_H
#define __AI_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include <stdbool.h>
#include <stdint.h>

/* Error codes for AI operations */
typedef enum {
  AI_OK = 0,
  AI_ERROR_RUNTIME_INIT,
  AI_ERROR_NETWORK_INIT,
  AI_ERROR_ACTIVATIONS,
  AI_ERROR_WEIGHTS,
  AI_ERROR_INPUTS,
  AI_ERROR_OUTPUTS,
  AI_ERROR_INFERENCE
} ai_error_t;

/**
  * @brief  Initialize the AI network
  * @retval AI_OK on success, error code otherwise
  */
ai_error_t ai_network_init(void);

/**
  * @brief  Run a single inference
  * @retval AI_OK on success, AI_ERROR_INFERENCE on failure
  */
ai_error_t ai_network_run_inference(void);

/**
  * @brief  Get pointer to input buffer
  * @retval Pointer to input buffer (type depends on quantization)
  */
void* ai_network_get_input(void);

/**
  * @brief  Get pointer to output buffer  
  * @retval Pointer to output buffer (type depends on quantization)
  */
void* ai_network_get_output(void);

/**
  * @brief  Get input size (number of elements)
  * @retval Input size
  */
uint32_t ai_network_get_input_size(void);

/**
  * @brief  Get output size (number of elements)
  * @retval Output size
  */
uint32_t ai_network_get_output_size(void);

/**
  * @brief  Prepare random input data for testing
  * @param  seed Random seed value
  */
void ai_network_prepare_test_input(uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* __AI_NETWORK_H */
