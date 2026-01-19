/**
  ******************************************************************************
  * @file    ai_network.c
  * @brief   AI Network implementation using ST.AI runtime
  * @details Handles network initialization, buffer management, and inference
  ******************************************************************************
  */

#include "ai_network.h"

/* Network files (from AI directory, added to include path by CMake) */
#include "network.h"
#include "network_data.h"

/* Network context and buffers */
STAI_ALIGNED(STAI_NETWORK_CONTEXT_ALIGNMENT)
static stai_network network[STAI_NETWORK_CONTEXT_SIZE];

STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations[STAI_NETWORK_ACTIVATIONS_SIZE];

/* Input/output buffers - type depends on quantization */
#ifdef QUANTIZED_INT8
static int8_t input_data[STAI_NETWORK_IN_1_SIZE];
static int8_t output_data[STAI_NETWORK_OUT_1_SIZE];
#else
static float input_data[STAI_NETWORK_IN_1_SIZE];
static float output_data[STAI_NETWORK_OUT_1_SIZE];
#endif

/**
  * @brief  Initialize the AI network
  */
ai_error_t ai_network_init(void)
{
  stai_return_code err;

  /* Initialize runtime */
  err = stai_runtime_init();
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_RUNTIME_INIT;
  }

  /* Initialize network */
  err = stai_network_init(network);
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_NETWORK_INIT;
  }

  /* Set activations */
  const stai_ptr act_ptrs[] = {(stai_ptr)activations};
  err = stai_network_set_activations(network, act_ptrs, 1);
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_ACTIVATIONS;
  }

  /* Set weights */
  const stai_ptr wgt_ptrs[] = {(stai_ptr)g_network_weights_array};
  err = stai_network_set_weights(network, wgt_ptrs, 1);
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_WEIGHTS;
  }

  /* Set inputs */
  const stai_ptr in_ptrs[] = {(stai_ptr)input_data};
  err = stai_network_set_inputs(network, in_ptrs, 1);
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_INPUTS;
  }

  /* Set outputs */
  const stai_ptr out_ptrs[] = {(stai_ptr)output_data};
  err = stai_network_set_outputs(network, out_ptrs, 1);
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_OUTPUTS;
  }

  return AI_OK;
}

/**
  * @brief  Run a single inference
  */
ai_error_t ai_network_run_inference(void)
{
  stai_return_code err = stai_network_run(network, STAI_MODE_SYNC);
  if (err != STAI_SUCCESS)
  {
    return AI_ERROR_INFERENCE;
  }
  return AI_OK;
}

/**
  * @brief  Get pointer to input buffer
  */
void* ai_network_get_input(void)
{
  return (void*)input_data;
}

/**
  * @brief  Get pointer to output buffer
  */
void* ai_network_get_output(void)
{
  return (void*)output_data;
}

/**
  * @brief  Get input size
  */
uint32_t ai_network_get_input_size(void)
{
  return STAI_NETWORK_IN_1_SIZE;
}

/**
  * @brief  Get output size
  */
uint32_t ai_network_get_output_size(void)
{
  return STAI_NETWORK_OUT_1_SIZE;
}

/**
  * @brief  Prepare random test input data
  */
void ai_network_prepare_test_input(uint32_t seed)
{
  for (uint32_t i = 0; i < STAI_NETWORK_IN_1_SIZE; i++)
  {
#ifdef QUANTIZED_INT8
    input_data[i] = (int8_t)((seed * 13 + i * 7) % 256 - 128);
#else
    input_data[i] = (float)((seed * 13 + i * 7) % 100) / 100.0f;
#endif
  }
}
