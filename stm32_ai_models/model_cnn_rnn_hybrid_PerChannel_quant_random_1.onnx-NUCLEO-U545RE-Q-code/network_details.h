/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-22T15:45:20+0000
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[9]) {
   { .size_bytes = 300, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 10}}, .scale = {1, (const float[1]){0.0039208256639540195}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "input_output" },
   { .size_bytes = 224, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 28, 8}}, .scale = {1, (const float[1]){0.00370704079978168}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_output_0_output" },
   { .size_bytes = 896, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 28, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_output_0_0_conversion_output" },
   { .size_bytes = 336, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 28, 1, 3}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_GRU_output_0_output0" },
   { .size_bytes = 12, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 3}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_GRU_output_0_output1" },
   { .size_bytes = 84, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 28, 1, 3}}, .scale = {1, (const float[1]){0.0039396691136062145}}, .zeropoint = {1, (const int16_t[1]){58}}, .name = "_GRU_output_0_0_0__GRU_output_0_out_0_transpose_conversion_output" },
   { .size_bytes = 84, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 3, 28}}, .scale = {1, (const float[1]){0.0039396691136062145}}, .zeropoint = {1, (const int16_t[1]){58}}, .name = "_GRU_output_0_out_0_transpose_output" },
   { .size_bytes = 3, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 3}}, .scale = {1, (const float[1]){0.0039396691136062145}}, .zeropoint = {1, (const int16_t[1]){58}}, .name = "_Gather_1_output_0_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {1, (const float[1]){0.003122653579339385}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "output_QuantizeLinear_Input_output" }
  },
  .nodes = (const stai_node_details[7]){
    {.id = 12, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _Relu_output_0 */
    {.id = 12, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _Relu_output_0_0_conversion */
    {.id = 18, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {2, (const int32_t[2]){3, 4}} }, /* _GRU_output_0 */
    {.id = 18, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _GRU_output_0_0_0__GRU_output_0_out_0_transpose_conversion */
    {.id = 18, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _GRU_output_0_out_0_transpose */
    {.id = 29, .type = AI_LAYER_GATHER_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _Gather_1_output_0 */
    {.id = 32, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} } /* output_QuantizeLinear_Input */
  },
  .n_nodes = 7
};
#endif

