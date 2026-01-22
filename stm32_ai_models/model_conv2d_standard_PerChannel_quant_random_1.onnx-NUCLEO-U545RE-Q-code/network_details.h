/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-22T15:42:32+0000
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
   { .size_bytes = 300, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 10}}, .scale = {1, (const float[1]){0.0039206212386488914}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "input_output" },
   { .size_bytes = 300, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 30, 10, 1}}, .scale = {1, (const float[1]){0.0039206212386488914}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Unsqueeze_output_0_to_chfirst_output" },
   { .size_bytes = 2400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 30, 10, 8}}, .scale = {1, (const float[1]){0.002955144504085183}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_output_0_output" },
   { .size_bytes = 9600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 8}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_MaxPool_output_0_DequantizeLinear_Output_output" },
   { .size_bytes = 14400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 12}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_1_output_0_output" },
   { .size_bytes = 21600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 18}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_2_output_0_output" },
   { .size_bytes = 5400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 30, 10, 18}}, .scale = {1, (const float[1]){0.0005715792067348957}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_2_output_0_0_0__Flatten_output_0_to_chlast_conversion_output" },
   { .size_bytes = 5400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 18, 30, 10}}, .scale = {1, (const float[1]){0.0005715792067348957}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Flatten_output_0_to_chlast_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {1, (const float[1]){6.971834227442741e-05}}, .zeropoint = {1, (const int16_t[1]){-40}}, .name = "output_QuantizeLinear_Input_output" }
  },
  .nodes = (const stai_node_details[8]){
    {.id = 9, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _Unsqueeze_output_0_to_chfirst */
    {.id = 12, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _Relu_output_0 */
    {.id = 17, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _MaxPool_output_0_DequantizeLinear_Output */
    {.id = 18, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _Relu_1_output_0 */
    {.id = 24, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _Relu_2_output_0 */
    {.id = 24, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _Relu_2_output_0_0_0__Flatten_output_0_to_chlast_conversion */
    {.id = 30, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _Flatten_output_0_to_chlast */
    {.id = 33, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} } /* output_QuantizeLinear_Input */
  },
  .n_nodes = 8
};
#endif

