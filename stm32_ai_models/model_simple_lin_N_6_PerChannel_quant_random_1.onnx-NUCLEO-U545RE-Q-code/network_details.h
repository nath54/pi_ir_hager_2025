/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-22T14:56:12+0000
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
   { .size_bytes = 300, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 10}}, .scale = {1, (const float[1]){0.003920554183423519}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "input_output" },
   { .size_bytes = 480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 30, 1, 16}}, .scale = {1, (const float[1]){0.008817403577268124}}, .zeropoint = {1, (const int16_t[1]){19}}, .name = "_MatMul_output_0_output" },
   { .size_bytes = 480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 16}}, .scale = {1, (const float[1]){0.002678550546988845}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_output_0_output" },
   { .size_bytes = 32, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {1, (const float[1]){0.0008682586485520005}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_1_output_0_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {1, (const float[1]){0.0009535410790704191}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_2_output_0_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 128}}, .scale = {1, (const float[1]){0.0010028103133663535}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_3_output_0_output" },
   { .size_bytes = 32, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {1, (const float[1]){0.0006012646481394768}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_4_output_0_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 16}}, .scale = {1, (const float[1]){0.0007524592219851911}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_5_output_0_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {1, (const float[1]){0.0007422412163577974}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "output_QuantizeLinear_Input_output" }
  },
  .nodes = (const stai_node_details[8]){
    {.id = 17, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _MatMul_output_0 */
    {.id = 20, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _Relu_output_0 */
    {.id = 26, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _Relu_1_output_0 */
    {.id = 29, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _Relu_2_output_0 */
    {.id = 32, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _Relu_3_output_0 */
    {.id = 35, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _Relu_4_output_0 */
    {.id = 38, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _Relu_5_output_0 */
    {.id = 41, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} } /* output_QuantizeLinear_Input */
  },
  .n_nodes = 8
};
#endif

