/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-17T10:27:06+0000
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
  .tensors = (const stai_tensor[14]) {
   { .size_bytes = 1200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 30, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_output" },
   { .size_bytes = 1200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Unsqueeze_output_0_to_chfirst_output" },
   { .size_bytes = 19200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Conv_output_0_output" },
   { .size_bytes = 19200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_output_0_output" },
   { .size_bytes = 28800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 24}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Conv_1_output_0_output" },
   { .size_bytes = 28800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 24}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_1_output_0_output" },
   { .size_bytes = 43200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 36}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Conv_2_output_0_output" },
   { .size_bytes = 43200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 36}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_2_output_0_output" },
   { .size_bytes = 64800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 54}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Conv_3_output_0_output" },
   { .size_bytes = 64800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 54}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_3_output_0_output" },
   { .size_bytes = 97200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 81}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Conv_4_output_0_output" },
   { .size_bytes = 97200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 10, 81}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Relu_4_output_0_output" },
   { .size_bytes = 97200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 81, 30, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Flatten_output_0_to_chlast_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "output_output" }
  },
  .nodes = (const stai_node_details[13]){
    {.id = 3, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _Unsqueeze_output_0_to_chfirst */
    {.id = 6, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _Conv_output_0 */
    {.id = 7, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _Relu_output_0 */
    {.id = 11, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _Conv_1_output_0 */
    {.id = 12, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _Relu_1_output_0 */
    {.id = 16, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _Conv_2_output_0 */
    {.id = 17, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _Relu_2_output_0 */
    {.id = 21, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* _Conv_3_output_0 */
    {.id = 22, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* _Relu_3_output_0 */
    {.id = 26, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* _Conv_4_output_0 */
    {.id = 27, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* _Relu_4_output_0 */
    {.id = 29, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* _Flatten_output_0_to_chlast */
    {.id = 31, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} } /* output */
  },
  .n_nodes = 13
};
#endif

