/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-19T21:04:59+0000
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
  .tensors = (const stai_tensor[5]) {
   { .size_bytes = 1200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 30, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_GRU_output_0_output0" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_GRU_output_0_output1" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Gather_1_output_0_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "output_output" }
  },
  .nodes = (const stai_node_details[3]){
    {.id = 17, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {2, (const int32_t[2]){1, 2}} }, /* _GRU_output_0 */
    {.id = 21, .type = AI_LAYER_GATHER_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _Gather_1_output_0 */
    {.id = 23, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} } /* output */
  },
  .n_nodes = 3
};
#endif

