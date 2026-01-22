/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-22T15:46:33+0000
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
  .tensors = (const stai_tensor[38]) {
   { .size_bytes = 300, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 10}}, .scale = {1, (const float[1]){0.00392132019624114}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "input_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_MatMul_4_output_0_bias_0_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output" },
   { .size_bytes = 300, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 30, 10, 1}}, .scale = {1, (const float[1]){0.00392132019624114}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Unsqueeze_output_0_to_chfirst_output" },
   { .size_bytes = 1792, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 28, 8, 8}}, .scale = {1, (const float[1]){0.004052215255796909}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Relu_output_0_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 224, 1, 12}}, .scale = {1, (const float[1]){0.004095243755728006}}, .zeropoint = {1, (const int16_t[1]){-25}}, .name = "_MatMul_output_0_gemm_to_dense_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.004095243755728006}}, .zeropoint = {1, (const int16_t[1]){-25}}, .name = "_MatMul_output_0_gemm_to_dense_out_transpose_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.005318579729646444}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "_Add_output_0_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 12}}, .scale = {1, (const float[1]){0.005318579729646444}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "transpose_a_MatMul_3_output_0_out_output" },
   { .size_bytes = 10752, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 12}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output" },
   { .size_bytes = 1792, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 8}}, .scale = {1, (const float[1]){0.004052215255796909}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_MatMul_1_output_0_gemm_to_dense_in_transpose_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 224, 1, 12}}, .scale = {1, (const float[1]){0.003841881174594164}}, .zeropoint = {1, (const int16_t[1]){-41}}, .name = "_MatMul_1_output_0_gemm_to_dense_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.003841881174594164}}, .zeropoint = {1, (const int16_t[1]){-41}}, .name = "_MatMul_1_output_0_gemm_to_dense_out_transpose_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.004937407560646534}}, .zeropoint = {1, (const int16_t[1]){-28}}, .name = "_Add_1_output_0_output" },
   { .size_bytes = 10752, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Add_1_output_0_0_1__MatMul_3_output_0_conversion_output" },
   { .size_bytes = 200704, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_MatMul_3_output_0_output" },
   { .size_bytes = 50176, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {1, (const float[1]){0.003893679240718484}}, .zeropoint = {1, (const int16_t[1]){-90}}, .name = "_MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output" },
   { .size_bytes = 50176, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {1, (const float[1]){0.003893679240718484}}, .zeropoint = {1, (const int16_t[1]){-90}}, .name = "transpose_out_MatMul_3_output_0_out_output" },
   { .size_bytes = 50176, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {1, (const float[1]){0.0011240083258599043}}, .zeropoint = {1, (const int16_t[1]){-90}}, .name = "_Div_output_0_output" },
   { .size_bytes = 200704, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Div_output_0_0_0__Softmax_output_0_conversion_output" },
   { .size_bytes = 200704, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Softmax_output_0_output" },
   { .size_bytes = 50176, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {1, (const float[1]){0.003921568859368563}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output" },
   { .size_bytes = 50176, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {1, (const float[1]){0.003921568859368563}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "transpose_a_MatMul_4_output_0_out_output" },
   { .size_bytes = 200704, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 224}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output" },
   { .size_bytes = 1792, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 8}}, .scale = {1, (const float[1]){0.004052215255796909}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_MatMul_2_output_0_gemm_to_dense_in_transpose_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 224, 1, 12}}, .scale = {1, (const float[1]){0.0034422490280121565}}, .zeropoint = {1, (const int16_t[1]){23}}, .name = "_MatMul_2_output_0_gemm_to_dense_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.0034422490280121565}}, .zeropoint = {1, (const int16_t[1]){23}}, .name = "_MatMul_2_output_0_gemm_to_dense_out_transpose_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.004716664552688599}}, .zeropoint = {1, (const int16_t[1]){31}}, .name = "_Add_2_output_0_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 12}}, .scale = {1, (const float[1]){0.004716664552688599}}, .zeropoint = {1, (const int16_t[1]){31}}, .name = "transpose_b_MatMul_4_output_0_out_output" },
   { .size_bytes = 10752, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 12}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output" },
   { .size_bytes = 10752, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 224, 12}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_MatMul_4_output_0_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 224, 12}}, .scale = {1, (const float[1]){0.002826663199812174}}, .zeropoint = {1, (const int16_t[1]){30}}, .name = "_MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output" },
   { .size_bytes = 2688, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {1, (const float[1]){0.002826663199812174}}, .zeropoint = {1, (const int16_t[1]){30}}, .name = "transpose_out_MatMul_4_output_0_out_output" },
   { .size_bytes = 10752, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 12, 224}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output" },
   { .size_bytes = 48, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 12, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_ReduceMean_output_0_output" },
   { .size_bytes = 48, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 12}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_ReduceMean_output_0_Mul_output" },
   { .size_bytes = 12, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 12}}, .scale = {1, (const float[1]){0.002824777038767934}}, .zeropoint = {1, (const int16_t[1]){30}}, .name = "_ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {1, (const float[1]){0.000644468585960567}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "output_QuantizeLinear_Input_output" }
  },
  .nodes = (const stai_node_details[37]){
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {0, NULL}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _MatMul_4_output_0_bias_0_conversion */
    {.id = 45, .type = AI_LAYER_NL_TYPE, .input_tensors = {0, NULL}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion */
    {.id = 12, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _Unsqueeze_output_0_to_chfirst */
    {.id = 15, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _Relu_output_0 */
    {.id = 24, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _MatMul_output_0_gemm_to_dense */
    {.id = 24, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _MatMul_output_0_gemm_to_dense_out_transpose */
    {.id = 33, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _Add_output_0 */
    {.id = 45, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* transpose_a_MatMul_3_output_0_out */
    {.id = 45, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion */
    {.id = 25, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* _MatMul_1_output_0_gemm_to_dense_in_transpose */
    {.id = 25, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* _MatMul_1_output_0_gemm_to_dense */
    {.id = 25, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* _MatMul_1_output_0_gemm_to_dense_out_transpose */
    {.id = 34, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* _Add_1_output_0 */
    {.id = 34, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* _Add_1_output_0_0_1__MatMul_3_output_0_conversion */
    {.id = 45, .type = AI_LAYER_MATMUL_TYPE, .input_tensors = {3, (const int32_t[3]){9, 14, 2}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* _MatMul_3_output_0 */
    {.id = 45, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion */
    {.id = 45, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* transpose_out_MatMul_3_output_0_out */
    {.id = 48, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* _Div_output_0 */
    {.id = 48, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* _Div_output_0_0_0__Softmax_output_0_conversion */
    {.id = 51, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* _Softmax_output_0 */
    {.id = 51, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion */
    {.id = 54, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* transpose_a_MatMul_4_output_0_out */
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion */
    {.id = 26, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* _MatMul_2_output_0_gemm_to_dense_in_transpose */
    {.id = 26, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* _MatMul_2_output_0_gemm_to_dense */
    {.id = 26, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* _MatMul_2_output_0_gemm_to_dense_out_transpose */
    {.id = 35, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* _Add_2_output_0 */
    {.id = 54, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* transpose_b_MatMul_4_output_0_out */
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion */
    {.id = 54, .type = AI_LAYER_MATMUL_TYPE, .input_tensors = {3, (const int32_t[3]){23, 29, 1}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* _MatMul_4_output_0 */
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion */
    {.id = 54, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* transpose_out_MatMul_4_output_0_out */
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion */
    {.id = 57, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){33}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* _ReduceMean_output_0 */
    {.id = 57, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* _ReduceMean_output_0_Mul */
    {.id = 57, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion */
    {.id = 60, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} } /* output_QuantizeLinear_Input */
  },
  .n_nodes = 37
};
#endif

