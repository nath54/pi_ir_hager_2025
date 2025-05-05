/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-05-05T13:38:58+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x7fa523093e9bf1508d01939707d20bfe"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-05-05T13:38:58+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_args_00_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 195700, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_args_00_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 195700, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  slice_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 190, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  nl_32_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_34_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  slice_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 190, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  nl_27_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  gemm_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_30_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  slice_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 190, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  nl_22_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  slice_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 190, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  nl_17_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_20_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  slice_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 190, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  slice_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 190, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  nl_7_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  gemm_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  concat_37_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  gemm_38_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  transpose_40_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  gemm_39_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_43_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  nl_45_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  transpose_51_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  gemm_50_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  gemm_53_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  nl_56_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  gemm_57_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  gemm_59_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  concat_60_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_62_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  reduce_63_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  reduce_63_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_65_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  reduce_66_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  reduce_66_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  nl_68_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_72_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_69_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_70_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_74_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_75_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  gemm_77_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  nl_78_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  gemm_79_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_80_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  reduce_81_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  reduce_81_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_83_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  reduce_84_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  reduce_84_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  nl_86_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_90_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_87_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_88_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_92_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_93_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  gemm_95_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  transpose_97_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  gemm_96_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  gemm_99_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_100_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  nl_102_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  gemm_103_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  gemm_105_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  gemm_106_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  transpose_108_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  gemm_107_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  gemm_110_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_111_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  nl_113_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  gemm_114_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  gemm_116_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  concat_117_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_119_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  reduce_120_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  reduce_120_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_122_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  reduce_123_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  reduce_123_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  nl_125_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_129_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_126_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_127_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_131_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  gemm_134_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  nl_135_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  gemm_136_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_137_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  reduce_138_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  reduce_138_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_140_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  reduce_141_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  reduce_141_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  nl_143_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_147_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_144_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_145_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_149_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_150_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  reduce_152_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  reduce_152_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_154_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  reduce_155_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  reduce_155_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  nl_157_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_161_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_158_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_159_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_163_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_164_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 408, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  gemm_167_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  nl_168_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  gemm_116_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  gemm_110_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  gemm_105_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  gemm_99_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  gemm_59_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  gemm_53_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant66_2D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant29_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant31_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant33_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant35_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant37_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant39_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  gemm_34_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  gemm_29_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  gemm_24_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  gemm_9_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  gemm_38_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  gemm_39_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_43_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_43_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  gemm_50_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  gemm_57_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4624, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  reduce_63_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  reduce_66_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_75_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_75_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  gemm_77_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  gemm_77_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  gemm_79_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  gemm_79_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_93_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_93_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  gemm_95_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  gemm_96_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  gemm_103_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  gemm_106_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  gemm_107_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  gemm_114_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4624, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  gemm_134_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  gemm_134_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  gemm_136_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2312, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  gemm_136_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_150_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_150_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_164_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_164_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 68, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  gemm_167_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2448, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  gemm_167_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant29, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &arith_constant29_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant31, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &arith_constant31_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant33, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &arith_constant33_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant35, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &arith_constant35_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant37, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &arith_constant37_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant39, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &arith_constant39_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant66_2D, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &arith_constant66_2D_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  concat_117_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &concat_117_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  concat_37_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &concat_37_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  concat_60_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &concat_60_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_12_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_12_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_scratch0, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 5), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_12_scratch0_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 5, 1), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_12_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_17_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_17_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 5), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_17_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 5, 1), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_17_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_22_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_22_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 5), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_22_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 5, 1), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_22_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_27_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_27_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_scratch0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 5), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_27_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 5, 1), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_27_weights_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_32_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_32_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 5), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_32_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 5, 1), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_32_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_7_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 5), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 5, 1), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &conv2d_7_weights_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_100_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_100_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &eltwise_10_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output0, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_10_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_111_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_111_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_119_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_119_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_119_output0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_119_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_122_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_122_output_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_122_output0, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_122_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_126_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_126_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_127_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_127_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_127_output0, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &eltwise_127_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_129_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_129_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_131_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_131_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_bias, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_132_bias_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_132_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_scale, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_132_scale_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_137_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_137_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_137_output0, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_137_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_140_output, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_140_output_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_140_output0, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_140_output_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_144_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_144_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_145_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_145_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_145_output0, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &eltwise_145_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_147_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_147_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_149_output, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_149_output_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_150_bias, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_150_bias_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_150_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_150_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_150_output0, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_150_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_150_scale, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_150_scale_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_154_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_154_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_154_output0, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_154_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_158_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_158_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_159_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_159_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_159_output0, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &eltwise_159_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_15_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &eltwise_15_output_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_15_output0, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_15_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_161_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_161_output_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_163_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_163_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_164_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_164_bias_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_164_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_164_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_164_output0, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 408, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1632, 1632),
  1, &eltwise_164_output_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_164_scale, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_164_scale_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_20_output, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &eltwise_20_output_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_20_output0, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_20_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_25_output, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &eltwise_25_output_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_25_output0, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_25_output_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_30_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &eltwise_30_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_30_output0, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_30_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_35_output, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &eltwise_35_output_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_35_output0, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_35_output_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_43_bias, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_43_bias_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_43_output, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_43_output_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_43_scale, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_43_scale_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_54_output_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_62_output, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_62_output_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_62_output0, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_62_output_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_65_output, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_65_output_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_65_output0, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_65_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_69_output, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_69_output_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_70_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_70_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_70_output0, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &eltwise_70_output_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_72_output, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_72_output_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_74_output, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_74_output_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_75_bias, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_75_bias_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_75_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_75_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_75_scale, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_75_scale_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_80_output, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_80_output_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_80_output0, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_80_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_83_output, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_83_output_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_83_output0, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 6, 1), AI_STRIDE_INIT(4, 4, 4, 272, 1632),
  1, &eltwise_83_output_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_87_output, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_87_output_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_88_output, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_88_output_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_88_output0, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &eltwise_88_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_90_output, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_90_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_92_output, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_92_output_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_93_bias, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_93_bias_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_93_output, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_93_output_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_93_scale, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &eltwise_93_scale_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  gemm_103_output, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_103_output_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  gemm_103_output0, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_103_output_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  gemm_103_weights, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_103_weights_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  gemm_105_bias, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_105_bias_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  gemm_105_output, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_105_output_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  gemm_105_output0, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_105_output_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  gemm_106_output, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_106_output_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  gemm_106_output0, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_106_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  gemm_106_weights, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_106_weights_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  gemm_107_output, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_107_output_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  gemm_107_output0, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_107_output_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  gemm_107_weights, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_107_weights_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_bias, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_110_bias_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_output, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &gemm_110_output_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_output0, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_110_output_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  gemm_114_output, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_114_output_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  gemm_114_output0, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_114_output_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  gemm_114_weights, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_114_weights_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  gemm_116_bias, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_116_bias_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  gemm_116_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_116_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  gemm_116_output0, AI_STATIC,
  132, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_116_output_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_bias, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_118_bias_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_output, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_118_output_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_weights, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 68, 68, 1, 1), AI_STRIDE_INIT(4, 4, 272, 18496, 18496),
  1, &gemm_118_weights_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  gemm_134_bias, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 1), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_134_bias_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  gemm_134_output, AI_STATIC,
  137, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_134_output_array, NULL)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  gemm_134_weights, AI_STATIC,
  138, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_134_weights_array, NULL)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  gemm_136_bias, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_136_bias_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  gemm_136_output, AI_STATIC,
  140, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_136_output_array, NULL)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  gemm_136_weights, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 34, 68, 1, 1), AI_STRIDE_INIT(4, 4, 136, 9248, 9248),
  1, &gemm_136_weights_array, NULL)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 17), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_14_output_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output0, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &gemm_14_output_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_weights, AI_STATIC,
  144, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_14_weights_array, NULL)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  gemm_167_bias, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_167_bias_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  gemm_167_output, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_167_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  gemm_167_weights, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 408, 6, 1, 1), AI_STRIDE_INIT(4, 4, 1632, 9792, 9792),
  1, &gemm_167_weights_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_output, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 17), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_19_output_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_output0, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &gemm_19_output_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_weights, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_19_weights_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_output, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 17), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_24_output_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_output0, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &gemm_24_output_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_weights, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_24_weights_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_output, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 17), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_29_output_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_output0, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &gemm_29_output_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_weights, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_29_weights_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_output, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 17), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_34_output_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_output0, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &gemm_34_output_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_weights, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_34_weights_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  gemm_38_output, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_38_output_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  gemm_38_output0, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_38_output_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  gemm_38_weights, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_38_weights_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  gemm_39_output, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_39_output_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  gemm_39_output0, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_39_output_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  gemm_39_weights, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_39_weights_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_bias, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_42_bias_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_output, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &gemm_42_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_output0, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_42_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_output, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_46_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_output0, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_46_output_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_weights, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_46_weights_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_bias, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_48_bias_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_output, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_48_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_output0, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_48_output_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_output, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_49_output_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_output0, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_49_output_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_weights, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_49_weights_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_output, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_50_output_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_output0, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_50_output_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_weights, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_50_weights_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  gemm_53_bias, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_53_bias_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  gemm_53_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &gemm_53_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  gemm_53_output0, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_53_output_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_output, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_57_output_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_output0, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_57_output_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_weights, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_57_weights_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_bias, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_59_bias_array, NULL)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_output, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_59_output_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_output0, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_59_output_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_bias, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_61_bias_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_output, AI_STATIC,
  191, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_61_output_array, NULL)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_weights, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 68, 68, 1, 1), AI_STRIDE_INIT(4, 4, 272, 18496, 18496),
  1, &gemm_61_weights_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  gemm_77_bias, AI_STATIC,
  193, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 1), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_77_bias_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  gemm_77_output, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_77_output_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_77_weights, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_77_weights_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  gemm_79_bias, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_79_bias_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  gemm_79_output, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 6), AI_STRIDE_INIT(4, 4, 4, 272, 272),
  1, &gemm_79_output_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  gemm_79_weights, AI_STATIC,
  198, 0x0,
  AI_SHAPE_INIT(4, 34, 68, 1, 1), AI_STRIDE_INIT(4, 4, 136, 9248, 9248),
  1, &gemm_79_weights_array, NULL)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  gemm_95_output, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_95_output_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  gemm_95_output0, AI_STATIC,
  200, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_95_output_array, NULL)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  gemm_95_weights, AI_STATIC,
  201, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_95_weights_array, NULL)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  gemm_96_output, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &gemm_96_output_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  gemm_96_output0, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 6, 1), AI_STRIDE_INIT(4, 4, 4, 136, 816),
  1, &gemm_96_output_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  gemm_96_weights, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 68, 34, 1, 1), AI_STRIDE_INIT(4, 4, 272, 9248, 9248),
  1, &gemm_96_weights_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  gemm_99_bias, AI_STATIC,
  205, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_99_bias_array, NULL)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  gemm_99_output, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &gemm_99_output_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  gemm_99_output0, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_99_output_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_output, AI_STATIC,
  208, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 17), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_9_output_array, NULL)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_output0, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 17, 1), AI_STRIDE_INIT(4, 4, 4, 16, 272),
  1, &gemm_9_output_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_weights, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_9_weights_array, NULL)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  nl_102_output, AI_STATIC,
  211, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_102_output_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  nl_102_output0, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &nl_102_output_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  nl_113_output, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_113_output_array, NULL)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  nl_113_output0, AI_STATIC,
  214, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &nl_113_output_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  nl_125_output, AI_STATIC,
  215, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_125_output_array, NULL)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  nl_125_output0, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_125_output_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_nl_output, AI_STATIC,
  217, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_12_nl_output_array, NULL)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  nl_135_nl_output, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &nl_135_nl_output_array, NULL)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  nl_143_output, AI_STATIC,
  219, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_143_output_array, NULL)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  nl_143_output0, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_143_output_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  nl_157_output, AI_STATIC,
  221, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_157_output_array, NULL)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  nl_157_output0, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_157_output_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  nl_168_nl_output, AI_STATIC,
  223, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_168_nl_output_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  nl_17_nl_output, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_17_nl_output_array, NULL)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  nl_22_nl_output, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_22_nl_output_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  nl_27_nl_output, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_27_nl_output_array, NULL)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  nl_32_nl_output, AI_STATIC,
  227, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_32_nl_output_array, NULL)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  nl_45_output, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_45_output_array, NULL)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  nl_45_output0, AI_STATIC,
  229, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &nl_45_output_array, NULL)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  nl_56_output, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 6), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_56_output_array, NULL)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  nl_56_output0, AI_STATIC,
  231, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 1), AI_STRIDE_INIT(4, 4, 4, 24, 144),
  1, &nl_56_output_array, NULL)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  nl_68_output, AI_STATIC,
  232, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_68_output_array, NULL)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  nl_68_output0, AI_STATIC,
  233, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_68_output_array, NULL)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  nl_78_nl_output, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 6), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &nl_78_nl_output_array, NULL)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  nl_7_nl_output, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 17), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_7_nl_output_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_output, AI_STATIC,
  236, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_86_output_array, NULL)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_output0, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_86_output_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  reduce_120_Mul_output, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_120_Mul_output_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  reduce_120_Mul_output0, AI_STATIC,
  239, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_120_Mul_output_array, NULL)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  reduce_120_output, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_120_output_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  reduce_120_output0, AI_STATIC,
  241, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_120_output_array, NULL)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  reduce_123_Mul_output, AI_STATIC,
  242, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_123_Mul_output_array, NULL)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  reduce_123_output, AI_STATIC,
  243, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_123_output_array, NULL)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  reduce_123_output0, AI_STATIC,
  244, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_123_output_array, NULL)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  reduce_138_Mul_output, AI_STATIC,
  245, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_138_Mul_output_array, NULL)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  reduce_138_Mul_output0, AI_STATIC,
  246, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_138_Mul_output_array, NULL)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  reduce_138_output, AI_STATIC,
  247, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_138_output_array, NULL)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  reduce_138_output0, AI_STATIC,
  248, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_138_output_array, NULL)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  reduce_141_Mul_output, AI_STATIC,
  249, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_141_Mul_output_array, NULL)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  reduce_141_output, AI_STATIC,
  250, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_141_output_array, NULL)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  reduce_141_output0, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_141_output_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  reduce_152_Mul_output, AI_STATIC,
  252, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_152_Mul_output_array, NULL)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  reduce_152_Mul_output0, AI_STATIC,
  253, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_152_Mul_output_array, NULL)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  reduce_152_output, AI_STATIC,
  254, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_152_output_array, NULL)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  reduce_152_output0, AI_STATIC,
  255, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_152_output_array, NULL)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  reduce_155_Mul_output, AI_STATIC,
  256, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_155_Mul_output_array, NULL)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  reduce_155_output, AI_STATIC,
  257, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_155_output_array, NULL)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  reduce_155_output0, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_155_output_array, NULL)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  reduce_63_Mul_output, AI_STATIC,
  259, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_63_Mul_output_array, NULL)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  reduce_63_Mul_output0, AI_STATIC,
  260, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_63_Mul_output_array, NULL)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  reduce_63_Mul_scale, AI_STATIC,
  261, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_63_Mul_scale_array, NULL)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  reduce_63_output, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_63_output_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  reduce_63_output0, AI_STATIC,
  263, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_63_output_array, NULL)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  reduce_66_Mul_bias, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_66_Mul_bias_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  reduce_66_Mul_output, AI_STATIC,
  265, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_66_Mul_output_array, NULL)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  reduce_66_output, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_66_output_array, NULL)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  reduce_66_output0, AI_STATIC,
  267, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_66_output_array, NULL)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_Mul_output, AI_STATIC,
  268, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_81_Mul_output_array, NULL)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_Mul_output0, AI_STATIC,
  269, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 6), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_81_Mul_output_array, NULL)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_output, AI_STATIC,
  270, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_81_output_array, NULL)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_output0, AI_STATIC,
  271, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_81_output_array, NULL)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  reduce_84_Mul_output, AI_STATIC,
  272, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_84_Mul_output_array, NULL)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  reduce_84_output, AI_STATIC,
  273, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 1), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &reduce_84_output_array, NULL)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  reduce_84_output0, AI_STATIC,
  274, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &reduce_84_output_array, NULL)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_args_00_Transpose_output, AI_STATIC,
  275, 0x0,
  AI_SHAPE_INIT(4, 1, 1030, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4120, 20600),
  1, &serving_default_args_00_Transpose_output_array, NULL)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_args_00_output, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 38, 1030), AI_STRIDE_INIT(4, 4, 4, 20, 760),
  1, &serving_default_args_00_output_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  slice_1_output, AI_STATIC,
  277, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &slice_1_output_array, NULL)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  slice_2_output, AI_STATIC,
  278, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &slice_2_output_array, NULL)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  slice_3_output, AI_STATIC,
  279, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &slice_3_output_array, NULL)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  slice_4_output, AI_STATIC,
  280, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &slice_4_output_array, NULL)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  slice_5_output, AI_STATIC,
  281, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &slice_5_output_array, NULL)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  slice_6_output, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 38), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &slice_6_output_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  transpose_108_output, AI_STATIC,
  283, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 34), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &transpose_108_output_array, NULL)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  transpose_108_output0, AI_STATIC,
  284, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 34, 1), AI_STRIDE_INIT(4, 4, 4, 24, 816),
  1, &transpose_108_output_array, NULL)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  transpose_40_output, AI_STATIC,
  285, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 34), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &transpose_40_output_array, NULL)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  transpose_40_output0, AI_STATIC,
  286, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 34, 1), AI_STRIDE_INIT(4, 4, 4, 24, 816),
  1, &transpose_40_output_array, NULL)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  transpose_51_output, AI_STATIC,
  287, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 34), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &transpose_51_output_array, NULL)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  transpose_51_output0, AI_STATIC,
  288, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 34, 1), AI_STRIDE_INIT(4, 4, 4, 24, 816),
  1, &transpose_51_output_array, NULL)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  transpose_97_output, AI_STATIC,
  289, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 34), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &transpose_97_output_array, NULL)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  transpose_97_output0, AI_STATIC,
  290, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 34, 1), AI_STRIDE_INIT(4, 4, 4, 24, 816),
  1, &transpose_97_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_168_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_168_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_168_nl_layer, 168,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_168_nl_chain,
  NULL, &nl_168_nl_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_167_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_164_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_167_weights, &gemm_167_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_167_layer, 168,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_167_chain,
  NULL, &nl_168_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_164_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_163_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_164_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_164_scale, &eltwise_164_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_164_layer, 165,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_164_chain,
  NULL, &gemm_167_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_163_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_161_output, &eltwise_159_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_163_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_163_layer, 163,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_163_chain,
  NULL, &eltwise_164_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_159_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant66_2D, &eltwise_158_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_159_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_159_layer, 159,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_159_chain,
  NULL, &eltwise_163_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_158_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_157_output, &reduce_152_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_158_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_158_layer, 158,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_158_chain,
  NULL, &eltwise_159_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_161_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_150_output, &nl_157_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_161_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_161_layer, 161,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_161_chain,
  NULL, &eltwise_158_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_157_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_155_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_157_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_157_layer, 157,
  NL_TYPE, 0x0, NULL,
  nl, forward_rsqrt,
  &nl_157_chain,
  NULL, &eltwise_161_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_155_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_155_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_155_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &reduce_66_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_155_Mul_layer, 156,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_155_Mul_chain,
  NULL, &nl_157_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_155_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_155_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_155_neutral_value_data, reduce_155_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_155_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_154_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_155_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_155_layer, 156,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_155_chain,
  NULL, &reduce_155_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_155_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_154_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_150_output, &reduce_152_Mul_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_154_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_154_layer, 154,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_154_chain,
  NULL, &reduce_155_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_152_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_152_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_152_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_152_Mul_layer, 152,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_152_Mul_chain,
  NULL, &eltwise_154_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_152_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_152_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_152_neutral_value_data, reduce_152_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_152_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_150_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_152_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_152_layer, 152,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_152_chain,
  NULL, &reduce_152_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_152_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_150_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_149_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_150_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_150_scale, &eltwise_150_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_150_layer, 151,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_150_chain,
  NULL, &reduce_152_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_149_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_147_output, &eltwise_145_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_149_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_149_layer, 149,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_149_chain,
  NULL, &eltwise_150_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_145_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant66_2D, &eltwise_144_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_145_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_145_layer, 145,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_145_chain,
  NULL, &eltwise_149_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_144_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_143_output, &reduce_138_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_144_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_144_layer, 144,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_144_chain,
  NULL, &eltwise_145_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_147_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_137_output, &nl_143_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_147_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_147_layer, 147,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_147_chain,
  NULL, &eltwise_144_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_143_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_141_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_143_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_143_layer, 143,
  NL_TYPE, 0x0, NULL,
  nl, forward_rsqrt,
  &nl_143_chain,
  NULL, &eltwise_147_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_141_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_141_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_141_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &reduce_66_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_141_Mul_layer, 142,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_141_Mul_chain,
  NULL, &nl_143_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_141_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_141_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_141_neutral_value_data, reduce_141_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_140_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_141_layer, 142,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_141_chain,
  NULL, &reduce_141_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_141_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_140_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_137_output, &reduce_138_Mul_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_140_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_140_layer, 140,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_140_chain,
  NULL, &reduce_141_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_138_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_138_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_138_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_138_Mul_layer, 138,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_138_Mul_chain,
  NULL, &eltwise_140_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_138_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_138_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_138_neutral_value_data, reduce_138_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_138_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_138_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_138_layer, 138,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_138_chain,
  NULL, &reduce_138_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_138_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_137_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_132_output, &gemm_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_137_layer, 137,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_137_chain,
  NULL, &reduce_138_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_136_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_135_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_136_weights, &gemm_136_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_136_layer, 136,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_136_chain,
  NULL, &eltwise_137_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_135_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_135_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_135_nl_layer, 135,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_135_nl_chain,
  NULL, &gemm_136_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_134_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_134_weights, &gemm_134_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_134_layer, 135,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_134_chain,
  NULL, &nl_135_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_132_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_131_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_132_scale, &eltwise_132_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_132_layer, 133,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_132_chain,
  NULL, &gemm_134_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_131_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_129_output, &eltwise_127_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_131_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_131_layer, 131,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_131_chain,
  NULL, &eltwise_132_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_127_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant66_2D, &eltwise_126_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_127_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_127_layer, 127,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_127_chain,
  NULL, &eltwise_131_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_126_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_125_output, &reduce_120_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_126_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_126_layer, 126,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_126_chain,
  NULL, &eltwise_127_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_129_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_119_output, &nl_125_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_129_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_129_layer, 129,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_129_chain,
  NULL, &eltwise_126_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_125_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_123_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_125_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_125_layer, 125,
  NL_TYPE, 0x0, NULL,
  nl, forward_rsqrt,
  &nl_125_chain,
  NULL, &eltwise_129_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_123_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_123_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_123_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &reduce_66_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_123_Mul_layer, 124,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_123_Mul_chain,
  NULL, &nl_125_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_123_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_123_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_123_neutral_value_data, reduce_123_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_123_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_122_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_123_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_123_layer, 124,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_123_chain,
  NULL, &reduce_123_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_123_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_122_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_119_output, &reduce_120_Mul_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_122_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_122_layer, 122,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_122_chain,
  NULL, &reduce_123_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_120_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_120_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_120_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_120_Mul_layer, 120,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_120_Mul_chain,
  NULL, &eltwise_122_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_120_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_120_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_120_neutral_value_data, reduce_120_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_120_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_119_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_120_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_120_layer, 120,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_120_chain,
  NULL, &reduce_120_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_120_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_93_output, &gemm_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_119_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_119_layer, 119,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_119_chain,
  NULL, &reduce_120_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_118_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_118_weights, &gemm_118_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_118_layer, 118,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_118_chain,
  NULL, &eltwise_119_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_117_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_105_output0, &gemm_116_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_117_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_117_layer, 117,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_117_chain,
  NULL, &gemm_118_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_116_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_113_output0, &gemm_114_output0, &gemm_116_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_116_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_116_layer, 116,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_116_chain,
  NULL, &concat_117_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_114_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_114_layer, 114,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_114_chain,
  NULL, &gemm_116_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_113_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_113_layer, 113,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_113_chain,
  NULL, &gemm_114_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_111_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_43_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_111_layer, 111,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_111_chain,
  NULL, &nl_113_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_110_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_107_output0, &transpose_108_output0, &gemm_110_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_110_layer, 110,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_110_chain,
  NULL, &eltwise_111_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_107_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_107_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_107_layer, 107,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_107_chain,
  NULL, &gemm_110_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_108_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_106_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_108_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_108_layer, 108,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_108_chain,
  NULL, &gemm_107_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_106_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_106_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_106_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_106_layer, 106,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_106_chain,
  NULL, &transpose_108_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_102_output0, &gemm_103_output0, &gemm_105_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_105_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_105_layer, 105,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_105_chain,
  NULL, &gemm_106_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_103_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_103_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_103_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_103_layer, 103,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_103_chain,
  NULL, &gemm_105_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_102_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_102_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_102_layer, 102,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_102_chain,
  NULL, &gemm_103_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_100_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_99_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_43_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_100_layer, 100,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_100_chain,
  NULL, &nl_102_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_99_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_96_output0, &transpose_97_output0, &gemm_99_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_99_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_99_layer, 99,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_99_chain,
  NULL, &eltwise_100_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_96_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_96_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_96_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_96_layer, 96,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_96_chain,
  NULL, &gemm_99_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_97_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_95_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_97_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_97_layer, 97,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_97_chain,
  NULL, &gemm_96_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_95_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_95_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_95_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_95_layer, 95,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_95_chain,
  NULL, &transpose_97_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_93_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_92_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_93_scale, &eltwise_93_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_93_layer, 94,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_93_chain,
  NULL, &gemm_95_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_92_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_90_output, &eltwise_88_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_92_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_92_layer, 92,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_92_chain,
  NULL, &eltwise_93_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant66_2D, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_88_layer, 88,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_88_chain,
  NULL, &eltwise_92_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_86_output, &reduce_81_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_87_layer, 87,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_87_chain,
  NULL, &eltwise_88_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_90_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_80_output, &nl_86_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_90_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_90_layer, 90,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_90_chain,
  NULL, &eltwise_87_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_84_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_86_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, forward_rsqrt,
  &nl_86_chain,
  NULL, &eltwise_90_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_84_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_84_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_84_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &reduce_66_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_84_Mul_layer, 85,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_84_Mul_chain,
  NULL, &nl_86_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_84_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_84_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_84_neutral_value_data, reduce_84_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_84_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_84_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_84_layer, 85,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_84_chain,
  NULL, &reduce_84_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_84_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_80_output, &reduce_81_Mul_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_83_layer, 83,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_83_chain,
  NULL, &reduce_84_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_81_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_81_Mul_layer, 81,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_81_Mul_chain,
  NULL, &eltwise_83_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_81_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_81_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_81_neutral_value_data, reduce_81_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_81_layer, 81,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_81_chain,
  NULL, &reduce_81_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_81_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_75_output, &gemm_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_80_layer, 80,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_80_chain,
  NULL, &reduce_81_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_78_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_79_weights, &gemm_79_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_79_layer, 79,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_79_chain,
  NULL, &eltwise_80_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_78_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_77_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_78_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_78_nl_layer, 78,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_78_nl_chain,
  NULL, &gemm_79_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_77_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_77_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_77_weights, &gemm_77_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_77_layer, 78,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_77_chain,
  NULL, &nl_78_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_74_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_75_scale, &eltwise_75_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_75_layer, 76,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_75_chain,
  NULL, &gemm_77_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_72_output, &eltwise_70_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_74_layer, 74,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_74_chain,
  NULL, &eltwise_75_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant66_2D, &eltwise_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_70_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_70_layer, 70,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_70_chain,
  NULL, &eltwise_74_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_68_output, &reduce_63_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_69_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_69_layer, 69,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_69_chain,
  NULL, &eltwise_70_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_72_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_62_output, &nl_68_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_72_layer, 72,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_72_chain,
  NULL, &eltwise_69_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_68_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_66_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_68_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_68_layer, 68,
  NL_TYPE, 0x0, NULL,
  nl, forward_rsqrt,
  &nl_68_chain,
  NULL, &eltwise_72_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_66_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_66_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_66_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &reduce_66_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_66_Mul_layer, 67,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_66_Mul_chain,
  NULL, &nl_68_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_66_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_66_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_66_neutral_value_data, reduce_66_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_65_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_66_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_66_layer, 67,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_66_chain,
  NULL, &reduce_66_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_66_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_62_output, &reduce_63_Mul_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_65_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_65_layer, 65,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_65_chain,
  NULL, &reduce_66_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_63_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_63_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_63_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_63_Mul_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_63_Mul_layer, 63,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_63_Mul_chain,
  NULL, &eltwise_65_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_63_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_63_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_63_neutral_value_data, reduce_63_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_62_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_63_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_63_layer, 63,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_63_chain,
  NULL, &reduce_63_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_63_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_37_output, &gemm_61_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_62_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_62_layer, 62,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_62_chain,
  NULL, &reduce_63_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_61_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_61_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_61_weights, &gemm_61_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_61_layer, 61,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_61_chain,
  NULL, &eltwise_62_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_48_output0, &gemm_59_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_60_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_60_layer, 60,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_60_chain,
  NULL, &gemm_61_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_56_output0, &gemm_57_output0, &gemm_59_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_59_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_59_layer, 59,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_59_chain,
  NULL, &concat_60_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_57_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_57_layer, 57,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_57_chain,
  NULL, &gemm_59_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_56_layer, 56,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_56_chain,
  NULL, &gemm_57_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_53_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_43_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_54_chain,
  NULL, &nl_56_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_50_output0, &transpose_51_output0, &gemm_53_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_53_layer, 53,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_53_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_50_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_50_layer, 50,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_50_chain,
  NULL, &gemm_53_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_51_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_51_layer, 51,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_51_chain,
  NULL, &gemm_50_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_49_layer, 49,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_49_chain,
  NULL, &transpose_51_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_45_output0, &gemm_46_output0, &gemm_48_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_48_layer, 48,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_48_chain,
  NULL, &gemm_49_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_46_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_46_chain,
  NULL, &gemm_48_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_45_layer, 45,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_45_chain,
  NULL, &gemm_46_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_43_scale, &eltwise_43_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_43_layer, 43,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_43_chain,
  NULL, &nl_45_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_39_output0, &transpose_40_output0, &gemm_42_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_42_layer, 42,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_42_chain,
  NULL, &eltwise_43_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_39_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_39_layer, 39,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_39_chain,
  NULL, &gemm_42_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_38_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_40_layer, 40,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_40_chain,
  NULL, &gemm_39_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_38_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_38_layer, 38,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_38_chain,
  NULL, &transpose_40_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 6, &eltwise_10_output0, &eltwise_15_output0, &eltwise_20_output0, &eltwise_25_output0, &eltwise_30_output0, &eltwise_35_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_37_layer, 37,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_37_chain,
  NULL, &gemm_38_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_9_output0, &arith_constant39),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_layer, 10,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_10_chain,
  NULL, &concat_37_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_7_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_9_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_9_layer, 9,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_9_chain,
  NULL, &eltwise_10_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_7_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_7_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_7_nl_layer, 7,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_7_nl_chain,
  NULL, &gemm_9_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_7_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_7_chain,
  NULL, &nl_7_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_u8 slice_1_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_axes, AI_ARRAY_FORMAT_U8,
    slice_1_axes_data, slice_1_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_starts_data[] = { 0, 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_starts, AI_ARRAY_FORMAT_S16,
    slice_1_starts_data, slice_1_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_ends_data[] = { 38, 5, 1 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_ends, AI_ARRAY_FORMAT_S16,
    slice_1_ends_data, slice_1_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_1_layer, 1,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_1_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .axes = &slice_1_axes, 
  .starts = &slice_1_starts, 
  .ends = &slice_1_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_output0, &arith_constant37),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_15_layer, 15,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_15_chain,
  NULL, &slice_1_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_14_chain,
  NULL, &eltwise_15_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_12_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_12_nl_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_12_nl_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_12_weights, &conv2d_12_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_12_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_12_chain,
  NULL, &nl_12_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_u8 slice_2_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_axes, AI_ARRAY_FORMAT_U8,
    slice_2_axes_data, slice_2_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_2_starts_data[] = { 0, 0, 1 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_starts, AI_ARRAY_FORMAT_S16,
    slice_2_starts_data, slice_2_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_2_ends_data[] = { 38, 5, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_ends, AI_ARRAY_FORMAT_S16,
    slice_2_ends_data, slice_2_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_2_layer, 2,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_2_chain,
  NULL, &conv2d_12_layer, AI_STATIC, 
  .axes = &slice_2_axes, 
  .starts = &slice_2_starts, 
  .ends = &slice_2_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_19_output0, &arith_constant35),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_20_layer, 20,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_20_chain,
  NULL, &slice_2_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_19_layer, 19,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_19_chain,
  NULL, &eltwise_20_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_17_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_17_nl_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_17_nl_chain,
  NULL, &gemm_19_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_17_weights, &conv2d_17_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_17_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_17_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_17_chain,
  NULL, &nl_17_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_u8 slice_3_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_axes, AI_ARRAY_FORMAT_U8,
    slice_3_axes_data, slice_3_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_starts_data[] = { 0, 0, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_starts, AI_ARRAY_FORMAT_S16,
    slice_3_starts_data, slice_3_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_ends_data[] = { 38, 5, 3 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_ends, AI_ARRAY_FORMAT_S16,
    slice_3_ends_data, slice_3_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_3_layer, 3,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_3_chain,
  NULL, &conv2d_17_layer, AI_STATIC, 
  .axes = &slice_3_axes, 
  .starts = &slice_3_starts, 
  .ends = &slice_3_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_24_output0, &arith_constant33),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_25_layer, 25,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_25_chain,
  NULL, &slice_3_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_24_layer, 24,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_24_chain,
  NULL, &eltwise_25_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_22_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_22_nl_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_22_nl_chain,
  NULL, &gemm_24_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_22_weights, &conv2d_22_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_22_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_22_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_22_chain,
  NULL, &nl_22_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_u8 slice_4_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_4_axes, AI_ARRAY_FORMAT_U8,
    slice_4_axes_data, slice_4_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_4_starts_data[] = { 0, 0, 3 };
AI_ARRAY_OBJ_DECLARE(
    slice_4_starts, AI_ARRAY_FORMAT_S16,
    slice_4_starts_data, slice_4_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_4_ends_data[] = { 38, 5, 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_4_ends, AI_ARRAY_FORMAT_S16,
    slice_4_ends_data, slice_4_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_4_layer, 4,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_4_chain,
  NULL, &conv2d_22_layer, AI_STATIC, 
  .axes = &slice_4_axes, 
  .starts = &slice_4_starts, 
  .ends = &slice_4_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_29_output0, &arith_constant31),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_30_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_30_layer, 30,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_30_chain,
  NULL, &slice_4_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_27_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_29_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_29_layer, 29,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_29_chain,
  NULL, &eltwise_30_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_27_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_27_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_27_nl_layer, 27,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_27_nl_chain,
  NULL, &gemm_29_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_27_weights, &conv2d_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_27_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_27_layer, 27,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_27_chain,
  NULL, &nl_27_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_u8 slice_5_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_axes, AI_ARRAY_FORMAT_U8,
    slice_5_axes_data, slice_5_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_starts_data[] = { 0, 0, 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_starts, AI_ARRAY_FORMAT_S16,
    slice_5_starts_data, slice_5_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_ends_data[] = { 38, 5, 5 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_ends, AI_ARRAY_FORMAT_S16,
    slice_5_ends_data, slice_5_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_5_layer, 5,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_5_chain,
  NULL, &conv2d_27_layer, AI_STATIC, 
  .axes = &slice_5_axes, 
  .starts = &slice_5_starts, 
  .ends = &slice_5_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_34_output0, &arith_constant29),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_35_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_35_layer, 35,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_35_chain,
  NULL, &slice_5_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_32_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_34_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_34_layer, 34,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_34_chain,
  NULL, &eltwise_35_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_32_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_32_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_32_nl_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_32_nl_chain,
  NULL, &gemm_34_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_32_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_32_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_32_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_32_chain,
  NULL, &nl_32_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_u8 slice_6_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_axes, AI_ARRAY_FORMAT_U8,
    slice_6_axes_data, slice_6_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_starts_data[] = { 0, 0, 5 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_starts, AI_ARRAY_FORMAT_S16,
    slice_6_starts_data, slice_6_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_ends_data[] = { 38, 5, 6 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_ends, AI_ARRAY_FORMAT_S16,
    slice_6_ends_data, slice_6_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_6_layer, 6,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_6_chain,
  NULL, &conv2d_32_layer, AI_STATIC, 
  .axes = &slice_6_axes, 
  .starts = &slice_6_starts, 
  .ends = &slice_6_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  serving_default_args_00_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_args_00_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  serving_default_args_00_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &serving_default_args_00_Transpose_chain,
  NULL, &slice_6_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 201340, 1, 1),
    201340, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1565600, 1, 1),
    1565600, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_args_00_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_168_nl_output),
  &serving_default_args_00_Transpose_layer, 0x73354d87, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 201340, 1, 1),
      201340, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1565600, 1, 1),
      1565600, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_args_00_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_168_nl_output),
  &serving_default_args_00_Transpose_layer, 0x73354d87, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_args_00_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    serving_default_args_00_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    serving_default_args_00_Transpose_output_array.data = AI_PTR(g_network_activations_map[0] + 782800);
    serving_default_args_00_Transpose_output_array.data_start = AI_PTR(g_network_activations_map[0] + 782800);
    slice_6_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    slice_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_32_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 760);
    conv2d_32_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 760);
    conv2d_32_output_array.data = AI_PTR(g_network_activations_map[0] + 860);
    conv2d_32_output_array.data_start = AI_PTR(g_network_activations_map[0] + 860);
    nl_32_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_32_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_34_output_array.data = AI_PTR(g_network_activations_map[0] + 68);
    gemm_34_output_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    eltwise_35_output_array.data = AI_PTR(g_network_activations_map[0] + 340);
    eltwise_35_output_array.data_start = AI_PTR(g_network_activations_map[0] + 340);
    slice_5_output_array.data = AI_PTR(g_network_activations_map[0] + 612);
    slice_5_output_array.data_start = AI_PTR(g_network_activations_map[0] + 612);
    conv2d_27_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_27_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_27_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    conv2d_27_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    nl_27_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_27_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_29_output_array.data = AI_PTR(g_network_activations_map[0] + 68);
    gemm_29_output_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    eltwise_30_output_array.data = AI_PTR(g_network_activations_map[0] + 612);
    eltwise_30_output_array.data_start = AI_PTR(g_network_activations_map[0] + 612);
    slice_4_output_array.data = AI_PTR(g_network_activations_map[0] + 884);
    slice_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 884);
    conv2d_22_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_22_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_22_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    conv2d_22_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    nl_22_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_22_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_24_output_array.data = AI_PTR(g_network_activations_map[0] + 68);
    gemm_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    eltwise_25_output_array.data = AI_PTR(g_network_activations_map[0] + 884);
    eltwise_25_output_array.data_start = AI_PTR(g_network_activations_map[0] + 884);
    slice_3_output_array.data = AI_PTR(g_network_activations_map[0] + 1156);
    slice_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1156);
    conv2d_17_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_17_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_17_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    conv2d_17_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    nl_17_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_17_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_19_output_array.data = AI_PTR(g_network_activations_map[0] + 68);
    gemm_19_output_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    eltwise_20_output_array.data = AI_PTR(g_network_activations_map[0] + 1156);
    eltwise_20_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1156);
    slice_2_output_array.data = AI_PTR(g_network_activations_map[0] + 1428);
    slice_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1428);
    conv2d_12_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_12_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_12_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    conv2d_12_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    nl_12_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_12_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_14_output_array.data = AI_PTR(g_network_activations_map[0] + 68);
    gemm_14_output_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    eltwise_15_output_array.data = AI_PTR(g_network_activations_map[0] + 1428);
    eltwise_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1428);
    slice_1_output_array.data = AI_PTR(g_network_activations_map[0] + 1700);
    slice_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1700);
    conv2d_7_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_7_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    conv2d_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    nl_7_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_7_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_9_output_array.data = AI_PTR(g_network_activations_map[0] + 68);
    gemm_9_output_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    eltwise_10_output_array.data = AI_PTR(g_network_activations_map[0] + 1700);
    eltwise_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1700);
    concat_37_output_array.data = AI_PTR(g_network_activations_map[0] + 1972);
    concat_37_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1972);
    gemm_38_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_38_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_40_output_array.data = AI_PTR(g_network_activations_map[0] + 816);
    transpose_40_output_array.data_start = AI_PTR(g_network_activations_map[0] + 816);
    gemm_39_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_39_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_42_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    gemm_42_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    eltwise_43_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_43_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_45_output_array.data = AI_PTR(g_network_activations_map[0] + 144);
    nl_45_output_array.data_start = AI_PTR(g_network_activations_map[0] + 144);
    gemm_46_output_array.data = AI_PTR(g_network_activations_map[0] + 288);
    gemm_46_output_array.data_start = AI_PTR(g_network_activations_map[0] + 288);
    gemm_48_output_array.data = AI_PTR(g_network_activations_map[0] + 1104);
    gemm_48_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1104);
    gemm_49_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_49_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_51_output_array.data = AI_PTR(g_network_activations_map[0] + 3604);
    transpose_51_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3604);
    gemm_50_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_50_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_53_output_array.data = AI_PTR(g_network_activations_map[0] + 816);
    gemm_53_output_array.data_start = AI_PTR(g_network_activations_map[0] + 816);
    eltwise_54_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_54_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_56_output_array.data = AI_PTR(g_network_activations_map[0] + 144);
    nl_56_output_array.data_start = AI_PTR(g_network_activations_map[0] + 144);
    gemm_57_output_array.data = AI_PTR(g_network_activations_map[0] + 288);
    gemm_57_output_array.data_start = AI_PTR(g_network_activations_map[0] + 288);
    gemm_59_output_array.data = AI_PTR(g_network_activations_map[0] + 3604);
    gemm_59_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3604);
    concat_60_output_array.data = AI_PTR(g_network_activations_map[0] + 4420);
    concat_60_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4420);
    gemm_61_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_61_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_62_output_array.data = AI_PTR(g_network_activations_map[0] + 3604);
    eltwise_62_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3604);
    reduce_63_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_63_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_63_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    reduce_63_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_65_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_65_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    reduce_66_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_66_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_66_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    reduce_66_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    nl_68_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_68_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_72_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_72_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_69_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_69_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_70_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_70_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_74_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_74_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_75_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_75_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_77_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    gemm_77_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    nl_78_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 2448);
    nl_78_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2448);
    gemm_79_output_array.data = AI_PTR(g_network_activations_map[0] + 3264);
    gemm_79_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3264);
    eltwise_80_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    eltwise_80_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_81_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_81_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_81_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    reduce_81_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_83_output_array.data = AI_PTR(g_network_activations_map[0] + 3264);
    eltwise_83_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3264);
    reduce_84_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_84_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_84_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    reduce_84_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    nl_86_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_86_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_90_output_array.data = AI_PTR(g_network_activations_map[0] + 3264);
    eltwise_90_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3264);
    eltwise_87_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_87_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_88_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_88_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_92_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_92_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_93_output_array.data = AI_PTR(g_network_activations_map[0] + 1656);
    eltwise_93_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1656);
    gemm_95_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_95_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_97_output_array.data = AI_PTR(g_network_activations_map[0] + 816);
    transpose_97_output_array.data_start = AI_PTR(g_network_activations_map[0] + 816);
    gemm_96_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_96_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_99_output_array.data = AI_PTR(g_network_activations_map[0] + 3288);
    gemm_99_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3288);
    eltwise_100_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_100_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_102_output_array.data = AI_PTR(g_network_activations_map[0] + 144);
    nl_102_output_array.data_start = AI_PTR(g_network_activations_map[0] + 144);
    gemm_103_output_array.data = AI_PTR(g_network_activations_map[0] + 288);
    gemm_103_output_array.data_start = AI_PTR(g_network_activations_map[0] + 288);
    gemm_105_output_array.data = AI_PTR(g_network_activations_map[0] + 3288);
    gemm_105_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3288);
    gemm_106_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_106_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_108_output_array.data = AI_PTR(g_network_activations_map[0] + 816);
    transpose_108_output_array.data_start = AI_PTR(g_network_activations_map[0] + 816);
    gemm_107_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_107_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_110_output_array.data = AI_PTR(g_network_activations_map[0] + 4104);
    gemm_110_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4104);
    eltwise_111_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_111_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_113_output_array.data = AI_PTR(g_network_activations_map[0] + 144);
    nl_113_output_array.data_start = AI_PTR(g_network_activations_map[0] + 144);
    gemm_114_output_array.data = AI_PTR(g_network_activations_map[0] + 288);
    gemm_114_output_array.data_start = AI_PTR(g_network_activations_map[0] + 288);
    gemm_116_output_array.data = AI_PTR(g_network_activations_map[0] + 4104);
    gemm_116_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4104);
    concat_117_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    concat_117_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_118_output_array.data = AI_PTR(g_network_activations_map[0] + 3288);
    gemm_118_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3288);
    eltwise_119_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_119_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_120_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_120_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_120_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 1656);
    reduce_120_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1656);
    eltwise_122_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_122_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    reduce_123_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_123_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_123_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    reduce_123_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    nl_125_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    nl_125_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    eltwise_129_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_129_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_126_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_126_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_127_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_127_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_131_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_131_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_132_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_132_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    gemm_134_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_134_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_135_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 816);
    nl_135_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 816);
    gemm_136_output_array.data = AI_PTR(g_network_activations_map[0] + 3312);
    gemm_136_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3312);
    eltwise_137_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_137_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_138_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_138_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_138_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 1656);
    reduce_138_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1656);
    eltwise_140_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_140_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    reduce_141_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_141_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    reduce_141_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    reduce_141_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    nl_143_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    nl_143_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    eltwise_147_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_147_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_144_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_144_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_145_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_145_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_149_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_149_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_150_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_150_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    reduce_152_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_152_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_152_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    reduce_152_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    eltwise_154_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_154_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    reduce_155_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_155_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_155_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    reduce_155_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    nl_157_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_157_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_161_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_161_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    eltwise_158_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_158_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_159_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_159_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_163_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_163_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    eltwise_164_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_164_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_167_output_array.data = AI_PTR(g_network_activations_map[0] + 1632);
    gemm_167_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1632);
    nl_168_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_168_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    gemm_116_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_116_bias_array.data = AI_PTR(g_network_weights_map[0] + 0);
    gemm_116_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    gemm_110_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_110_bias_array.data = AI_PTR(g_network_weights_map[0] + 4);
    gemm_110_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    gemm_105_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_105_bias_array.data = AI_PTR(g_network_weights_map[0] + 8);
    gemm_105_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 8);
    gemm_99_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_99_bias_array.data = AI_PTR(g_network_weights_map[0] + 12);
    gemm_99_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12);
    gemm_59_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_59_bias_array.data = AI_PTR(g_network_weights_map[0] + 16);
    gemm_59_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16);
    gemm_53_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_53_bias_array.data = AI_PTR(g_network_weights_map[0] + 20);
    gemm_53_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 20);
    gemm_48_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_48_bias_array.data = AI_PTR(g_network_weights_map[0] + 24);
    gemm_48_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 24);
    gemm_42_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_42_bias_array.data = AI_PTR(g_network_weights_map[0] + 28);
    gemm_42_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 28);
    arith_constant66_2D_array.format |= AI_FMT_FLAG_CONST;
    arith_constant66_2D_array.data = AI_PTR(g_network_weights_map[0] + 32);
    arith_constant66_2D_array.data_start = AI_PTR(g_network_weights_map[0] + 32);
    arith_constant29_array.format |= AI_FMT_FLAG_CONST;
    arith_constant29_array.data = AI_PTR(g_network_weights_map[0] + 36);
    arith_constant29_array.data_start = AI_PTR(g_network_weights_map[0] + 36);
    arith_constant31_array.format |= AI_FMT_FLAG_CONST;
    arith_constant31_array.data = AI_PTR(g_network_weights_map[0] + 308);
    arith_constant31_array.data_start = AI_PTR(g_network_weights_map[0] + 308);
    arith_constant33_array.format |= AI_FMT_FLAG_CONST;
    arith_constant33_array.data = AI_PTR(g_network_weights_map[0] + 580);
    arith_constant33_array.data_start = AI_PTR(g_network_weights_map[0] + 580);
    arith_constant35_array.format |= AI_FMT_FLAG_CONST;
    arith_constant35_array.data = AI_PTR(g_network_weights_map[0] + 852);
    arith_constant35_array.data_start = AI_PTR(g_network_weights_map[0] + 852);
    arith_constant37_array.format |= AI_FMT_FLAG_CONST;
    arith_constant37_array.data = AI_PTR(g_network_weights_map[0] + 1124);
    arith_constant37_array.data_start = AI_PTR(g_network_weights_map[0] + 1124);
    arith_constant39_array.format |= AI_FMT_FLAG_CONST;
    arith_constant39_array.data = AI_PTR(g_network_weights_map[0] + 1396);
    arith_constant39_array.data_start = AI_PTR(g_network_weights_map[0] + 1396);
    conv2d_32_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_32_weights_array.data = AI_PTR(g_network_weights_map[0] + 1668);
    conv2d_32_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1668);
    conv2d_32_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_32_bias_array.data = AI_PTR(g_network_weights_map[0] + 1768);
    conv2d_32_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1768);
    gemm_34_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_34_weights_array.data = AI_PTR(g_network_weights_map[0] + 1772);
    gemm_34_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1772);
    conv2d_27_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_27_weights_array.data = AI_PTR(g_network_weights_map[0] + 1788);
    conv2d_27_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1788);
    conv2d_27_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_27_bias_array.data = AI_PTR(g_network_weights_map[0] + 1888);
    conv2d_27_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1888);
    gemm_29_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_29_weights_array.data = AI_PTR(g_network_weights_map[0] + 1892);
    gemm_29_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1892);
    conv2d_22_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_weights_array.data = AI_PTR(g_network_weights_map[0] + 1908);
    conv2d_22_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1908);
    conv2d_22_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_bias_array.data = AI_PTR(g_network_weights_map[0] + 2008);
    conv2d_22_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2008);
    gemm_24_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_24_weights_array.data = AI_PTR(g_network_weights_map[0] + 2012);
    gemm_24_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2012);
    conv2d_17_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_weights_array.data = AI_PTR(g_network_weights_map[0] + 2028);
    conv2d_17_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2028);
    conv2d_17_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_bias_array.data = AI_PTR(g_network_weights_map[0] + 2128);
    conv2d_17_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2128);
    gemm_19_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_19_weights_array.data = AI_PTR(g_network_weights_map[0] + 2132);
    gemm_19_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2132);
    conv2d_12_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_weights_array.data = AI_PTR(g_network_weights_map[0] + 2148);
    conv2d_12_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2148);
    conv2d_12_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_bias_array.data = AI_PTR(g_network_weights_map[0] + 2248);
    conv2d_12_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2248);
    gemm_14_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_weights_array.data = AI_PTR(g_network_weights_map[0] + 2252);
    gemm_14_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2252);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_network_weights_map[0] + 2268);
    conv2d_7_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2268);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_network_weights_map[0] + 2368);
    conv2d_7_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2368);
    gemm_9_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_9_weights_array.data = AI_PTR(g_network_weights_map[0] + 2372);
    gemm_9_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2372);
    gemm_38_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_38_weights_array.data = AI_PTR(g_network_weights_map[0] + 2388);
    gemm_38_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2388);
    gemm_39_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_39_weights_array.data = AI_PTR(g_network_weights_map[0] + 11636);
    gemm_39_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11636);
    eltwise_43_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_43_scale_array.data = AI_PTR(g_network_weights_map[0] + 20884);
    eltwise_43_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 20884);
    eltwise_43_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_43_bias_array.data = AI_PTR(g_network_weights_map[0] + 20908);
    eltwise_43_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 20908);
    gemm_46_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_46_weights_array.data = AI_PTR(g_network_weights_map[0] + 20932);
    gemm_46_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 20932);
    gemm_49_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_49_weights_array.data = AI_PTR(g_network_weights_map[0] + 30180);
    gemm_49_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 30180);
    gemm_50_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_50_weights_array.data = AI_PTR(g_network_weights_map[0] + 39428);
    gemm_50_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 39428);
    gemm_57_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_57_weights_array.data = AI_PTR(g_network_weights_map[0] + 48676);
    gemm_57_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 48676);
    gemm_61_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_61_weights_array.data = AI_PTR(g_network_weights_map[0] + 57924);
    gemm_61_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 57924);
    gemm_61_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_61_bias_array.data = AI_PTR(g_network_weights_map[0] + 76420);
    gemm_61_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 76420);
    reduce_63_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    reduce_63_Mul_scale_array.data = AI_PTR(g_network_weights_map[0] + 76692);
    reduce_63_Mul_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 76692);
    reduce_66_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    reduce_66_Mul_bias_array.data = AI_PTR(g_network_weights_map[0] + 76716);
    reduce_66_Mul_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 76716);
    eltwise_75_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_75_scale_array.data = AI_PTR(g_network_weights_map[0] + 76740);
    eltwise_75_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 76740);
    eltwise_75_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_75_bias_array.data = AI_PTR(g_network_weights_map[0] + 77012);
    eltwise_75_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 77012);
    gemm_77_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_77_weights_array.data = AI_PTR(g_network_weights_map[0] + 77284);
    gemm_77_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 77284);
    gemm_77_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_77_bias_array.data = AI_PTR(g_network_weights_map[0] + 86532);
    gemm_77_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 86532);
    gemm_79_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_79_weights_array.data = AI_PTR(g_network_weights_map[0] + 86668);
    gemm_79_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 86668);
    gemm_79_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_79_bias_array.data = AI_PTR(g_network_weights_map[0] + 95916);
    gemm_79_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 95916);
    eltwise_93_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_93_scale_array.data = AI_PTR(g_network_weights_map[0] + 96188);
    eltwise_93_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 96188);
    eltwise_93_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_93_bias_array.data = AI_PTR(g_network_weights_map[0] + 96460);
    eltwise_93_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 96460);
    gemm_95_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_95_weights_array.data = AI_PTR(g_network_weights_map[0] + 96732);
    gemm_95_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 96732);
    gemm_96_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_96_weights_array.data = AI_PTR(g_network_weights_map[0] + 105980);
    gemm_96_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 105980);
    gemm_103_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_103_weights_array.data = AI_PTR(g_network_weights_map[0] + 115228);
    gemm_103_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 115228);
    gemm_106_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_106_weights_array.data = AI_PTR(g_network_weights_map[0] + 124476);
    gemm_106_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 124476);
    gemm_107_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_107_weights_array.data = AI_PTR(g_network_weights_map[0] + 133724);
    gemm_107_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 133724);
    gemm_114_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_114_weights_array.data = AI_PTR(g_network_weights_map[0] + 142972);
    gemm_114_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 142972);
    gemm_118_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_118_weights_array.data = AI_PTR(g_network_weights_map[0] + 152220);
    gemm_118_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 152220);
    gemm_118_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_118_bias_array.data = AI_PTR(g_network_weights_map[0] + 170716);
    gemm_118_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 170716);
    eltwise_132_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_132_scale_array.data = AI_PTR(g_network_weights_map[0] + 170988);
    eltwise_132_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 170988);
    eltwise_132_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_132_bias_array.data = AI_PTR(g_network_weights_map[0] + 171260);
    eltwise_132_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 171260);
    gemm_134_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_134_weights_array.data = AI_PTR(g_network_weights_map[0] + 171532);
    gemm_134_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 171532);
    gemm_134_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_134_bias_array.data = AI_PTR(g_network_weights_map[0] + 180780);
    gemm_134_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 180780);
    gemm_136_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_136_weights_array.data = AI_PTR(g_network_weights_map[0] + 180916);
    gemm_136_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 180916);
    gemm_136_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_136_bias_array.data = AI_PTR(g_network_weights_map[0] + 190164);
    gemm_136_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 190164);
    eltwise_150_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_150_scale_array.data = AI_PTR(g_network_weights_map[0] + 190436);
    eltwise_150_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 190436);
    eltwise_150_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_150_bias_array.data = AI_PTR(g_network_weights_map[0] + 190708);
    eltwise_150_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 190708);
    eltwise_164_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_164_scale_array.data = AI_PTR(g_network_weights_map[0] + 190980);
    eltwise_164_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 190980);
    eltwise_164_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_164_bias_array.data = AI_PTR(g_network_weights_map[0] + 191252);
    eltwise_164_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 191252);
    gemm_167_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_167_weights_array.data = AI_PTR(g_network_weights_map[0] + 191524);
    gemm_167_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 191524);
    gemm_167_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_167_bias_array.data = AI_PTR(g_network_weights_map[0] + 201316);
    gemm_167_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 201316);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 470827,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x73354d87,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 470827,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x73354d87,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

