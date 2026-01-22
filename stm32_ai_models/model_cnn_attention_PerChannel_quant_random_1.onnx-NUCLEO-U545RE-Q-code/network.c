/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-22T15:46:33+0000
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0xb5397d57cd31df9565cc7ffea42b059f"
#define _STAI_NETWORK_DATETIME            "2026-01-22T15:46:33+0000"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(11, 0, 0),
  .tool_version = STAI_INIT_VERSION(3, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 30, 10),
      STAI_DECLARE_ARRAY(float, 1, 0.00392132019624114f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.000644468585960567f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 224008),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 600),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00392132019624114f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Unsqueeze_output_0_to_chfirst_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00392132019624114f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_output_0_gemm_to_dense_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004095243755728006f),
    AI_PACK_INTQ_ZP(-25)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_output_0_gemm_to_dense_out_transpose_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004095243755728006f),
    AI_PACK_INTQ_ZP(-25)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005318579729646444f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_6_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023670452646911144f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_a_MatMul_3_output_0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005318579729646444f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004052215255796909f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_1_output_0_gemm_to_dense_in_transpose_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004052215255796909f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_1_output_0_gemm_to_dense_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003841881174594164f),
    AI_PACK_INTQ_ZP(-41)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_1_output_0_gemm_to_dense_out_transpose_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003841881174594164f),
    AI_PACK_INTQ_ZP(-41)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004937407560646534f),
    AI_PACK_INTQ_ZP(-28)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_8_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0025396989658474922f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003893679240718484f),
    AI_PACK_INTQ_ZP(-90)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_out_MatMul_3_output_0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003893679240718484f),
    AI_PACK_INTQ_ZP(-90)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Div_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0011240083258599043f),
    AI_PACK_INTQ_ZP(-90)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_11_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02727638930082321f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_a_MatMul_4_output_0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_2_output_0_gemm_to_dense_in_transpose_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004052215255796909f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_2_output_0_gemm_to_dense_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0034422490280121565f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_2_output_0_gemm_to_dense_out_transpose_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0034422490280121565f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004716664552688599f),
    AI_PACK_INTQ_ZP(31)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_10_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002643187763169408f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_b_MatMul_4_output_0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004716664552688599f),
    AI_PACK_INTQ_ZP(31)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002826663199812174f),
    AI_PACK_INTQ_ZP(30)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_out_MatMul_4_output_0_out_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002826663199812174f),
    AI_PACK_INTQ_ZP(30)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 300, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 300, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_output_0_gemm_to_dense_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_output_0_gemm_to_dense_out_transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_6_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  transpose_a_MatMul_3_output_0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_in_transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_out_transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_8_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_0_1__MatMul_3_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_3_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 50176, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  transpose_out_MatMul_3_output_0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 50176, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Div_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 50176, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_11_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 50176, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  transpose_a_MatMul_4_output_0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 50176, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_in_transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_out_transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _Add_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_10_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  transpose_b_MatMul_4_output_0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_4_output_0_bias_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_4_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  transpose_out_MatMul_4_output_0_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 10, 30), AI_STRIDE_INIT(4, 1, 1, 1, 10),
  1, &_Unsqueeze_output_0_to_chfirst_output_array, &_Unsqueeze_output_0_to_chfirst_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output0, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 30, 1), AI_STRIDE_INIT(4, 1, 1, 10, 300),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_output_0_gemm_to_dense_out_transpose_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_MatMul_output_0_gemm_to_dense_out_transpose_output_array, &_MatMul_output_0_gemm_to_dense_out_transpose_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_output_0_gemm_to_dense_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 224), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &_MatMul_output_0_gemm_to_dense_output_array, &_MatMul_output_0_gemm_to_dense_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Add_output_0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_Add_output_0_output_array, &_Add_output_0_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_6_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 12), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Constant_6_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_6_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  transpose_a_MatMul_3_output_0_out_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 224), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &transpose_a_MatMul_3_output_0_out_output_array, &transpose_a_MatMul_3_output_0_out_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_in_transpose_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 224), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &_MatMul_1_output_0_gemm_to_dense_in_transpose_output_array, &_MatMul_1_output_0_gemm_to_dense_in_transpose_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output0, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 8), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_Relu_output_0_output_array, &_Relu_output_0_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_out_transpose_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_MatMul_1_output_0_gemm_to_dense_out_transpose_output_array, &_MatMul_1_output_0_gemm_to_dense_out_transpose_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 224), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &_MatMul_1_output_0_gemm_to_dense_output_array, &_MatMul_1_output_0_gemm_to_dense_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_Add_1_output_0_output_array, &_Add_1_output_0_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_8_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 12), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Constant_8_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_8_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_0_1__MatMul_3_output_0_conversion_output0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 224, 12, 1), AI_STRIDE_INIT(4, 4, 4, 896, 10752),
  1, &_Add_1_output_0_0_1__MatMul_3_output_0_conversion_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_3_output_0_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 224, 224, 1), AI_STRIDE_INIT(4, 4, 4, 896, 200704),
  1, &_MatMul_3_output_0_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 224, 1), AI_STRIDE_INIT(4, 4, 4, 48, 10752),
  1, &transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output0, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 224), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output_array, &_MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  transpose_out_MatMul_3_output_0_out_output, AI_STATIC,
  66, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 224), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &transpose_out_MatMul_3_output_0_out_output_array, &transpose_out_MatMul_3_output_0_out_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_11_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Constant_11_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_11_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _Div_output_0_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 224), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_Div_output_0_output_array, &_Div_output_0_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 224), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output_array, &_Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  transpose_a_MatMul_4_output_0_out_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 224), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &transpose_a_MatMul_4_output_0_out_output_array, &transpose_a_MatMul_4_output_0_out_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_in_transpose_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 224), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &_MatMul_2_output_0_gemm_to_dense_in_transpose_output_array, &_MatMul_2_output_0_gemm_to_dense_in_transpose_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_out_transpose_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_MatMul_2_output_0_gemm_to_dense_out_transpose_output_array, &_MatMul_2_output_0_gemm_to_dense_out_transpose_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 224), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &_MatMul_2_output_0_gemm_to_dense_output_array, &_MatMul_2_output_0_gemm_to_dense_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _Add_2_output_0_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &_Add_2_output_0_output_array, &_Add_2_output_0_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_10_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 12), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Constant_10_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_10_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  transpose_b_MatMul_4_output_0_out_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 224), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &transpose_b_MatMul_4_output_0_out_output_array, &transpose_b_MatMul_4_output_0_out_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_4_output_0_bias_0_conversion_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_MatMul_4_output_0_bias_0_conversion_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_4_output_0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 224, 1), AI_STRIDE_INIT(4, 4, 4, 48, 10752),
  1, &_MatMul_4_output_0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output0, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 224, 224, 1), AI_STRIDE_INIT(4, 4, 4, 896, 200704),
  1, &transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output0, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 224, 1), AI_STRIDE_INIT(4, 4, 4, 48, 10752),
  1, &transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output0, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 224), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &_MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output_array, &_MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  transpose_out_MatMul_4_output_0_out_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 1, 1, 224, 224),
  1, &transpose_out_MatMul_4_output_0_out_output_array, &transpose_out_MatMul_4_output_0_out_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 12), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_ReduceMean_output_0_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 224, 1, 12), AI_STRIDE_INIT(4, 4, 4, 896, 896),
  1, &transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output_array, NULL)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Unsqueeze_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_layer, 12,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Unsqueeze_output_0_to_chfirst_chain,
  NULL, &_Unsqueeze_output_0_to_chfirst_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_output_0_gemm_to_dense_out_transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_output_0_gemm_to_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_output_0_gemm_to_dense_out_transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_output_0_gemm_to_dense_out_transpose_layer, 24,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_MatMul_output_0_gemm_to_dense_out_transpose_chain,
  NULL, &_MatMul_output_0_gemm_to_dense_out_transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Constant_6_output_0_DequantizeLinear_Output_const_3D, &_MatMul_output_0_gemm_to_dense_out_transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_output_0_layer, 33,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_output_0_chain,
  NULL, &_Add_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_a_MatMul_3_output_0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_a_MatMul_3_output_0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_a_MatMul_3_output_0_out_layer, 45,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_a_MatMul_3_output_0_out_chain,
  NULL, &transpose_a_MatMul_3_output_0_out_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_in_transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_1_output_0_gemm_to_dense_in_transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_in_transpose_layer, 25,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_MatMul_1_output_0_gemm_to_dense_in_transpose_chain,
  NULL, &_MatMul_1_output_0_gemm_to_dense_in_transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_out_transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_1_output_0_gemm_to_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_1_output_0_gemm_to_dense_out_transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_1_output_0_gemm_to_dense_out_transpose_layer, 25,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_MatMul_1_output_0_gemm_to_dense_out_transpose_chain,
  NULL, &_MatMul_1_output_0_gemm_to_dense_out_transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Constant_8_output_0_DequantizeLinear_Output_const_3D, &_MatMul_1_output_0_gemm_to_dense_out_transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_1_output_0_layer, 34,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_1_output_0_chain,
  NULL, &_Add_1_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output0, &_Add_1_output_0_0_1__MatMul_3_output_0_conversion_output0, &_MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_3_output_0_layer, 45,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &_MatMul_3_output_0_chain,
  NULL, &_MatMul_3_output_0_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_out_MatMul_3_output_0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_out_MatMul_3_output_0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_out_MatMul_3_output_0_out_layer, 45,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_out_MatMul_3_output_0_out_chain,
  NULL, &transpose_out_MatMul_3_output_0_out_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &transpose_out_MatMul_3_output_0_out_output, &_Constant_11_output_0_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Div_output_0_layer, 48,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Div_output_0_chain,
  NULL, &_Div_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_a_MatMul_4_output_0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_a_MatMul_4_output_0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_a_MatMul_4_output_0_out_layer, 54,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_a_MatMul_4_output_0_out_chain,
  NULL, &transpose_a_MatMul_4_output_0_out_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_in_transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_2_output_0_gemm_to_dense_in_transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_in_transpose_layer, 26,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_MatMul_2_output_0_gemm_to_dense_in_transpose_chain,
  NULL, &_MatMul_2_output_0_gemm_to_dense_in_transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_out_transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_2_output_0_gemm_to_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_2_output_0_gemm_to_dense_out_transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_2_output_0_gemm_to_dense_out_transpose_layer, 26,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_MatMul_2_output_0_gemm_to_dense_out_transpose_chain,
  NULL, &_MatMul_2_output_0_gemm_to_dense_out_transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Constant_10_output_0_DequantizeLinear_Output_const_3D, &_MatMul_2_output_0_gemm_to_dense_out_transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_2_output_0_layer, 35,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_2_output_0_chain,
  NULL, &_Add_2_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_b_MatMul_4_output_0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_b_MatMul_4_output_0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_b_MatMul_4_output_0_out_layer, 54,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_b_MatMul_4_output_0_out_chain,
  NULL, &transpose_b_MatMul_4_output_0_out_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _MatMul_4_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output0, &transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output0, &_MatMul_4_output_0_bias_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_4_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _MatMul_4_output_0_layer, 54,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &_MatMul_4_output_0_chain,
  NULL, &_MatMul_4_output_0_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_out_MatMul_4_output_0_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_out_MatMul_4_output_0_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_out_MatMul_4_output_0_out_layer, 54,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_out_MatMul_4_output_0_out_chain,
  NULL, &transpose_out_MatMul_4_output_0_out_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_float _ReduceMean_output_0_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _ReduceMean_output_0_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _ReduceMean_output_0_neutral_value_data, _ReduceMean_output_0_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_layer, 57,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_ReduceMean_output_0_chain,
  NULL, &_ReduceMean_output_0_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_ReduceMean_output_0_neutral_value, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite__Unsqueeze_output_0_to_chfirst(_stai_network_context* net_ctx)
{
  input_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  input_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  _Unsqueeze_output_0_to_chfirst_output_array.data = AI_PTR(net_ctx->_activations[0] + 202804);
  _Unsqueeze_output_0_to_chfirst_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202804);
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, { input_output0.data->data});
  forward_transpose(&_Unsqueeze_output_0_to_chfirst_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, { _Unsqueeze_output_0_to_chfirst_output.data->data});
}
void forward_lite__MatMul_output_0_gemm_to_dense_out_transpose(_stai_network_context* net_ctx)
{
  _MatMul_output_0_gemm_to_dense_output_array.data = AI_PTR(net_ctx->_activations[0] + 202656);
  _MatMul_output_0_gemm_to_dense_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202656);
  _MatMul_output_0_gemm_to_dense_out_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 205344);
  _MatMul_output_0_gemm_to_dense_out_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 205344);
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, { _MatMul_output_0_gemm_to_dense_output.data->data});
  forward_transpose(&_MatMul_output_0_gemm_to_dense_out_transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, { _MatMul_output_0_gemm_to_dense_out_transpose_output.data->data});
}
void forward_lite__Add_output_0(_stai_network_context* net_ctx)
{
  _Constant_6_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 20);
  _Constant_6_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 20);
  _MatMul_output_0_gemm_to_dense_out_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 205344);
  _MatMul_output_0_gemm_to_dense_out_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 205344);
  _Add_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 205344);
  _Add_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 205344);
  _STAI_NETWORK_EVENT_NODE_START_CB(33, 2, { _Constant_6_output_0_DequantizeLinear_Output_const_3D.data->data,_MatMul_output_0_gemm_to_dense_out_transpose_output.data->data});
  forward_eltwise_integer_INT8(&_Add_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(33, 1, { _Add_output_0_output.data->data});
}
void forward_lite_transpose_a_MatMul_3_output_0_out(_stai_network_context* net_ctx)
{
  _Add_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 205344);
  _Add_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 205344);
  transpose_a_MatMul_3_output_0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 198016);
  transpose_a_MatMul_3_output_0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 198016);
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, { _Add_output_0_output.data->data});
  forward_transpose(&transpose_a_MatMul_3_output_0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, { transpose_a_MatMul_3_output_0_out_output.data->data});
}
void forward_lite__MatMul_1_output_0_gemm_to_dense_in_transpose(_stai_network_context* net_ctx)
{
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 200704);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 200704);
  _MatMul_1_output_0_gemm_to_dense_in_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 213256);
  _MatMul_1_output_0_gemm_to_dense_in_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 213256);
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, { _Relu_output_0_output0.data->data});
  forward_transpose(&_MatMul_1_output_0_gemm_to_dense_in_transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, { _MatMul_1_output_0_gemm_to_dense_in_transpose_output.data->data});
}
void forward_lite__MatMul_1_output_0_gemm_to_dense_out_transpose(_stai_network_context* net_ctx)
{
  _MatMul_1_output_0_gemm_to_dense_output_array.data = AI_PTR(net_ctx->_activations[0] + 215048);
  _MatMul_1_output_0_gemm_to_dense_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 215048);
  _MatMul_1_output_0_gemm_to_dense_out_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 198016);
  _MatMul_1_output_0_gemm_to_dense_out_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 198016);
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, { _MatMul_1_output_0_gemm_to_dense_output.data->data});
  forward_transpose(&_MatMul_1_output_0_gemm_to_dense_out_transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, { _MatMul_1_output_0_gemm_to_dense_out_transpose_output.data->data});
}
void forward_lite__Add_1_output_0(_stai_network_context* net_ctx)
{
  _Constant_8_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 8);
  _Constant_8_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 8);
  _MatMul_1_output_0_gemm_to_dense_out_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 198016);
  _MatMul_1_output_0_gemm_to_dense_out_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 198016);
  _Add_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 195328);
  _Add_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 195328);
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 2, { _Constant_8_output_0_DequantizeLinear_Output_const_3D.data->data,_MatMul_1_output_0_gemm_to_dense_out_transpose_output.data->data});
  forward_eltwise_integer_INT8(&_Add_1_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, { _Add_1_output_0_output.data->data});
}
void forward_lite__MatMul_3_output_0(_stai_network_context* net_ctx)
{
  transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 202504);
  transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202504);
  _Add_1_output_0_0_1__MatMul_3_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 213256);
  _Add_1_output_0_0_1__MatMul_3_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 213256);
  _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 202496);
  _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202496);
  _MatMul_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _MatMul_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 3, { transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_output0.data->data,_Add_1_output_0_0_1__MatMul_3_output_0_conversion_output0.data->data,_MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_output.data->data});
  forward_matmul(&_MatMul_3_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, { _MatMul_3_output_0_output.data->data});
}
void forward_lite_transpose_out_MatMul_3_output_0_out(_stai_network_context* net_ctx)
{
  _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_out_MatMul_3_output_0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 50176);
  transpose_out_MatMul_3_output_0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 50176);
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, { _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_output0.data->data});
  forward_transpose(&transpose_out_MatMul_3_output_0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, { transpose_out_MatMul_3_output_0_out_output.data->data});
}
void forward_lite__Div_output_0(_stai_network_context* net_ctx)
{
  transpose_out_MatMul_3_output_0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 50176);
  transpose_out_MatMul_3_output_0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 50176);
  _Constant_11_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 32);
  _Constant_11_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 32);
  _Div_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 150528);
  _Div_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 150528);
  _STAI_NETWORK_EVENT_NODE_START_CB(48, 2, { transpose_out_MatMul_3_output_0_out_output.data->data,_Constant_11_output_0_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&_Div_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(48, 1, { _Div_output_0_output.data->data});
}
void forward_lite_transpose_a_MatMul_4_output_0_out(_stai_network_context* net_ctx)
{
  _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_a_MatMul_4_output_0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 150528);
  transpose_a_MatMul_4_output_0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 150528);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, { _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_output.data->data});
  forward_transpose(&transpose_a_MatMul_4_output_0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { transpose_a_MatMul_4_output_0_out_output.data->data});
}
void forward_lite__MatMul_2_output_0_gemm_to_dense_in_transpose(_stai_network_context* net_ctx)
{
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 200704);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 200704);
  _MatMul_2_output_0_gemm_to_dense_in_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 202504);
  _MatMul_2_output_0_gemm_to_dense_in_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202504);
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, { _Relu_output_0_output0.data->data});
  forward_transpose(&_MatMul_2_output_0_gemm_to_dense_in_transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { _MatMul_2_output_0_gemm_to_dense_in_transpose_output.data->data});
}
void forward_lite__MatMul_2_output_0_gemm_to_dense_out_transpose(_stai_network_context* net_ctx)
{
  _MatMul_2_output_0_gemm_to_dense_output_array.data = AI_PTR(net_ctx->_activations[0] + 204296);
  _MatMul_2_output_0_gemm_to_dense_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 204296);
  _MatMul_2_output_0_gemm_to_dense_out_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 206984);
  _MatMul_2_output_0_gemm_to_dense_out_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 206984);
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, { _MatMul_2_output_0_gemm_to_dense_output.data->data});
  forward_transpose(&_MatMul_2_output_0_gemm_to_dense_out_transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { _MatMul_2_output_0_gemm_to_dense_out_transpose_output.data->data});
}
void forward_lite__Add_2_output_0(_stai_network_context* net_ctx)
{
  _Constant_10_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 36);
  _Constant_10_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 36);
  _MatMul_2_output_0_gemm_to_dense_out_transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 206984);
  _MatMul_2_output_0_gemm_to_dense_out_transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 206984);
  _Add_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 202504);
  _Add_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202504);
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 2, { _Constant_10_output_0_DequantizeLinear_Output_const_3D.data->data,_MatMul_2_output_0_gemm_to_dense_out_transpose_output.data->data});
  forward_eltwise_integer_INT8(&_Add_2_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, { _Add_2_output_0_output.data->data});
}
void forward_lite_transpose_b_MatMul_4_output_0_out(_stai_network_context* net_ctx)
{
  _Add_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 202504);
  _Add_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202504);
  transpose_b_MatMul_4_output_0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 205192);
  transpose_b_MatMul_4_output_0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 205192);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, { _Add_2_output_0_output.data->data});
  forward_transpose(&transpose_b_MatMul_4_output_0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { transpose_b_MatMul_4_output_0_out_output.data->data});
}
void forward_lite__MatMul_4_output_0(_stai_network_context* net_ctx)
{
  transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 213256);
  transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 213256);
  _MatMul_4_output_0_bias_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 202500);
  _MatMul_4_output_0_bias_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202500);
  _MatMul_4_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 202504);
  _MatMul_4_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 202504);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 3, { transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_output0.data->data,transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_output0.data->data,_MatMul_4_output_0_bias_0_conversion_output.data->data});
  forward_matmul(&_MatMul_4_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { _MatMul_4_output_0_output.data->data});
}
void forward_lite_transpose_out_MatMul_4_output_0_out(_stai_network_context* net_ctx)
{
  _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_out_MatMul_4_output_0_out_output_array.data = AI_PTR(net_ctx->_activations[0] + 2688);
  transpose_out_MatMul_4_output_0_out_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2688);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, { _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_output0.data->data});
  forward_transpose(&transpose_out_MatMul_4_output_0_out_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { transpose_out_MatMul_4_output_0_out_output.data->data});
}
void forward_lite__ReduceMean_output_0(_stai_network_context* net_ctx)
{
  transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 5376);
  transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5376);
  _ReduceMean_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _ReduceMean_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(57, 1, { transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_output.data->data});
  forward_reduce(&_ReduceMean_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(57, 1, { _ReduceMean_output_0_output.data->data});
}

/*****************************************************************************/


static const ai_u32 _MatMul_4_output_0_bias_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float _MatMul_4_output_0_bias_0_conversion_t_in_0_fmt_scale_const_f32 = 0.002826663199812174f;
static const ai_i8 _MatMul_4_output_0_bias_0_conversion_t_in_0_fmt_zero_const_s8 = 30;

static const ai_u32 _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.003893679240718484f;
static const ai_i8 _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_fmt_zero_const_s8 = -90;


static const ai_u16 _Relu_output_0_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 _Relu_output_0_t_in_0_shape_h_const_u16 = 30;
static const ai_u16 _Relu_output_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 _Relu_output_0_t_out_0_shape_ch_const_u16 = 8;
static const ai_u16 _Relu_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _Relu_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_i32 _Relu_output_0_l_pad_W_0_const_s32 = 0;
static const ai_i32 _Relu_output_0_l_pad_H_0_const_s32 = 0;
static const ai_i8 _Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.00392132019624114f;
static const ai_float _Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.004052215255796909f;
static const ai_float _Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0022042631171643734f, 0.0025125849060714245f, 0.002254191320389509f, 0.002120143501088023f, 0.002294437261298299f, 0.0023663968313485384f, 0.002276182873174548f, 0.002575209364295006f);
static const ai_layer_format_type _Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _Relu_output_0_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 _Relu_output_0_t_out_0_shape_h_const_u16 = 28;

static const ai_u16 _MatMul_output_0_gemm_to_dense_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 _MatMul_output_0_gemm_to_dense_t_in_0_shape_h_const_u16 = 224;
static const ai_u16 _MatMul_output_0_gemm_to_dense_l_stride_1_const_u16 = 1;
static const ai_u16 _MatMul_output_0_gemm_to_dense_l_stride_0_const_u16 = 1;
static const ai_u16 _MatMul_output_0_gemm_to_dense_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 _MatMul_output_0_gemm_to_dense_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _MatMul_output_0_gemm_to_dense_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _MatMul_output_0_gemm_to_dense_t_out_0_fmt_zero_const_s8 = -25;
static const ai_float _MatMul_output_0_gemm_to_dense_t_in_0_fmt_scale_const_f32 = 0.004052215255796909f;
static const ai_float _MatMul_output_0_gemm_to_dense_t_out_0_fmt_scale_const_f32 = 0.004095243755728006f;
static const ai_float _MatMul_output_0_gemm_to_dense_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0027328624855726957f, 0.002232017694041133f, 0.0027705428656190634f, 0.0027684452943503857f, 0.0022705725859850645f, 0.0027693226002156734f, 0.0026929068844765425f, 0.00257574999704957f, 0.002628007438033819f, 0.002183976350352168f, 0.002391006564721465f, 0.002751474268734455f);
static const ai_layer_format_type _MatMul_output_0_gemm_to_dense_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u32 transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 2688;
static const ai_float transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.005318579729646444f;
static const ai_i8 transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_fmt_zero_const_s8 = -20;


static const ai_u16 _MatMul_1_output_0_gemm_to_dense_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 _MatMul_1_output_0_gemm_to_dense_t_in_0_shape_h_const_u16 = 224;
static const ai_u16 _MatMul_1_output_0_gemm_to_dense_l_stride_1_const_u16 = 1;
static const ai_u16 _MatMul_1_output_0_gemm_to_dense_l_stride_0_const_u16 = 1;
static const ai_u16 _MatMul_1_output_0_gemm_to_dense_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 _MatMul_1_output_0_gemm_to_dense_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _MatMul_1_output_0_gemm_to_dense_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _MatMul_1_output_0_gemm_to_dense_t_out_0_fmt_zero_const_s8 = -41;
static const ai_float _MatMul_1_output_0_gemm_to_dense_t_in_0_fmt_scale_const_f32 = 0.004052215255796909f;
static const ai_float _MatMul_1_output_0_gemm_to_dense_t_out_0_fmt_scale_const_f32 = 0.003841881174594164f;
static const ai_float _MatMul_1_output_0_gemm_to_dense_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0012837790418416262f, 0.0025637508369982243f, 0.002187179634347558f, 0.0026050880551338196f, 0.002763984026387334f, 0.0026011448353528976f, 0.002586688380688429f, 0.002566338051110506f, 0.0026644712779670954f, 0.0027172635309398174f, 0.002364447107538581f, 0.0017254971899092197f);
static const ai_layer_format_type _MatMul_1_output_0_gemm_to_dense_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u32 _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 2688;
static const ai_float _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.004937407560646534f;
static const ai_i8 _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_fmt_zero_const_s8 = -28;


static const ai_u32 _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 50176;
static const ai_float _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_fmt_scale_const_f32 = 0.003893679240718484f;
static const ai_i8 _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_fmt_zero_const_s8 = -90;



static const ai_u32 _Div_output_0_0_0__Softmax_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 50176;
static const ai_float _Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.0011240083258599043f;
static const ai_i8 _Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_fmt_zero_const_s8 = -90;

static const ai_i32 _Softmax_output_0_t_in_0_shape_ch_h_prod_const_s32 = 50176;

static const ai_u32 _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 50176;
static const ai_float _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_i8 _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 50176;
static const ai_float transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_i8 transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_fmt_zero_const_s8 = -128;


static const ai_u16 _MatMul_2_output_0_gemm_to_dense_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 _MatMul_2_output_0_gemm_to_dense_t_in_0_shape_h_const_u16 = 224;
static const ai_u16 _MatMul_2_output_0_gemm_to_dense_l_stride_1_const_u16 = 1;
static const ai_u16 _MatMul_2_output_0_gemm_to_dense_l_stride_0_const_u16 = 1;
static const ai_u16 _MatMul_2_output_0_gemm_to_dense_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 _MatMul_2_output_0_gemm_to_dense_t_out_0_shape_ch_const_u16 = 12;
static const ai_i8 _MatMul_2_output_0_gemm_to_dense_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _MatMul_2_output_0_gemm_to_dense_t_out_0_fmt_zero_const_s8 = 23;
static const ai_float _MatMul_2_output_0_gemm_to_dense_t_in_0_fmt_scale_const_f32 = 0.004052215255796909f;
static const ai_float _MatMul_2_output_0_gemm_to_dense_t_out_0_fmt_scale_const_f32 = 0.0034422490280121565f;
static const ai_float _MatMul_2_output_0_gemm_to_dense_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0020179850980639458f, 0.0027524775359779596f, 0.0027830323670059443f, 0.0026932074688374996f, 0.0020805648528039455f, 0.002524763345718384f, 0.002299726475030184f, 0.0025465693324804306f, 0.002776668407022953f, 0.002637261990457773f, 0.0027282179798930883f, 0.0025389601942151785f);
static const ai_layer_format_type _MatMul_2_output_0_gemm_to_dense_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u32 transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 2688;
static const ai_float transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.004716664552688599f;
static const ai_i8 transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_fmt_zero_const_s8 = 31;


static const ai_u32 _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 2688;
static const ai_float _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_fmt_scale_const_f32 = 0.002826663199812174f;
static const ai_i8 _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_fmt_zero_const_s8 = 30;


static const ai_u32 transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 2688;
static const ai_float transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.002826663199812174f;
static const ai_i8 transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_fmt_zero_const_s8 = 30;



static const ai_u32 _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12;
static const ai_float _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_fmt_scale_const_f32 = 0.002824777038767934f;
static const ai_i8 _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_fmt_zero_const_s8 = 30;

static const ai_i8 output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = 30;
static const ai_i8 output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = -128;
static const ai_u16 output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 12;
static const ai_u16 output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 1;
static const ai_u32 output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32 = 1;
static const ai_float output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.002824777038767934f;
static const ai_float output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.000644468585960567f;
static const ai_float output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32 = 0.0021276604384183884f;
STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN _MatMul_4_output_0_bias_0_conversion */
  {
      const ai_i8* _MatMul_4_output_0_bias_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    ai_float* _MatMul_4_output_0_bias_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 202500);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) _MatMul_4_output_0_bias_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_MatMul_4_output_0_bias_0_conversion_t_in_0_ptr_const_s8, _MatMul_4_output_0_bias_0_conversion_t_out_0_ptr_f32, _MatMul_4_output_0_bias_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _MatMul_4_output_0_bias_0_conversion_t_in_0_fmt_scale_const_f32, _MatMul_4_output_0_bias_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) _MatMul_4_output_0_bias_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _MatMul_4_output_0_bias_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion */
  {
      const ai_i8* _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4);
    ai_float* _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 202496);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, {(stai_ptr) _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8, _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_out_0_ptr_f32, _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_fmt_scale_const_f32, _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _MatMul_3_output_0_bias_0_2__MatMul_3_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _Unsqueeze_output_0_to_chfirst */
  {
    
  forward_lite__Unsqueeze_output_0_to_chfirst(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Unsqueeze_output_0_to_chfirst */
  /* LITE_KERNEL_SECTION BEGIN _Relu_output_0 */
  {
      const ai_i8* _Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 202804);
    const ai_i8* _Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 48);
    const ai_i32* _Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 120);
    ai_i8* _Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 200704);
    ai_i16* _Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 203104);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) _Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_Relu_output_0_t_in_0_ptr_const_s8, _Relu_output_0_t_in_0_shape_w_const_u16, _Relu_output_0_t_in_0_shape_h_const_u16, _Relu_output_0_t_in_0_shape_ch_const_u16, _Relu_output_0_t_weight_0_ptr_const_s8, _Relu_output_0_t_out_0_shape_ch_const_u16, _Relu_output_0_t_weight_0_shape_w_const_u16, _Relu_output_0_t_weight_0_shape_h_const_u16, _Relu_output_0_l_stride_1_const_u16, _Relu_output_0_l_stride_0_const_u16, _Relu_output_0_l_pad_W_0_const_s32, _Relu_output_0_l_pad_H_0_const_s32, _Relu_output_0_t_weight_1_ptr_const_s32, _Relu_output_0_t_in_0_fmt_zero_const_s8, _Relu_output_0_t_out_0_fmt_zero_const_s8, _Relu_output_0_t_in_0_fmt_scale_const_f32, _Relu_output_0_t_out_0_fmt_scale_const_f32, _Relu_output_0_t_weight_0_fmt_scale_const_f32, _Relu_output_0_l_out_ch_format_const_layer_format_type, _Relu_output_0_t_out_0_ptr_s8, _Relu_output_0_t_out_0_shape_w_const_u16, _Relu_output_0_t_out_0_shape_h_const_u16, 1, 292, _Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) _Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_output_0_gemm_to_dense */
  {
      const ai_i8* _MatMul_output_0_gemm_to_dense_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 200704);
    const ai_i8* _MatMul_output_0_gemm_to_dense_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 152);
    const ai_i32* _MatMul_output_0_gemm_to_dense_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 248);
    ai_i8* _MatMul_output_0_gemm_to_dense_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 202656);
    ai_i16* _MatMul_output_0_gemm_to_dense_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 202504);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, {(stai_ptr) _MatMul_output_0_gemm_to_dense_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_MatMul_output_0_gemm_to_dense_t_in_0_ptr_const_s8, _MatMul_output_0_gemm_to_dense_t_in_0_shape_w_const_u16, _MatMul_output_0_gemm_to_dense_t_in_0_shape_h_const_u16, _MatMul_output_0_gemm_to_dense_l_stride_1_const_u16, _MatMul_output_0_gemm_to_dense_l_stride_0_const_u16, _MatMul_output_0_gemm_to_dense_t_in_0_shape_ch_const_u16, _MatMul_output_0_gemm_to_dense_t_weight_0_ptr_const_s8, _MatMul_output_0_gemm_to_dense_t_out_0_shape_ch_const_u16, _MatMul_output_0_gemm_to_dense_t_weight_1_ptr_const_s32, _MatMul_output_0_gemm_to_dense_t_in_0_fmt_zero_const_s8, _MatMul_output_0_gemm_to_dense_t_out_0_fmt_zero_const_s8, _MatMul_output_0_gemm_to_dense_t_in_0_fmt_scale_const_f32, _MatMul_output_0_gemm_to_dense_t_out_0_fmt_scale_const_f32, _MatMul_output_0_gemm_to_dense_t_weight_0_fmt_scale_const_f32, _MatMul_output_0_gemm_to_dense_l_out_ch_format_const_layer_format_type, _MatMul_output_0_gemm_to_dense_t_out_0_ptr_s8, 1, 152, _MatMul_output_0_gemm_to_dense_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, {(stai_ptr) _MatMul_output_0_gemm_to_dense_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _MatMul_output_0_gemm_to_dense */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_output_0_gemm_to_dense_out_transpose */
  {
    
  forward_lite__MatMul_output_0_gemm_to_dense_out_transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_output_0_gemm_to_dense_out_transpose */
  /* LITE_KERNEL_SECTION BEGIN _Add_output_0 */
  {
    
  forward_lite__Add_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Add_output_0 */
  /* LITE_KERNEL_SECTION BEGIN transpose_a_MatMul_3_output_0_out */
  {
    
  forward_lite_transpose_a_MatMul_3_output_0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_a_MatMul_3_output_0_out */
  /* LITE_KERNEL_SECTION BEGIN transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion */
  {
      const ai_i8* transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 198016);
    ai_float* transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 202504);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, {(stai_ptr) transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8, transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_out_0_ptr_f32, transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_fmt_scale_const_f32, transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END transpose_a_MatMul_3_output_0_out_0_0__MatMul_3_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_1_output_0_gemm_to_dense_in_transpose */
  {
    
  forward_lite__MatMul_1_output_0_gemm_to_dense_in_transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_1_output_0_gemm_to_dense_in_transpose */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_1_output_0_gemm_to_dense */
  {
      const ai_i8* _MatMul_1_output_0_gemm_to_dense_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 213256);
    const ai_i8* _MatMul_1_output_0_gemm_to_dense_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 296);
    const ai_i32* _MatMul_1_output_0_gemm_to_dense_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 248);
    ai_i8* _MatMul_1_output_0_gemm_to_dense_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 215048);
    ai_i16* _MatMul_1_output_0_gemm_to_dense_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 200552);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, {(stai_ptr) _MatMul_1_output_0_gemm_to_dense_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_MatMul_1_output_0_gemm_to_dense_t_in_0_ptr_const_s8, _MatMul_1_output_0_gemm_to_dense_t_in_0_shape_w_const_u16, _MatMul_1_output_0_gemm_to_dense_t_in_0_shape_h_const_u16, _MatMul_1_output_0_gemm_to_dense_l_stride_1_const_u16, _MatMul_1_output_0_gemm_to_dense_l_stride_0_const_u16, _MatMul_1_output_0_gemm_to_dense_t_in_0_shape_ch_const_u16, _MatMul_1_output_0_gemm_to_dense_t_weight_0_ptr_const_s8, _MatMul_1_output_0_gemm_to_dense_t_out_0_shape_ch_const_u16, _MatMul_1_output_0_gemm_to_dense_t_weight_1_ptr_const_s32, _MatMul_1_output_0_gemm_to_dense_t_in_0_fmt_zero_const_s8, _MatMul_1_output_0_gemm_to_dense_t_out_0_fmt_zero_const_s8, _MatMul_1_output_0_gemm_to_dense_t_in_0_fmt_scale_const_f32, _MatMul_1_output_0_gemm_to_dense_t_out_0_fmt_scale_const_f32, _MatMul_1_output_0_gemm_to_dense_t_weight_0_fmt_scale_const_f32, _MatMul_1_output_0_gemm_to_dense_l_out_ch_format_const_layer_format_type, _MatMul_1_output_0_gemm_to_dense_t_out_0_ptr_s8, 1, 152, _MatMul_1_output_0_gemm_to_dense_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, {(stai_ptr) _MatMul_1_output_0_gemm_to_dense_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _MatMul_1_output_0_gemm_to_dense */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_1_output_0_gemm_to_dense_out_transpose */
  {
    
  forward_lite__MatMul_1_output_0_gemm_to_dense_out_transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_1_output_0_gemm_to_dense_out_transpose */
  /* LITE_KERNEL_SECTION BEGIN _Add_1_output_0 */
  {
    
  forward_lite__Add_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Add_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Add_1_output_0_0_1__MatMul_3_output_0_conversion */
  {
      const ai_i8* _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 195328);
    ai_float* _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 213256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_ptr_const_s8, _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_out_0_ptr_f32, _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_fmt_scale_const_f32, _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) _Add_1_output_0_0_1__MatMul_3_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _Add_1_output_0_0_1__MatMul_3_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_3_output_0 */
  {
    
  forward_lite__MatMul_3_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_3_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion */
  {
      const ai_float* _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, {(stai_ptr) _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_in_0_ptr_const_f32, _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_ptr_s8, _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_fmt_scale_const_f32, _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _MatMul_3_output_0_0_0_transpose_out_MatMul_3_output_0_out_conversion */
  /* LITE_KERNEL_SECTION BEGIN transpose_out_MatMul_3_output_0_out */
  {
    
  forward_lite_transpose_out_MatMul_3_output_0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_out_MatMul_3_output_0_out */
  /* LITE_KERNEL_SECTION BEGIN _Div_output_0 */
  {
    
  forward_lite__Div_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Div_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Div_output_0_0_0__Softmax_output_0_conversion */
  {
      const ai_i8* _Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 150528);
    ai_float* _Div_output_0_0_0__Softmax_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(48, 1, {(stai_ptr) _Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_ptr_const_s8, _Div_output_0_0_0__Softmax_output_0_conversion_t_out_0_ptr_f32, _Div_output_0_0_0__Softmax_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_fmt_scale_const_f32, _Div_output_0_0_0__Softmax_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(48, 1, {(stai_ptr) _Div_output_0_0_0__Softmax_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _Div_output_0_0_0__Softmax_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _Softmax_output_0 */
  {
      ai_handle _Softmax_output_0_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle _Softmax_output_0_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(51, 1, {(stai_ptr) _Softmax_output_0_t_in_0_ptr_const_handle});
    
  forward_lite_nl_softmax_if32of32(_Softmax_output_0_t_out_0_ptr_handle, _Softmax_output_0_t_in_0_ptr_const_handle, _Softmax_output_0_t_in_0_shape_ch_h_prod_const_s32, 224, 224);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(51, 1, {(stai_ptr) _Softmax_output_0_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END _Softmax_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion */
  {
      const ai_float* _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(51, 1, {(stai_ptr) _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_in_0_ptr_const_f32, _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_ptr_s8, _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_fmt_scale_const_f32, _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(51, 1, {(stai_ptr) _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _Softmax_output_0_0_0_transpose_a_MatMul_4_output_0_out_conversion */
  /* LITE_KERNEL_SECTION BEGIN transpose_a_MatMul_4_output_0_out */
  {
    
  forward_lite_transpose_a_MatMul_4_output_0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_a_MatMul_4_output_0_out */
  /* LITE_KERNEL_SECTION BEGIN transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion */
  {
      const ai_i8* transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 150528);
    ai_float* transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_ptr_const_s8, transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_out_0_ptr_f32, transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_fmt_scale_const_f32, transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END transpose_a_MatMul_4_output_0_out_0_0__MatMul_4_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_2_output_0_gemm_to_dense_in_transpose */
  {
    
  forward_lite__MatMul_2_output_0_gemm_to_dense_in_transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_2_output_0_gemm_to_dense_in_transpose */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_2_output_0_gemm_to_dense */
  {
      const ai_i8* _MatMul_2_output_0_gemm_to_dense_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 202504);
    const ai_i8* _MatMul_2_output_0_gemm_to_dense_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 392);
    const ai_i32* _MatMul_2_output_0_gemm_to_dense_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 248);
    ai_i8* _MatMul_2_output_0_gemm_to_dense_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 204296);
    ai_i16* _MatMul_2_output_0_gemm_to_dense_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 200704);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, {(stai_ptr) _MatMul_2_output_0_gemm_to_dense_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_MatMul_2_output_0_gemm_to_dense_t_in_0_ptr_const_s8, _MatMul_2_output_0_gemm_to_dense_t_in_0_shape_w_const_u16, _MatMul_2_output_0_gemm_to_dense_t_in_0_shape_h_const_u16, _MatMul_2_output_0_gemm_to_dense_l_stride_1_const_u16, _MatMul_2_output_0_gemm_to_dense_l_stride_0_const_u16, _MatMul_2_output_0_gemm_to_dense_t_in_0_shape_ch_const_u16, _MatMul_2_output_0_gemm_to_dense_t_weight_0_ptr_const_s8, _MatMul_2_output_0_gemm_to_dense_t_out_0_shape_ch_const_u16, _MatMul_2_output_0_gemm_to_dense_t_weight_1_ptr_const_s32, _MatMul_2_output_0_gemm_to_dense_t_in_0_fmt_zero_const_s8, _MatMul_2_output_0_gemm_to_dense_t_out_0_fmt_zero_const_s8, _MatMul_2_output_0_gemm_to_dense_t_in_0_fmt_scale_const_f32, _MatMul_2_output_0_gemm_to_dense_t_out_0_fmt_scale_const_f32, _MatMul_2_output_0_gemm_to_dense_t_weight_0_fmt_scale_const_f32, _MatMul_2_output_0_gemm_to_dense_l_out_ch_format_const_layer_format_type, _MatMul_2_output_0_gemm_to_dense_t_out_0_ptr_s8, 1, 152, _MatMul_2_output_0_gemm_to_dense_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, {(stai_ptr) _MatMul_2_output_0_gemm_to_dense_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _MatMul_2_output_0_gemm_to_dense */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_2_output_0_gemm_to_dense_out_transpose */
  {
    
  forward_lite__MatMul_2_output_0_gemm_to_dense_out_transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_2_output_0_gemm_to_dense_out_transpose */
  /* LITE_KERNEL_SECTION BEGIN _Add_2_output_0 */
  {
    
  forward_lite__Add_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Add_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN transpose_b_MatMul_4_output_0_out */
  {
    
  forward_lite_transpose_b_MatMul_4_output_0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_b_MatMul_4_output_0_out */
  /* LITE_KERNEL_SECTION BEGIN transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion */
  {
      const ai_i8* transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 205192);
    ai_float* transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 213256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_ptr_const_s8, transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_out_0_ptr_f32, transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_fmt_scale_const_f32, transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END transpose_b_MatMul_4_output_0_out_0_1__MatMul_4_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_4_output_0 */
  {
    
  forward_lite__MatMul_4_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _MatMul_4_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion */
  {
      const ai_float* _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 202504);
    ai_i8* _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_in_0_ptr_const_f32, _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_ptr_s8, _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_fmt_scale_const_f32, _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _MatMul_4_output_0_0_0_transpose_out_MatMul_4_output_0_out_conversion */
  /* LITE_KERNEL_SECTION BEGIN transpose_out_MatMul_4_output_0_out */
  {
    
  forward_lite_transpose_out_MatMul_4_output_0_out(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_out_MatMul_4_output_0_out */
  /* LITE_KERNEL_SECTION BEGIN transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion */
  {
      const ai_i8* transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2688);
    ai_float* transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 5376);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_ptr_const_s8, transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_out_0_ptr_f32, transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_fmt_scale_const_f32, transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END transpose_out_MatMul_4_output_0_out_0_0__ReduceMean_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _ReduceMean_output_0 */
  {
    
  forward_lite__ReduceMean_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _ReduceMean_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _ReduceMean_output_0_Mul */
  {
      ai_float* _ReduceMean_output_0_Mul_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 48);
    const ai_float* _ReduceMean_output_0_Mul_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_float* _ReduceMean_output_0_Mul_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 488);
    const ai_float* _ReduceMean_output_0_Mul_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 536);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(57, 1, {(stai_ptr) _ReduceMean_output_0_Mul_t_in_0_ptr_const_f32});
    
  forward_lite_bn_if32of32wf32(_ReduceMean_output_0_Mul_t_out_0_ptr_f32, _ReduceMean_output_0_Mul_t_in_0_ptr_const_f32, _ReduceMean_output_0_Mul_t_weight_0_ptr_const_f32, _ReduceMean_output_0_Mul_t_weight_1_ptr_const_f32, (ai_u32)(12), (ai_size)(12));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(57, 1, {(stai_ptr) _ReduceMean_output_0_Mul_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _ReduceMean_output_0_Mul */
  /* LITE_KERNEL_SECTION BEGIN _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion */
  {
      const ai_float* _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 48);
    ai_i8* _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(57, 1, {(stai_ptr) _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_in_0_ptr_const_f32, _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_ptr_s8, _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_fmt_scale_const_f32, _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(57, 1, {(stai_ptr) _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _ReduceMean_output_0_Mul_0_0_output_QuantizeLinear_Input_conversion */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
      ai_i8* output_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[0] + 0);
    const ai_i8* output_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* output_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 584);
    const ai_i32* output_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 596);
    ai_i16* output_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(60, 1, {(stai_ptr) output_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(output_QuantizeLinear_Input_t_out_0_ptr_s8, output_QuantizeLinear_Input_t_in_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_1_ptr_const_s32, output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32, output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(60, 1, {(stai_ptr) output_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END output_QuantizeLinear_Input */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 202504;

  net_ctx->_outputs[0] = activations[0] + 36;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

