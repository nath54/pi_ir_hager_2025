/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-17T10:29:04+0000
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

#include "lite_operators.h"

#include "ai_lite_inspect.h"
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
#define _STAI_NETWORK_MODEL_SIGNATURE     "0xa83b8c32431908f4194065981094b90c"
#define _STAI_NETWORK_DATETIME            "2026-01-17T10:29:04+0000"
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
      STAI_DECLARE_ARRAY(float, 1, 0.003916791640222073f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 2.540426248742733e-05f),
      STAI_DECLARE_ARRAY(int16_t, 1, -23)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 109296),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 297668),
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
    AI_PACK_INTQ_SCALE(0.003916791640222073f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Unsqueeze_output_0_to_chfirst_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003916791640222073f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00037799568963237107f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Flatten_output_0_to_chlast_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00037799568963237107f),
    AI_PACK_INTQ_ZP(-128)))



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
  _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24300, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24300, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 10, 30), AI_STRIDE_INIT(4, 1, 1, 1, 10),
  1, &_Unsqueeze_output_0_to_chfirst_output_array, &_Unsqueeze_output_0_to_chfirst_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output0, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 30, 1), AI_STRIDE_INIT(4, 1, 1, 10, 300),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 30, 81), AI_STRIDE_INIT(4, 1, 1, 10, 300),
  1, &_Flatten_output_0_to_chlast_output_array, &_Flatten_output_0_to_chlast_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 81, 10, 30), AI_STRIDE_INIT(4, 1, 1, 81, 810),
  1, &_Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output_array, &_Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Unsqueeze_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Unsqueeze_output_0_to_chfirst_layer, 13,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Unsqueeze_output_0_to_chfirst_chain,
  NULL, &_Unsqueeze_output_0_to_chfirst_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Flatten_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Flatten_output_0_to_chlast_layer, 46,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Flatten_output_0_to_chlast_chain,
  NULL, &_Flatten_output_0_to_chlast_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite__Unsqueeze_output_0_to_chfirst(_stai_network_context* net_ctx)
{
  input_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  input_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  _Unsqueeze_output_0_to_chfirst_output_array.data = AI_PTR(net_ctx->_activations[0] + 44420);
  _Unsqueeze_output_0_to_chfirst_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 44420);
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, { input_output0.data->data});
  forward_transpose(&_Unsqueeze_output_0_to_chfirst_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, { _Unsqueeze_output_0_to_chfirst_output.data->data});
}
void forward_lite__Flatten_output_0_to_chlast(_stai_network_context* net_ctx)
{
  _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 3564);
  _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3564);
  _Flatten_output_0_to_chlast_output_array.data = AI_PTR(net_ctx->_activations[0] + 27864);
  _Flatten_output_0_to_chlast_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 27864);
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, { _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_output.data->data});
  forward_transpose(&_Flatten_output_0_to_chlast_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, { _Flatten_output_0_to_chlast_output.data->data});
}

/*****************************************************************************/



static const ai_u16 _Relu_output_0_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 _Relu_output_0_t_in_0_shape_h_const_u16 = 30;
static const ai_u16 _Relu_output_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 _Relu_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_u16 _Relu_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _Relu_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _Relu_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _Relu_output_0_l_stride_0_const_u16 = 1;
static const ai_i32 _Relu_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _Relu_output_0_l_pad_H_0_const_s32 = 1;
static const ai_i8 _Relu_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _Relu_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _Relu_output_0_t_in_0_fmt_scale_const_f32 = 0.003916791640222073f;
static const ai_float _Relu_output_0_t_out_0_fmt_scale_const_f32 = 0.0038277541752904654f;
static const ai_float _Relu_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0025296518579125404f, 0.002515533473342657f, 0.0023252612445503473f, 0.0025153551250696182f, 0.002545488765463233f, 0.002402606653049588f, 0.0022779074497520924f, 0.002435168484225869f, 0.0022977408953011036f, 0.0025471763219684362f, 0.0025070211850106716f, 0.002611519768834114f, 0.002606332302093506f, 0.002536545041948557f, 0.0025000451132655144f, 0.002576937898993492f);
static const ai_layer_format_type _Relu_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _Relu_output_0_t_out_0_shape_w_const_u16 = 10;
static const ai_u16 _Relu_output_0_t_out_0_shape_h_const_u16 = 30;

static const ai_u32 _MaxPool_output_0_DequantizeLinear_Output_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4800;
static const ai_float _MaxPool_output_0_DequantizeLinear_Output_t_in_0_fmt_scale_const_f32 = 0.0038277541752904654f;
static const ai_i8 _MaxPool_output_0_DequantizeLinear_Output_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 _Relu_1_output_0_t_in_0_shape_ch_const_u32 = 16;
static const ai_u32 _Relu_1_output_0_t_out_0_shape_ch_const_u32 = 24;
static const ai_u32 _Relu_1_output_0_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_1_output_0_t_in_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_1_output_0_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_1_output_0_t_out_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_1_output_0_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 _Relu_1_output_0_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 _Relu_1_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _Relu_1_output_0_l_pad_H_0_const_s32 = 1;
static const ai_u16 _Relu_1_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _Relu_1_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _Relu_1_output_0_l_dilation_H_const_u16 = 1;
static const ai_u16 _Relu_1_output_0_l_dilation_W_const_u16 = 1;

static const ai_u32 _Relu_2_output_0_t_in_0_shape_ch_const_u32 = 24;
static const ai_u32 _Relu_2_output_0_t_out_0_shape_ch_const_u32 = 36;
static const ai_u32 _Relu_2_output_0_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_2_output_0_t_in_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_2_output_0_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_2_output_0_t_out_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_2_output_0_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 _Relu_2_output_0_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 _Relu_2_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _Relu_2_output_0_l_pad_H_0_const_s32 = 1;
static const ai_u16 _Relu_2_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _Relu_2_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _Relu_2_output_0_l_dilation_H_const_u16 = 1;
static const ai_u16 _Relu_2_output_0_l_dilation_W_const_u16 = 1;

static const ai_u32 _Relu_3_output_0_t_in_0_shape_ch_const_u32 = 36;
static const ai_u32 _Relu_3_output_0_t_out_0_shape_ch_const_u32 = 54;
static const ai_u32 _Relu_3_output_0_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_3_output_0_t_in_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_3_output_0_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_3_output_0_t_out_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_3_output_0_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 _Relu_3_output_0_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 _Relu_3_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _Relu_3_output_0_l_pad_H_0_const_s32 = 1;
static const ai_u16 _Relu_3_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _Relu_3_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _Relu_3_output_0_l_dilation_H_const_u16 = 1;
static const ai_u16 _Relu_3_output_0_l_dilation_W_const_u16 = 1;

static const ai_u32 _Relu_4_output_0_t_in_0_shape_ch_const_u32 = 54;
static const ai_u32 _Relu_4_output_0_t_out_0_shape_ch_const_u32 = 81;
static const ai_u32 _Relu_4_output_0_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_4_output_0_t_in_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_4_output_0_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 _Relu_4_output_0_t_out_0_shape_h_const_u32 = 30;
static const ai_u32 _Relu_4_output_0_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 _Relu_4_output_0_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 _Relu_4_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _Relu_4_output_0_l_pad_H_0_const_s32 = 1;
static const ai_u16 _Relu_4_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _Relu_4_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _Relu_4_output_0_l_dilation_H_const_u16 = 1;
static const ai_u16 _Relu_4_output_0_l_dilation_W_const_u16 = 1;

static const ai_u32 _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 24300;
static const ai_float _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_fmt_scale_const_f32 = 0.00037799568963237107f;
static const ai_i8 _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_i8 output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = -23;
static const ai_u16 output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 24300;
static const ai_u16 output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 1;
static const ai_u32 output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32 = 1;
static const ai_float output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.00037799568963237107f;
static const ai_float output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 2.540426248742733e-05f;
static const ai_float output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32 = 5.0510221626609564e-05f;
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


  /* LITE_KERNEL_SECTION BEGIN _Unsqueeze_output_0_to_chfirst */
  {
    
  forward_lite__Unsqueeze_output_0_to_chfirst(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Unsqueeze_output_0_to_chfirst */
  /* LITE_KERNEL_SECTION BEGIN _Relu_output_0 */
  {
      const ai_i8* _Relu_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 44420);
    const ai_i8* _Relu_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    const ai_i32* _Relu_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 144);
    ai_i8* _Relu_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 44720);
    ai_i16* _Relu_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 43872);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) _Relu_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_Relu_output_0_t_in_0_ptr_const_s8, _Relu_output_0_t_in_0_shape_w_const_u16, _Relu_output_0_t_in_0_shape_h_const_u16, _Relu_output_0_t_in_0_shape_ch_const_u16, _Relu_output_0_t_weight_0_ptr_const_s8, _Relu_output_0_t_out_0_shape_ch_const_u16, _Relu_output_0_t_weight_0_shape_w_const_u16, _Relu_output_0_t_weight_0_shape_h_const_u16, _Relu_output_0_l_stride_1_const_u16, _Relu_output_0_l_stride_0_const_u16, _Relu_output_0_l_pad_W_0_const_s32, _Relu_output_0_l_pad_H_0_const_s32, _Relu_output_0_t_weight_1_ptr_const_s32, _Relu_output_0_t_in_0_fmt_zero_const_s8, _Relu_output_0_t_out_0_fmt_zero_const_s8, _Relu_output_0_t_in_0_fmt_scale_const_f32, _Relu_output_0_t_out_0_fmt_scale_const_f32, _Relu_output_0_t_weight_0_fmt_scale_const_f32, _Relu_output_0_l_out_ch_format_const_layer_format_type, _Relu_output_0_t_out_0_ptr_s8, _Relu_output_0_t_out_0_shape_w_const_u16, _Relu_output_0_t_out_0_shape_h_const_u16, 1, 548, _Relu_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) _Relu_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _MaxPool_output_0_DequantizeLinear_Output */
  {
      const ai_i8* _MaxPool_output_0_DequantizeLinear_Output_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 44720);
    ai_float* _MaxPool_output_0_DequantizeLinear_Output_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 30320);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(21, 1, {(stai_ptr) _MaxPool_output_0_DequantizeLinear_Output_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_MaxPool_output_0_DequantizeLinear_Output_t_in_0_ptr_const_s8, _MaxPool_output_0_DequantizeLinear_Output_t_out_0_ptr_f32, _MaxPool_output_0_DequantizeLinear_Output_t_out_0_shape_h_w_ch_d_prod_const_u32, _MaxPool_output_0_DequantizeLinear_Output_t_in_0_fmt_scale_const_f32, _MaxPool_output_0_DequantizeLinear_Output_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(21, 1, {(stai_ptr) _MaxPool_output_0_DequantizeLinear_Output_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _MaxPool_output_0_DequantizeLinear_Output */
  /* LITE_KERNEL_SECTION BEGIN _Relu_1_output_0 */
  {
      const ai_float* _Relu_1_output_0_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 30320);
    ai_float* _Relu_1_output_0_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 18192);
    const ai_u8* _Relu_1_output_0_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208);
    const ai_u8* _Relu_1_output_0_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 14032);
    ai_float* _Relu_1_output_0_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 49520);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, {(stai_ptr) _Relu_1_output_0_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(_Relu_1_output_0_t_in_0_ptr_const_f32, _Relu_1_output_0_t_out_0_ptr_f32, _Relu_1_output_0_t_weight_0_ptr_const_u8, _Relu_1_output_0_t_weight_1_ptr_const_u8, _Relu_1_output_0_t_scratch_0_ptr_f32, _Relu_1_output_0_t_in_0_shape_ch_const_u32, _Relu_1_output_0_t_out_0_shape_ch_const_u32, _Relu_1_output_0_t_in_0_shape_w_const_u32, _Relu_1_output_0_t_in_0_shape_h_const_u32, _Relu_1_output_0_t_out_0_shape_w_const_u32, _Relu_1_output_0_t_out_0_shape_h_const_u32, _Relu_1_output_0_t_weight_0_shape_w_const_u32, _Relu_1_output_0_t_weight_0_shape_h_const_u32, _Relu_1_output_0_l_pad_W_0_const_s32, _Relu_1_output_0_l_pad_H_0_const_s32, _Relu_1_output_0_l_stride_1_const_u16, _Relu_1_output_0_l_stride_0_const_u16, 3, 3, _Relu_1_output_0_l_dilation_H_const_u16, _Relu_1_output_0_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, {(stai_ptr) _Relu_1_output_0_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _Relu_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_2_output_0 */
  {
      const ai_float* _Relu_2_output_0_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 18192);
    ai_float* _Relu_2_output_0_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_u8* _Relu_2_output_0_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 14128);
    const ai_u8* _Relu_2_output_0_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 45232);
    ai_float* _Relu_2_output_0_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 46992);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 1, {(stai_ptr) _Relu_2_output_0_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(_Relu_2_output_0_t_in_0_ptr_const_f32, _Relu_2_output_0_t_out_0_ptr_f32, _Relu_2_output_0_t_weight_0_ptr_const_u8, _Relu_2_output_0_t_weight_1_ptr_const_u8, _Relu_2_output_0_t_scratch_0_ptr_f32, _Relu_2_output_0_t_in_0_shape_ch_const_u32, _Relu_2_output_0_t_out_0_shape_ch_const_u32, _Relu_2_output_0_t_in_0_shape_w_const_u32, _Relu_2_output_0_t_in_0_shape_h_const_u32, _Relu_2_output_0_t_out_0_shape_w_const_u32, _Relu_2_output_0_t_out_0_shape_h_const_u32, _Relu_2_output_0_t_weight_0_shape_w_const_u32, _Relu_2_output_0_t_weight_0_shape_h_const_u32, _Relu_2_output_0_l_pad_W_0_const_s32, _Relu_2_output_0_l_pad_H_0_const_s32, _Relu_2_output_0_l_stride_1_const_u16, _Relu_2_output_0_l_stride_0_const_u16, 3, 3, _Relu_2_output_0_l_dilation_H_const_u16, _Relu_2_output_0_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, {(stai_ptr) _Relu_2_output_0_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _Relu_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_3_output_0 */
  {
      const ai_float* _Relu_3_output_0_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_float* _Relu_3_output_0_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 44496);
    const ai_u8* _Relu_3_output_0_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 45376);
    const ai_u8* _Relu_3_output_0_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 115360);
    ai_float* _Relu_3_output_0_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 43200);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) _Relu_3_output_0_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(_Relu_3_output_0_t_in_0_ptr_const_f32, _Relu_3_output_0_t_out_0_ptr_f32, _Relu_3_output_0_t_weight_0_ptr_const_u8, _Relu_3_output_0_t_weight_1_ptr_const_u8, _Relu_3_output_0_t_scratch_0_ptr_f32, _Relu_3_output_0_t_in_0_shape_ch_const_u32, _Relu_3_output_0_t_out_0_shape_ch_const_u32, _Relu_3_output_0_t_in_0_shape_w_const_u32, _Relu_3_output_0_t_in_0_shape_h_const_u32, _Relu_3_output_0_t_out_0_shape_w_const_u32, _Relu_3_output_0_t_out_0_shape_h_const_u32, _Relu_3_output_0_t_weight_0_shape_w_const_u32, _Relu_3_output_0_t_weight_0_shape_h_const_u32, _Relu_3_output_0_l_pad_W_0_const_s32, _Relu_3_output_0_l_pad_H_0_const_s32, _Relu_3_output_0_l_stride_1_const_u16, _Relu_3_output_0_l_stride_0_const_u16, 3, 3, _Relu_3_output_0_l_dilation_H_const_u16, _Relu_3_output_0_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) _Relu_3_output_0_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _Relu_3_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_4_output_0 */
  {
      const ai_float* _Relu_4_output_0_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 44496);
    ai_float* _Relu_4_output_0_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 3564);
    const ai_u8* _Relu_4_output_0_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 115576);
    const ai_u8* _Relu_4_output_0_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 273040);
    ai_float* _Relu_4_output_0_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, {(stai_ptr) _Relu_4_output_0_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(_Relu_4_output_0_t_in_0_ptr_const_f32, _Relu_4_output_0_t_out_0_ptr_f32, _Relu_4_output_0_t_weight_0_ptr_const_u8, _Relu_4_output_0_t_weight_1_ptr_const_u8, _Relu_4_output_0_t_scratch_0_ptr_f32, _Relu_4_output_0_t_in_0_shape_ch_const_u32, _Relu_4_output_0_t_out_0_shape_ch_const_u32, _Relu_4_output_0_t_in_0_shape_w_const_u32, _Relu_4_output_0_t_in_0_shape_h_const_u32, _Relu_4_output_0_t_out_0_shape_w_const_u32, _Relu_4_output_0_t_out_0_shape_h_const_u32, _Relu_4_output_0_t_weight_0_shape_w_const_u32, _Relu_4_output_0_t_weight_0_shape_h_const_u32, _Relu_4_output_0_l_pad_W_0_const_s32, _Relu_4_output_0_l_pad_H_0_const_s32, _Relu_4_output_0_l_stride_1_const_u16, _Relu_4_output_0_l_stride_0_const_u16, 3, 3, _Relu_4_output_0_l_dilation_H_const_u16, _Relu_4_output_0_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, {(stai_ptr) _Relu_4_output_0_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _Relu_4_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion */
  {
      const ai_float* _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 3564);
    ai_i8* _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3564);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, {(stai_ptr) _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_in_0_ptr_const_f32, _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_ptr_s8, _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_fmt_scale_const_f32, _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, {(stai_ptr) _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _Relu_4_output_0_0_0__Flatten_output_0_to_chlast_conversion */
  /* LITE_KERNEL_SECTION BEGIN _Flatten_output_0_to_chlast */
  {
    
  forward_lite__Flatten_output_0_to_chlast(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Flatten_output_0_to_chlast */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
      ai_i8* output_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[0] + 0);
    const ai_i8* output_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 27864);
    const ai_i8* output_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 273364);
    const ai_i32* output_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 297664);
    ai_i16* output_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 52164);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 1, {(stai_ptr) output_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(output_QuantizeLinear_Input_t_out_0_ptr_s8, output_QuantizeLinear_Input_t_in_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_1_ptr_const_s32, output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32, output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, {(stai_ptr) output_QuantizeLinear_Input_t_out_0_ptr_s8});
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
  net_ctx->_inputs[0] = activations[0] + 44120;

  net_ctx->_outputs[0] = activations[0] + 0;
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

