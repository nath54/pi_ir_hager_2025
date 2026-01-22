/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-22T15:25:47+0000
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
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x029770423a520f17e46922a46fdbcd54"
#define _STAI_NETWORK_DATETIME            "2026-01-22T15:25:47+0000"
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
      STAI_DECLARE_ARRAY(float, 1, 0.003921019844710827f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.00019271689234301448f),
      STAI_DECLARE_ARRAY(int16_t, 1, 127)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 1792),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 19140),
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
AI_INTQ_INFO_LIST_OBJ_DECLARE(_MatMul_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0076378085650503635f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0042329211719334126f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_5_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023861019872128963f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0012928383657708764f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003593045985326171f, 0.0003592711000237614f, 0.0003585994418244809f, 0.00035791355185210705f, 0.0003583494108170271f, 0.0003586055536288768f, 0.00035744643537327647f, 0.0003592558787204325f, 0.0003579722251743078f, 0.00035891254083253443f, 0.00035927671706303954f, 0.0003590664709918201f, 0.00035912179737351835f, 0.0003586976381484419f, 0.0003587708924897015f, 0.00035882130032405257f, 0.0003592429275158793f, 0.00035902156378142536f, 0.00035832819412462413f, 0.00035924059920944273f, 0.0003588753461372107f, 0.0003591658896766603f, 0.00035844603553414345f, 0.00035809268592856824f, 0.0003583032521419227f, 0.00035875747562386096f, 0.00035683432361111045f, 0.0003586440288927406f, 0.00035854175803251565f, 0.0003586403909139335f, 0.0003579224576242268f, 0.00035914950422011316f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008705024956725538f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013676044763997197f, 0.0013184347189962864f, 0.0013196668587625027f, 0.0012691066367551684f, 0.0013730151113122702f, 0.0013659351971000433f, 0.0012558589223772287f, 0.00135115219745785f, 0.0013214591890573502f, 0.0013315805699676275f, 0.001231309026479721f, 0.0013794087572023273f, 0.0013623811537399888f, 0.0013910540146753192f, 0.0013879211619496346f, 0.0013640589313581586f, 0.001269509200938046f, 0.0013835217105224729f, 0.0013350775698199868f, 0.001288577332161367f, 0.0013595342170447111f, 0.0013785966439172626f, 0.0013183841947466135f, 0.0013660097029060125f, 0.0013914508745074272f, 0.001268340740352869f, 0.0013815625570714474f, 0.0013550507137551904f, 0.0013361521996557713f, 0.001361445407383144f, 0.001381464535370469f, 0.0013886223314329982f, 0.0013753711245954037f, 0.0013500553322955966f, 0.0013468468096107244f, 0.0013896558666601777f, 0.0013208042364567518f, 0.0012800585245713592f, 0.0013415097491815686f, 0.0013567192945629358f, 0.0013823220506310463f, 0.0013810162199661136f, 0.001358418958261609f, 0.0013807647628709674f, 0.0013615196803584695f, 0.001366073964163661f, 0.0013852567644789815f, 0.0013757169945165515f, 0.0012975302524864674f, 0.0013643030542880297f, 0.0013647072482854128f, 0.0013723440933972597f, 0.0013910307316109538f, 0.0013711807550862432f, 0.0013846419751644135f, 0.0013773603131994605f, 0.0012333551421761513f, 0.0012820695992559195f, 0.0012842853320762515f, 0.0013591452734544873f, 0.0013526309048756957f, 0.0013268116163089871f, 0.001343333045952022f, 0.0013114232569932938f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_3_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005235839635133743f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_3_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009822299471125007f, 0.0009606266976334155f, 0.0009665341349318624f, 0.0009778422536328435f, 0.0009832135401666164f, 0.0009835445089265704f, 0.0009548466769047081f, 0.0009318877710029483f, 0.000978151336312294f, 0.0009587274398654699f, 0.0009800875559449196f, 0.0009770527249202132f, 0.0009340447722934186f, 0.0009681695373728871f, 0.0009254464530386031f, 0.0009703052928671241f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  _MatMul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 480, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 480, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_5_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 15360, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 640, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 352, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_5_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Constant_5_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_5_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 30), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_MatMul_output_0_output_array, &_MatMul_output_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 30), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Relu_output_0_output_array, &_Relu_output_0_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_Relu_1_output_0_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_Relu_1_output_0_output_array, &_Relu_1_output_0_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 640, 1, 1), AI_STRIDE_INIT(4, 2, 2, 1280, 1280),
  1, &_Relu_1_output_0_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_weights, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 480, 32, 1, 1), AI_STRIDE_INIT(4, 1, 480, 15360, 15360),
  1, &_Relu_1_output_0_weights_array, &_Relu_1_output_0_weights_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output0, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 480, 1, 1), AI_STRIDE_INIT(4, 1, 1, 480, 480),
  1, &_Relu_output_0_output_array, &_Relu_output_0_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Relu_2_output_0_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_Relu_2_output_0_output_array, &_Relu_2_output_0_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_scratch0, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 352, 1, 1), AI_STRIDE_INIT(4, 2, 2, 704, 704),
  1, &_Relu_2_output_0_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_weights, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 32, 64, 1, 1), AI_STRIDE_INIT(4, 1, 32, 2048, 2048),
  1, &_Relu_2_output_0_weights_array, &_Relu_2_output_0_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_Relu_3_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Relu_3_output_0_output_array, &_Relu_3_output_0_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &_Relu_3_output_0_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_weights, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 64, 16, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1024, 1024),
  1, &_Relu_3_output_0_weights_array, &_Relu_3_output_0_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Constant_5_output_0_DequantizeLinear_Output_const_3D, &_MatMul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Relu_output_0_layer, 16,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Relu_output_0_chain,
  NULL, &_Relu_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_1_output_0_weights, &_Relu_1_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_1_output_0_layer, 22,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_1_output_0_chain,
  NULL, &_Relu_1_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_2_output_0_weights, &_Relu_2_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_2_output_0_layer, 25,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_2_output_0_chain,
  NULL, &_Relu_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_3_output_0_weights, &_Relu_3_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_3_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_3_output_0_layer, 28,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_3_output_0_chain,
  NULL, &_Relu_3_output_0_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite__Relu_output_0(_stai_network_context* net_ctx)
{
  _Constant_5_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  _Constant_5_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  _MatMul_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 500);
  _MatMul_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 500);
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 2, { _Constant_5_output_0_DequantizeLinear_Output_const_3D.data->data,_MatMul_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, { _Relu_output_0_output.data->data});
}
void forward_lite__Relu_1_output_0(_stai_network_context* net_ctx)
{
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_1_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 240);
  _Relu_1_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 240);
  _Relu_1_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 15600);
  _Relu_1_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 15600);
  _Relu_1_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 480);
  _Relu_1_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 480);
  _Relu_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1760);
  _Relu_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1760);
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, { _Relu_output_0_output0.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_1_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, { _Relu_1_output_0_output.data->data});
}
void forward_lite__Relu_2_output_0(_stai_network_context* net_ctx)
{
  _Relu_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1760);
  _Relu_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1760);
  _Relu_2_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 15728);
  _Relu_2_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 15728);
  _Relu_2_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 17776);
  _Relu_2_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 17776);
  _Relu_2_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_2_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 704);
  _Relu_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 704);
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, { _Relu_1_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_2_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, { _Relu_2_output_0_output.data->data});
}
void forward_lite__Relu_3_output_0(_stai_network_context* net_ctx)
{
  _Relu_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 704);
  _Relu_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 704);
  _Relu_3_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 18032);
  _Relu_3_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 18032);
  _Relu_3_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 19056);
  _Relu_3_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 19056);
  _Relu_3_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_3_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 288);
  _Relu_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 288);
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 1, { _Relu_2_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_3_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, { _Relu_3_output_0_output.data->data});
}

/*****************************************************************************/


static const ai_u16 _MatMul_output_0_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 _MatMul_output_0_t_in_0_shape_h_const_u16 = 30;
static const ai_u16 _MatMul_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _MatMul_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _MatMul_output_0_t_in_0_shape_ch_const_u16 = 10;
static const ai_u16 _MatMul_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 _MatMul_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _MatMul_output_0_t_out_0_fmt_zero_const_s8 = 23;
static const ai_float _MatMul_output_0_t_in_0_fmt_scale_const_f32 = 0.003921019844710827f;
static const ai_float _MatMul_output_0_t_out_0_fmt_scale_const_f32 = 0.0076378085650503635f;
static const ai_float _MatMul_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002481255680322647f, 0.0021937908604741096f, 0.0023434970062226057f, 0.0024459066335111856f, 0.0021648453548550606f, 0.002357511781156063f, 0.00241601443849504f, 0.0022066703531891108f, 0.002343744272366166f, 0.0023884305264800787f, 0.0024224629160016775f, 0.0021200089249759912f, 0.002360170939937234f, 0.002447280799970031f, 0.002356066135689616f, 0.002479203976690769f);
static const ai_layer_format_type _MatMul_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;





static const ai_i8 output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = 127;
static const ai_u16 output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 1;
static const ai_u32 output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32 = 1;
static const ai_float output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0005235839635133743f;
static const ai_float output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.00019271689234301448f;
static const ai_float output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32 = 0.0018922205781564116f;
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


  /* LITE_KERNEL_SECTION BEGIN _MatMul_output_0 */
  {
      const ai_i8* _MatMul_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* _MatMul_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 16);
    const ai_i32* _MatMul_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 176);
    ai_i8* _MatMul_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 500);
    ai_i16* _MatMul_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 300);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) _MatMul_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_MatMul_output_0_t_in_0_ptr_const_s8, _MatMul_output_0_t_in_0_shape_w_const_u16, _MatMul_output_0_t_in_0_shape_h_const_u16, _MatMul_output_0_l_stride_1_const_u16, _MatMul_output_0_l_stride_0_const_u16, _MatMul_output_0_t_in_0_shape_ch_const_u16, _MatMul_output_0_t_weight_0_ptr_const_s8, _MatMul_output_0_t_out_0_shape_ch_const_u16, _MatMul_output_0_t_weight_1_ptr_const_s32, _MatMul_output_0_t_in_0_fmt_zero_const_s8, _MatMul_output_0_t_out_0_fmt_zero_const_s8, _MatMul_output_0_t_in_0_fmt_scale_const_f32, _MatMul_output_0_t_out_0_fmt_scale_const_f32, _MatMul_output_0_t_weight_0_fmt_scale_const_f32, _MatMul_output_0_l_out_ch_format_const_layer_format_type, _MatMul_output_0_t_out_0_ptr_s8, 1, 200, _MatMul_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) _MatMul_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _MatMul_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_output_0 */
  {
    
  forward_lite__Relu_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_1_output_0 */
  {
    
  forward_lite__Relu_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_2_output_0 */
  {
    
  forward_lite__Relu_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_3_output_0 */
  {
    
  forward_lite__Relu_3_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_3_output_0 */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
      ai_i8* output_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[0] + 0);
    const ai_i8* output_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 288);
    const ai_i8* output_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 19120);
    const ai_i32* output_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 19136);
    ai_i16* output_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(31, 1, {(stai_ptr) output_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(output_QuantizeLinear_Input_t_out_0_ptr_s8, output_QuantizeLinear_Input_t_in_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_1_ptr_const_s32, output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32, output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(31, 1, {(stai_ptr) output_QuantizeLinear_Input_t_out_0_ptr_s8});
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
  net_ctx->_inputs[0] = activations[0] + 0;

  net_ctx->_outputs[0] = activations[0] + 32;
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

