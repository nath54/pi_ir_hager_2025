/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-22T14:56:12+0000
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
#define _STAI_NETWORK_MODEL_SIGNATURE     "0xb0c3fb87d767ab542a8da76ba22e12b1"
#define _STAI_NETWORK_DATETIME            "2026-01-22T14:56:12+0000"
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
      STAI_DECLARE_ARRAY(float, 1, 0.003920554183423519f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.0007422412163577974f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
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
      STAI_DECLARE_ARRAY(int32_t, 1, 31556),
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
    AI_PACK_INTQ_SCALE(0.008817403577268124f),
    AI_PACK_INTQ_ZP(19)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002678550546988845f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_7_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023657314013689756f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008682586485520005f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003592477005440742f, 0.0003578253381419927f, 0.00035880462382920086f, 0.00035929580917581916f, 0.0003592298016883433f, 0.0003585113154258579f, 0.00035935791675001383f, 0.00035823616781271994f, 0.00035738511360250413f, 0.00035937430220656097f, 0.0003592902503442019f, 0.00035906239645555615f, 0.0003593754954636097f, 0.00035815994488075376f, 0.0003592307330109179f, 0.0003586857346817851f, 0.0003581410273909569f, 0.00035849487176164985f, 0.00035797044984064996f, 0.0003590533451642841f, 0.00035507077700458467f, 0.0003588899562601f, 0.0003593337896745652f, 0.0003575954760890454f, 0.0003592688881326467f, 0.00035919088986702263f, 0.00035827202373184264f, 0.000359315105015412f, 0.00035925855627283454f, 0.0003590935666579753f, 0.0003593823639675975f, 0.0003593387664295733f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009535410790704191f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013788597425445914f, 0.0013722492149099708f, 0.0013918970944359899f, 0.0013771012891083956f, 0.0013807970099151134f, 0.0013531696749851108f, 0.0013485222589224577f, 0.0013556526973843575f, 0.0013892082497477531f, 0.0013606382999569178f, 0.001384881092235446f, 0.001362545881420374f, 0.0013737346744164824f, 0.001370980404317379f, 0.0013707401230931282f, 0.0013654818758368492f, 0.0013694025110453367f, 0.0013576321071013808f, 0.0013776568230241537f, 0.0013679872499778867f, 0.0013666959712281823f, 0.0013400522293522954f, 0.0013888146495446563f, 0.0013844913337379694f, 0.0013736768160015345f, 0.0013889074325561523f, 0.001337662572041154f, 0.0013871926348656416f, 0.0013588776346296072f, 0.001367133460007608f, 0.0013657278614118695f, 0.0013888322282582521f, 0.0013572047464549541f, 0.0012792871566489339f, 0.001364004798233509f, 0.0012962202308699489f, 0.0013594223419204354f, 0.001346574630588293f, 0.0013440230395644903f, 0.0013033199356868863f, 0.0013824310153722763f, 0.0013865685323253274f, 0.0013690463965758681f, 0.001330552389845252f, 0.0013836477883160114f, 0.0013769459910690784f, 0.0013789546210318804f, 0.001381899113766849f, 0.0013645028229802847f, 0.00136997748631984f, 0.0013427101075649261f, 0.0013910214183852077f, 0.0013875047443434596f, 0.00139031489379704f, 0.001375629915855825f, 0.0013782007154077291f, 0.0013421912444755435f, 0.0013659907272085547f, 0.001371326157823205f, 0.001373184029944241f, 0.0013419410679489374f, 0.0012939473381265998f, 0.001365892356261611f, 0.0012924533803015947f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_3_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010028103133663535f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_3_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009610237320885062f, 0.0009718865621834993f, 0.0009681049850769341f, 0.000912660441827029f, 0.0009802915155887604f, 0.0009576636948622763f, 0.0009430455393157899f, 0.0009780001128092408f, 0.000981165561825037f, 0.0009218777995556593f, 0.0009785277070477605f, 0.0009739076485857368f, 0.0009586212690919638f, 0.0009808515897020698f, 0.0009299602243117988f, 0.0009638260235078633f, 0.0009737894870340824f, 0.0009622484794817865f, 0.0009646086255088449f, 0.0009692168096080422f, 0.0009693583124317229f, 0.000982868019491434f, 0.0009775985963642597f, 0.0009821181884035468f, 0.0009491375531069934f, 0.0009780643740668893f, 0.0009724359260872006f, 0.0009820640552788973f, 0.0009765247232280672f, 0.0009826893219724298f, 0.0009668456623330712f, 0.0009505050838924944f, 0.0009363294811919332f, 0.0009725022246129811f, 0.0009774698410183191f, 0.0009807354072108865f, 0.0009829618502408266f, 0.0009790132753551006f, 0.0009686806006357074f, 0.0009706427226774395f, 0.0009577067685313523f, 0.0009716757340356708f, 0.0009725026902742684f, 0.0009636936592869461f, 0.0009756520157679915f, 0.0009595337323844433f, 0.0009499412844888866f, 0.0009821710409596562f, 0.0009811797644942999f, 0.0009835840901359916f, 0.0009825527667999268f, 0.0009553460404276848f, 0.0009796498343348503f, 0.0009810345945879817f, 0.0009711801540106535f, 0.0009672589949332178f, 0.0009824311127886176f, 0.000966897641774267f, 0.0009014724055305123f, 0.0009673996828496456f, 0.0009840036509558558f, 0.0009422785369679332f, 0.0009711799211800098f, 0.0009676105109974742f, 0.0009648850536905229f, 0.0009703657124191523f, 0.0009760443354025483f, 0.0009759762906469405f, 0.0009749497985467315f, 0.0009828825714066625f, 0.0009533832198940217f, 0.0009579820325598121f, 0.0009561412152834237f, 0.0009808234171941876f, 0.0009756788494996727f, 0.0009661053773015738f, 0.0009595078299753368f, 0.0009619676857255399f, 0.0009826729074120522f, 0.0009708619327284396f, 0.000984120648354292f, 0.0009796336526051164f, 0.0009813468204811215f, 0.0009793243370950222f, 0.0009358348324894905f, 0.0009811841882765293f, 0.0009818726684898138f, 0.0009525776258669794f, 0.000983029487542808f, 0.0009721656097099185f, 0.0009713105973787606f, 0.0009829929331317544f, 0.0009668939746916294f, 0.0009832698851823807f, 0.0009690883453004062f, 0.0009777054656296968f, 0.0009489927906543016f, 0.000982189434580505f, 0.0009623681544326246f, 0.0009650576394051313f, 0.0009582677739672363f, 0.0009830883936956525f, 0.0009742761612869799f, 0.0008840639493428171f, 0.0009820249397307634f, 0.00094947888283059f, 0.0009826959576457739f, 0.0009817596292123199f, 0.0009824007283896208f, 0.0009620188502594829f, 0.0009476332343183458f, 0.0009701085509732366f, 0.0009725872660055757f, 0.0009101542527787387f, 0.000980002456344664f, 0.0009797976817935705f, 0.0009838377591222525f, 0.0009428428020328283f, 0.0009831080678850412f, 0.0009729312150739133f, 0.0009842334548011422f, 0.0009770498145371675f, 0.0009604551596567035f, 0.0009783755522221327f, 0.0009720392408780754f, 0.0009676971239969134f, 0.0009728515287861228f, 0.0009818024700507522f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_4_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006012646481394768f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_4_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006940662278793752f, 0.0006852550432085991f, 0.0006908634095452726f, 0.0006885810289531946f, 0.000689844076987356f, 0.000694882299285382f, 0.0006935868877917528f, 0.000693600217346102f, 0.0006833053776063025f, 0.0006849935161881149f, 0.0006907318020239472f, 0.000686358951497823f, 0.0006931757088750601f, 0.0006925655179657042f, 0.000689424283336848f, 0.0006914489786140621f, 0.0006893218960613012f, 0.0006934692501090467f, 0.0006940673338249326f, 0.0006932247779332101f, 0.0006945460336282849f, 0.0006938597070984542f, 0.0006954394048079848f, 0.0006874909158796072f, 0.0006905542104505002f, 0.0006855747196823359f, 0.0006957409204915166f, 0.0006953396950848401f, 0.0006715297931805253f, 0.000674122478812933f, 0.0006933744880370796f, 0.000694785441737622f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_5_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007524592219851911f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_5_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013877790188416839f, 0.001383492723107338f, 0.0013810863019898534f, 0.0013868563110008836f, 0.0013707805192098022f, 0.0013849292881786823f, 0.0013507427647709846f, 0.001353506464511156f, 0.0013843928463757038f, 0.0013411076506599784f, 0.001381365000270307f, 0.0013578556245192885f, 0.0013524647802114487f, 0.0013248780742287636f, 0.0013787548523396254f, 0.0012504535261541605f),
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
  _Constant_7_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
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
  NULL, NULL, 128, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_3_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 704, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 288, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 112, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_7_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Constant_7_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_7_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 30), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_MatMul_output_0_output_array, &_MatMul_output_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output, AI_STATIC,
  25, 0x1,
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
  26, 0x1,
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
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_Relu_3_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_Relu_3_output_0_output_array, &_Relu_3_output_0_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 704, 1, 1), AI_STRIDE_INIT(4, 2, 2, 1408, 1408),
  1, &_Relu_3_output_0_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_3_output_0_weights, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 64, 128, 1, 1), AI_STRIDE_INIT(4, 1, 64, 8192, 8192),
  1, &_Relu_3_output_0_weights_array, &_Relu_3_output_0_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_bias, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_Relu_4_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_Relu_4_output_0_output_array, &_Relu_4_output_0_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_scratch0, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 288, 1, 1), AI_STRIDE_INIT(4, 2, 2, 576, 576),
  1, &_Relu_4_output_0_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_weights, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 128, 32, 1, 1), AI_STRIDE_INIT(4, 1, 128, 4096, 4096),
  1, &_Relu_4_output_0_weights_array, &_Relu_4_output_0_weights_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_bias, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_Relu_5_output_0_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Relu_5_output_0_output_array, &_Relu_5_output_0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_scratch0, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 2, 2, 224, 224),
  1, &_Relu_5_output_0_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_weights, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 32, 16, 1, 1), AI_STRIDE_INIT(4, 1, 32, 512, 512),
  1, &_Relu_5_output_0_weights_array, &_Relu_5_output_0_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Constant_7_output_0_DequantizeLinear_Output_const_3D, &_MatMul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Relu_output_0_layer, 20,
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
  _Relu_1_output_0_layer, 26,
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
  _Relu_2_output_0_layer, 29,
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
  _Relu_3_output_0_layer, 32,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_3_output_0_chain,
  NULL, &_Relu_3_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_4_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_4_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_4_output_0_weights, &_Relu_4_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_4_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_4_output_0_layer, 35,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_4_output_0_chain,
  NULL, &_Relu_4_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_5_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_4_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_5_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_5_output_0_weights, &_Relu_5_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_5_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_5_output_0_layer, 38,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_5_output_0_chain,
  NULL, &_Relu_5_output_0_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite__Relu_output_0(_stai_network_context* net_ctx)
{
  _Constant_7_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  _Constant_7_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  _MatMul_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 500);
  _MatMul_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 500);
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(20, 2, { _Constant_7_output_0_DequantizeLinear_Output_const_3D.data->data,_MatMul_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(20, 1, { _Relu_output_0_output.data->data});
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
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, { _Relu_output_0_output0.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_1_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { _Relu_1_output_0_output.data->data});
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
  _Relu_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1696);
  _Relu_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1696);
  _STAI_NETWORK_EVENT_NODE_START_CB(29, 1, { _Relu_1_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_2_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(29, 1, { _Relu_2_output_0_output.data->data});
}
void forward_lite__Relu_3_output_0(_stai_network_context* net_ctx)
{
  _Relu_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1696);
  _Relu_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1696);
  _Relu_3_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 18032);
  _Relu_3_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 18032);
  _Relu_3_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 26224);
  _Relu_3_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 26224);
  _Relu_3_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_3_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  _Relu_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 1, { _Relu_2_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_3_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, { _Relu_3_output_0_output.data->data});
}
void forward_lite__Relu_4_output_0(_stai_network_context* net_ctx)
{
  _Relu_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  _Relu_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _Relu_4_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 26736);
  _Relu_4_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 26736);
  _Relu_4_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 30832);
  _Relu_4_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 30832);
  _Relu_4_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_4_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_4_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 576);
  _Relu_4_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 576);
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, { _Relu_3_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_4_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, { _Relu_4_output_0_output.data->data});
}
void forward_lite__Relu_5_output_0(_stai_network_context* net_ctx)
{
  _Relu_4_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 576);
  _Relu_4_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 576);
  _Relu_5_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 30960);
  _Relu_5_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 30960);
  _Relu_5_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 31472);
  _Relu_5_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 31472);
  _Relu_5_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_5_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_5_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 224);
  _Relu_5_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 224);
  _STAI_NETWORK_EVENT_NODE_START_CB(38, 1, { _Relu_4_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_5_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(38, 1, { _Relu_5_output_0_output.data->data});
}

/*****************************************************************************/


static const ai_u16 _MatMul_output_0_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 _MatMul_output_0_t_in_0_shape_h_const_u16 = 30;
static const ai_u16 _MatMul_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _MatMul_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _MatMul_output_0_t_in_0_shape_ch_const_u16 = 10;
static const ai_u16 _MatMul_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 _MatMul_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _MatMul_output_0_t_out_0_fmt_zero_const_s8 = 19;
static const ai_float _MatMul_output_0_t_in_0_fmt_scale_const_f32 = 0.003920554183423519f;
static const ai_float _MatMul_output_0_t_out_0_fmt_scale_const_f32 = 0.008817403577268124f;
static const ai_float _MatMul_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00228742859326303f, 0.002489308826625347f, 0.00246290466748178f, 0.0023513284977525473f, 0.0024505883920937777f, 0.0024827816523611546f, 0.0018668370321393013f, 0.002432584296911955f, 0.0023550051264464855f, 0.0024066262412816286f, 0.00246261665597558f, 0.002161405747756362f, 0.00228102202527225f, 0.0023070024326443672f, 0.002324157627299428f, 0.0022851228713989258f);
static const ai_layer_format_type _MatMul_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;







static const ai_i8 output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = -128;
static const ai_u16 output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 1;
static const ai_u32 output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32 = 1;
static const ai_float output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0007524592219851911f;
static const ai_float output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.0007422412163577974f;
static const ai_float output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32 = 0.0018104941118508577f;
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
  
  _STAI_NETWORK_EVENT_NODE_START_CB(17, 1, {(stai_ptr) _MatMul_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_MatMul_output_0_t_in_0_ptr_const_s8, _MatMul_output_0_t_in_0_shape_w_const_u16, _MatMul_output_0_t_in_0_shape_h_const_u16, _MatMul_output_0_l_stride_1_const_u16, _MatMul_output_0_l_stride_0_const_u16, _MatMul_output_0_t_in_0_shape_ch_const_u16, _MatMul_output_0_t_weight_0_ptr_const_s8, _MatMul_output_0_t_out_0_shape_ch_const_u16, _MatMul_output_0_t_weight_1_ptr_const_s32, _MatMul_output_0_t_in_0_fmt_zero_const_s8, _MatMul_output_0_t_out_0_fmt_zero_const_s8, _MatMul_output_0_t_in_0_fmt_scale_const_f32, _MatMul_output_0_t_out_0_fmt_scale_const_f32, _MatMul_output_0_t_weight_0_fmt_scale_const_f32, _MatMul_output_0_l_out_ch_format_const_layer_format_type, _MatMul_output_0_t_out_0_ptr_s8, 1, 200, _MatMul_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(17, 1, {(stai_ptr) _MatMul_output_0_t_out_0_ptr_s8});
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
  /* LITE_KERNEL_SECTION BEGIN _Relu_4_output_0 */
  {
    
  forward_lite__Relu_4_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_4_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Relu_5_output_0 */
  {
    
  forward_lite__Relu_5_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_5_output_0 */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
      ai_i8* output_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[0] + 0);
    const ai_i8* output_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 224);
    const ai_i8* output_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 31536);
    const ai_i32* output_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 31552);
    ai_i16* output_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(41, 1, {(stai_ptr) output_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(output_QuantizeLinear_Input_t_out_0_ptr_s8, output_QuantizeLinear_Input_t_in_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_1_ptr_const_s32, output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32, output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(41, 1, {(stai_ptr) output_QuantizeLinear_Input_t_out_0_ptr_s8});
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

