/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-22T15:31:23+0000
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
#define _STAI_NETWORK_MODEL_SIGNATURE     "0xecfba01143f6a55e53ff30e9443dc45c"
#define _STAI_NETWORK_DATETIME            "2026-01-22T15:31:23+0000"
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
      STAI_DECLARE_ARRAY(float, 1, 0.003921113442629576f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.0006810501799918711f),
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
      STAI_DECLARE_ARRAY(int32_t, 1, 37956),
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
    AI_PACK_INTQ_SCALE(0.007366224192082882f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003700567176565528f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_8_output_0_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0024228354450315237f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010575256310403347f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00035916277556680143f, 0.000359352066880092f, 0.00035920264781452715f, 0.0003592076536733657f, 0.0003588491235859692f, 0.00035762839252129197f, 0.0003582143399398774f, 0.0003593942674342543f, 0.00035938419750891626f, 0.00035881847725249827f, 0.00035920526715926826f, 0.0003591268614400178f, 0.00035931335878558457f, 0.00035930320154875517f, 0.00035817292518913746f, 0.0003585616941563785f, 0.000357973447535187f, 0.00035910200676880777f, 0.00035874609602615237f, 0.0003589079715311527f, 0.00035864257370121777f, 0.00035827705869451165f, 0.0003587129176594317f, 0.00035925867268815637f, 0.00035933280014432967f, 0.0003588736290112138f, 0.0003586164384614676f, 0.0003587680112104863f, 0.00035889114951714873f, 0.00035708112409338355f, 0.00035885791294276714f, 0.00035896539338864386f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000998710747808218f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001288631116040051f, 0.0013504744274541736f, 0.001345379976555705f, 0.001389033393934369f, 0.001238746684975922f, 0.0012597825843840837f, 0.0013406788930296898f, 0.001383638707920909f, 0.001390816061757505f, 0.0013839808525517583f, 0.0013864743523299694f, 0.0013854611897841096f, 0.001325584831647575f, 0.0013512329896911979f, 0.0013461726484820247f, 0.0013159048976376653f, 0.001345390803180635f, 0.0013825849164277315f, 0.0013598839286714792f, 0.0013367378851398826f, 0.0013882593484595418f, 0.0013839462772011757f, 0.001338256406597793f, 0.0012517640134319663f, 0.001387560274451971f, 0.0013550107832998037f, 0.0013604735722765326f, 0.001381974550895393f, 0.0013500504428520799f, 0.0013334605609998107f, 0.0013072899309918284f, 0.0013854179996997118f, 0.0013045992236584425f, 0.001389079843647778f, 0.0013143265387043357f, 0.001236362848430872f, 0.0013451164122670889f, 0.0013289882335811853f, 0.0013842310290783644f, 0.0013175195781514049f, 0.0013029330875724554f, 0.001364521449431777f, 0.0013773903483524919f, 0.0013759101275354624f, 0.0013860076433047652f, 0.0013078553602099419f, 0.0013844934292137623f, 0.0013828796800225973f, 0.001383092487230897f, 0.0013354072580114007f, 0.001130343647673726f, 0.0012728467117995024f, 0.001333787338808179f, 0.001362256589345634f, 0.0012912299716845155f, 0.0013668921310454607f, 0.0013645520666614175f, 0.001327543519437313f, 0.0013580152299255133f, 0.001387732452712953f, 0.0013569631846621633f, 0.0013794433325529099f, 0.0013395053101703525f, 0.0013848836533725262f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_3_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007493404555134475f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_3_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009842145955190063f, 0.0009474687394686043f, 0.0009560093167237937f, 0.0009751216857694089f, 0.0009695093031041324f, 0.0009180743945762515f, 0.0009798700921237469f, 0.0009821307612583041f, 0.000982380472123623f, 0.0009807681199163198f, 0.0009512828546576202f, 0.0009695324115455151f, 0.0009625433012843132f, 0.0009703165851533413f, 0.0009503593319095671f, 0.0009698268258944154f, 0.0009800153784453869f, 0.0009826173773035407f, 0.0009700484806671739f, 0.0009708402212709188f, 0.0009672470623627305f, 0.0009754085331223905f, 0.0009765790309756994f, 0.0009801783598959446f, 0.0009795846417546272f, 0.0009717139764688909f, 0.000981891411356628f, 0.0009797115344554186f, 0.0009797710226848722f, 0.0009723493712954223f, 0.0009828354232013226f, 0.0009747696458362043f, 0.0009618687909096479f, 0.0009805085137486458f, 0.000961428799200803f, 0.0009634196758270264f, 0.0009455218678340316f, 0.0009529360686428845f, 0.0009634090238250792f, 0.0009834154043346643f, 0.000982428784482181f, 0.0009642983786761761f, 0.0009580411715433002f, 0.0009682350791990757f, 0.0009768041782081127f, 0.0009510774980299175f, 0.000983766047284007f, 0.0009521716274321079f, 0.0009738939115777612f, 0.0009824192384257913f, 0.0009804950095713139f, 0.0009841009741649032f, 0.0009567369124852121f, 0.0009723154362291098f, 0.000977422809228301f, 0.0009755682549439371f, 0.0009648441919125617f, 0.0009815461235120893f, 0.0009710242156870663f, 0.0009612428257241845f, 0.0009797185193747282f, 0.0009369756444357336f, 0.0009651092696003616f, 0.0009742300608195364f, 0.000983073958195746f, 0.0009723392431624234f, 0.0009743176051415503f, 0.0009537139558233321f, 0.000980216427706182f, 0.0009590036352165043f, 0.0009516916470602155f, 0.0009827680187299848f, 0.0009740461828187108f, 0.0009823631262406707f, 0.0009716649656184018f, 0.0009636233444325626f, 0.0009566380176693201f, 0.0009707430726848543f, 0.0009829518385231495f, 0.0009415596723556519f, 0.0009829356567934155f, 0.0009411248611286283f, 0.0009709095465950668f, 0.0009718614746816456f, 0.0009576449519954622f, 0.0009818174876272678f, 0.0009708935976959765f, 0.0009825964225456119f, 0.0009708108846098185f, 0.0009692731546238065f, 0.0009823705768212676f, 0.0009780852124094963f, 0.0009208049159497023f, 0.0009829170303419232f, 0.0009798690443858504f, 0.0008847136050462723f, 0.0009777494706213474f, 0.000976255105342716f, 0.0009499857551418245f, 0.0009778981329873204f, 0.0009720333619043231f, 0.0009778158273547888f, 0.0009676159243099391f, 0.0009689686703495681f, 0.0009801345877349377f, 0.0009818526450544596f, 0.00098320038523525f, 0.0009673350723460317f, 0.0009643050725571811f, 0.0009668141137808561f, 0.0009739380329847336f, 0.0009796154918149114f, 0.0009716429049149156f, 0.0009533112752251327f, 0.0009670056751929224f, 0.0009647445986047387f, 0.0009801527485251427f, 0.0009764279238879681f, 0.0009810851188376546f, 0.0009487834759056568f, 0.0009825421730056405f, 0.0009388771140947938f, 0.0009664776735007763f, 0.0009451162186451256f, 0.0009573862189427018f, 0.0009784104768186808f, 0.000977043411694467f, 0.000980839366093278f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_4_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005026027793064713f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_4_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006676659104414284f, 0.0006939632585272193f, 0.0006805300363339484f, 0.0006868985365144908f, 0.0006959046004340053f, 0.0006870399811305106f, 0.0006938592414371669f, 0.0006940306629985571f, 0.0006941314786672592f, 0.0006922787870280445f, 0.0006953761330805719f, 0.0006873599486425519f, 0.0006878112908452749f, 0.0006925181369297206f, 0.0006934152916073799f, 0.0006908977520652115f, 0.0006891212542541325f, 0.0006836220854893327f, 0.0006917084683664143f, 0.0006938164588063955f, 0.000687271065544337f, 0.0006958621088415384f, 0.0006846309988759458f, 0.0006878343410789967f, 0.000694568851031363f, 0.0006847378681413829f, 0.0006757067167200148f, 0.000688043306581676f, 0.0006940221646800637f, 0.0006949006929062307f, 0.0006944140768609941f, 0.0006859441054984927f, 0.000686274841427803f, 0.0006951986579224467f, 0.0006926418282091618f, 0.0006937452708370984f, 0.0006913599208928645f, 0.0006943722255527973f, 0.0006925153429619968f, 0.00068674172507599f, 0.0006893527461215854f, 0.000691380409989506f, 0.0006850850768387318f, 0.0006900227745063603f, 0.0006948876543901861f, 0.0006956254364922643f, 0.0006901518208906054f, 0.000691190070938319f, 0.000694887014105916f, 0.0006952033145353198f, 0.0006928932853043079f, 0.0006933195400051773f, 0.000682699668686837f, 0.0006802377756685019f, 0.0006954155978746712f, 0.0006925431080162525f, 0.0006906140479259193f, 0.000692675937898457f, 0.0006921943859197199f, 0.0006873097154311836f, 0.0006832168437540531f, 0.000695510592777282f, 0.0006946613430045545f, 0.0006875781691633165f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_5_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004808547964785248f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_5_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009764356655068696f, 0.0009742432157509029f, 0.000983654404990375f, 0.0009541113977320492f, 0.0009684969554655254f, 0.000976872630417347f, 0.0009794009383767843f, 0.0009579616016708314f, 0.0009759358363226056f, 0.0009525155182927847f, 0.0009530630195513368f, 0.0009808394825085998f, 0.0009680237853899598f, 0.0009763253619894385f, 0.0009713314939290285f, 0.0009739755769260228f, 0.000978129799477756f, 0.0009429326746612787f, 0.0009810362244024873f, 0.000984211452305317f, 0.0009828077163547277f, 0.0009671690058894455f, 0.0009771680925041437f, 0.0009804465807974339f, 0.0009560796315781772f, 0.0009312116890214384f, 0.000983802368864417f, 0.0009353776695206761f, 0.000981619581580162f, 0.000967318715993315f, 0.0009620418422855437f, 0.0009818299440667033f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_6_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007091359002515674f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_6_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013854348799213767f, 0.0013459547189995646f, 0.0013527701376006007f, 0.0013173014158383012f, 0.0013320837169885635f, 0.0013394339475780725f, 0.0013189691817387938f, 0.0013669091276824474f, 0.001351822167634964f, 0.0013745317701250315f, 0.0013912293361499906f, 0.0013516368344426155f, 0.0013397426810115576f, 0.0013807278592139482f, 0.0013427810044959188f, 0.0013842214830219746f),
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
  _Constant_8_output_0_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
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
  NULL, NULL, 64, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_4_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 448, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_5_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 224, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_6_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_6_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_6_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_6_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 112, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_8_output_0_DequantizeLinear_Output_const_3D, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Constant_8_output_0_DequantizeLinear_Output_const_3D_array, &_Constant_8_output_0_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _MatMul_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 30), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_MatMul_output_0_output_array, &_MatMul_output_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output, AI_STATIC,
  29, 0x1,
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
  30, 0x1,
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
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Relu_4_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_Relu_4_output_0_output_array, &_Relu_4_output_0_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_scratch0, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 448, 1, 1), AI_STRIDE_INIT(4, 2, 2, 896, 896),
  1, &_Relu_4_output_0_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_4_output_0_weights, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 128, 64, 1, 1), AI_STRIDE_INIT(4, 1, 128, 8192, 8192),
  1, &_Relu_4_output_0_weights_array, &_Relu_4_output_0_weights_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_bias, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_Relu_5_output_0_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_Relu_5_output_0_output_array, &_Relu_5_output_0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_scratch0, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 224, 1, 1), AI_STRIDE_INIT(4, 2, 2, 448, 448),
  1, &_Relu_5_output_0_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_5_output_0_weights, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 64, 32, 1, 1), AI_STRIDE_INIT(4, 1, 64, 2048, 2048),
  1, &_Relu_5_output_0_weights_array, &_Relu_5_output_0_weights_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_6_output_0_bias, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_Relu_6_output_0_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_6_output_0_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_Relu_6_output_0_output_array, &_Relu_6_output_0_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_6_output_0_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 2, 2, 224, 224),
  1, &_Relu_6_output_0_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_6_output_0_weights, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 32, 16, 1, 1), AI_STRIDE_INIT(4, 1, 32, 512, 512),
  1, &_Relu_6_output_0_weights_array, &_Relu_6_output_0_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Constant_8_output_0_DequantizeLinear_Output_const_3D, &_MatMul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Relu_output_0_layer, 22,
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
  _Relu_1_output_0_layer, 28,
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
  _Relu_2_output_0_layer, 31,
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
  _Relu_3_output_0_layer, 34,
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
  _Relu_4_output_0_layer, 37,
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
  _Relu_5_output_0_layer, 40,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_5_output_0_chain,
  NULL, &_Relu_5_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_6_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_5_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_6_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_6_output_0_weights, &_Relu_6_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_6_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_6_output_0_layer, 43,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_6_output_0_chain,
  NULL, &_Relu_6_output_0_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite__Relu_output_0(_stai_network_context* net_ctx)
{
  _Constant_8_output_0_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  _Constant_8_output_0_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  _MatMul_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 800);
  _MatMul_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 800);
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1280);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1280);
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 2, { _Constant_8_output_0_DequantizeLinear_Output_const_3D.data->data,_MatMul_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, { _Relu_output_0_output.data->data});
}
void forward_lite__Relu_1_output_0(_stai_network_context* net_ctx)
{
  _Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1280);
  _Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1280);
  _Relu_1_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 240);
  _Relu_1_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 240);
  _Relu_1_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 15600);
  _Relu_1_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 15600);
  _Relu_1_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_1_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1760);
  _Relu_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1760);
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 1, { _Relu_output_0_output0.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_1_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, { _Relu_1_output_0_output.data->data});
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
  _STAI_NETWORK_EVENT_NODE_START_CB(31, 1, { _Relu_1_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_2_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(31, 1, { _Relu_2_output_0_output.data->data});
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
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, { _Relu_2_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_3_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, { _Relu_3_output_0_output.data->data});
}
void forward_lite__Relu_4_output_0(_stai_network_context* net_ctx)
{
  _Relu_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  _Relu_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _Relu_4_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 26736);
  _Relu_4_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 26736);
  _Relu_4_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 34928);
  _Relu_4_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 34928);
  _Relu_4_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_4_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_4_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 896);
  _Relu_4_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 896);
  _STAI_NETWORK_EVENT_NODE_START_CB(37, 1, { _Relu_3_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_4_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(37, 1, { _Relu_4_output_0_output.data->data});
}
void forward_lite__Relu_5_output_0(_stai_network_context* net_ctx)
{
  _Relu_4_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 896);
  _Relu_4_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 896);
  _Relu_5_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 35184);
  _Relu_5_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 35184);
  _Relu_5_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 37232);
  _Relu_5_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 37232);
  _Relu_5_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_5_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_5_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 448);
  _Relu_5_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 448);
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, { _Relu_4_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_5_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, { _Relu_5_output_0_output.data->data});
}
void forward_lite__Relu_6_output_0(_stai_network_context* net_ctx)
{
  _Relu_5_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 448);
  _Relu_5_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 448);
  _Relu_6_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 37360);
  _Relu_6_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 37360);
  _Relu_6_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 37872);
  _Relu_6_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 37872);
  _Relu_6_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_6_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _Relu_6_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 224);
  _Relu_6_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 224);
  _STAI_NETWORK_EVENT_NODE_START_CB(43, 1, { _Relu_5_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&_Relu_6_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(43, 1, { _Relu_6_output_0_output.data->data});
}

/*****************************************************************************/


static const ai_u16 _MatMul_output_0_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 _MatMul_output_0_t_in_0_shape_h_const_u16 = 30;
static const ai_u16 _MatMul_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _MatMul_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _MatMul_output_0_t_in_0_shape_ch_const_u16 = 10;
static const ai_u16 _MatMul_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 _MatMul_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _MatMul_output_0_t_out_0_fmt_zero_const_s8 = 11;
static const ai_float _MatMul_output_0_t_in_0_fmt_scale_const_f32 = 0.003921113442629576f;
static const ai_float _MatMul_output_0_t_out_0_fmt_scale_const_f32 = 0.007366224192082882f;
static const ai_float _MatMul_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0024451599456369877f, 0.002179856412112713f, 0.0024677265901118517f, 0.0014634841354563832f, 0.0024138663429766893f, 0.0024172156117856503f, 0.0021222904324531555f, 0.002457189606502652f, 0.0021221071947366f, 0.0024895102251321077f, 0.0022399239242076874f, 0.0024467722978442907f, 0.002438352443277836f, 0.0023522183764725924f, 0.002461843891069293f, 0.0024502016603946686f);
static const ai_layer_format_type _MatMul_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;








static const ai_i8 output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = -128;
static const ai_u16 output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 1;
static const ai_u32 output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32 = 1;
static const ai_float output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0007091359002515674f;
static const ai_float output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.0006810501799918711f;
static const ai_float output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32 = 0.0019416813738644123f;
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
    ai_i8* _MatMul_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 800);
    ai_i16* _MatMul_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1580);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(19, 1, {(stai_ptr) _MatMul_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_MatMul_output_0_t_in_0_ptr_const_s8, _MatMul_output_0_t_in_0_shape_w_const_u16, _MatMul_output_0_t_in_0_shape_h_const_u16, _MatMul_output_0_l_stride_1_const_u16, _MatMul_output_0_l_stride_0_const_u16, _MatMul_output_0_t_in_0_shape_ch_const_u16, _MatMul_output_0_t_weight_0_ptr_const_s8, _MatMul_output_0_t_out_0_shape_ch_const_u16, _MatMul_output_0_t_weight_1_ptr_const_s32, _MatMul_output_0_t_in_0_fmt_zero_const_s8, _MatMul_output_0_t_out_0_fmt_zero_const_s8, _MatMul_output_0_t_in_0_fmt_scale_const_f32, _MatMul_output_0_t_out_0_fmt_scale_const_f32, _MatMul_output_0_t_weight_0_fmt_scale_const_f32, _MatMul_output_0_l_out_ch_format_const_layer_format_type, _MatMul_output_0_t_out_0_ptr_s8, 1, 200, _MatMul_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(19, 1, {(stai_ptr) _MatMul_output_0_t_out_0_ptr_s8});
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
  /* LITE_KERNEL_SECTION BEGIN _Relu_6_output_0 */
  {
    
  forward_lite__Relu_6_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Relu_6_output_0 */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
      ai_i8* output_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[0] + 0);
    const ai_i8* output_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 224);
    const ai_i8* output_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 37936);
    const ai_i32* output_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 37952);
    ai_i16* output_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, {(stai_ptr) output_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(output_QuantizeLinear_Input_t_out_0_ptr_s8, output_QuantizeLinear_Input_t_in_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_0_ptr_const_s8, output_QuantizeLinear_Input_t_weight_1_ptr_const_s32, output_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, output_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, output_QuantizeLinear_Input_t_out_0_shape_h_w_prod_const_u32, output_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, output_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, {(stai_ptr) output_QuantizeLinear_Input_t_out_0_ptr_s8});
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
  net_ctx->_inputs[0] = activations[0] + 1280;

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

