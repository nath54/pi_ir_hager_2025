#pragma once
// Portable C tensor definitions suitable for MCU/desktop backends.
// Default layout choices target CMSIS-NN compatibility (NHWC for 4D tensors).

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Configuration
#ifndef TENSOR_MAX_DIMS
#define TENSOR_MAX_DIMS 5  // e.g., [N,H,W,C] or generic up to 5D
#endif

// If defined non-zero, use 16-bit shape/stride where possible to save RAM
#ifndef TENSOR_USE_16BIT_SHAPE
#define TENSOR_USE_16BIT_SHAPE 0
#endif

#if TENSOR_USE_16BIT_SHAPE
typedef uint16_t tensor_index_t;
typedef uint16_t tensor_stride_t;  // stride in elements
#else
typedef uint32_t tensor_index_t;
typedef uint32_t tensor_stride_t;  // stride in elements
#endif

// Element data type
typedef enum {
    TENSOR_DTYPE_F32 = 0,
    TENSOR_DTYPE_I8  = 1,
    TENSOR_DTYPE_I16 = 2,
    TENSOR_DTYPE_U8  = 3
} tensor_dtype_t;

// Memory layout hint (semantic order for dims[]) for common cases
typedef enum {
    // Generic row-major (last dim contiguous). dims[] interpreted by op
    TENSOR_LAYOUT_ROW_MAJOR = 0,
    // 4D image tensors as [N, H, W, C] (CMSIS-NN friendly)
    TENSOR_LAYOUT_NHWC = 1,
    // 4D image tensors as [N, C, H, W]
    TENSOR_LAYOUT_NCHW = 2
} tensor_layout_t;

// Quantisation parameters (optional)
typedef struct {
    // If true, this tensor uses quantised representation
    uint8_t is_quantized;

    // Per-tensor (asymmetric or symmetric) params
    float scale;           // scale for dequant: real = scale * (q - zero_point)
    int32_t zero_point;    // usually in [-128,127] for int8; 0 for symmetric

    // Optional per-channel params (e.g., weights). If non-null, overrides per-tensor
    const float *per_channel_scales;     // length = channels
    const int32_t *per_channel_zp;       // optional, length = channels
    tensor_index_t per_channel_axis;     // which axis is channel (e.g., C)

    // Optional precomputed fixed-point requant params used by some backends (e.g., CMSIS-NN)
    // If provided, length matches channels for per-channel or 1 for per-tensor
    const int32_t *requant_multipliers;  // Q31 multiplier(s)
    const int32_t *requant_shifts;       // right shift(s), typically in [0,31]
} tensor_quant_params_t;

// Tensor view/handle. Does not own memory.
typedef struct {
    void *data;                         // data pointer (element type given by dtype)
    tensor_dtype_t dtype;               // element type
    tensor_layout_t layout;             // layout hint

    uint8_t dims_count;                 // number of active dims (<= TENSOR_MAX_DIMS)
    tensor_index_t dims[TENSOR_MAX_DIMS];    // sizes per dimension
    tensor_stride_t strides[TENSOR_MAX_DIMS]; // stride per dim in elements (0 => contiguous default)

    const tensor_quant_params_t *qparams;     // optional quant params (NULL if not quantised)
} Tensor;

// ---- Small helpers (header-only, no allocation) ----

static inline size_t tensor_element_size(const Tensor *t) {
    switch (t->dtype) {
        case TENSOR_DTYPE_F32: return sizeof(float);
        case TENSOR_DTYPE_I8:  return sizeof(int8_t);
        case TENSOR_DTYPE_U8:  return sizeof(uint8_t);
        case TENSOR_DTYPE_I16: return sizeof(int16_t);
        default: return 0u;
    }
}

static inline size_t tensor_numel(const Tensor *t) {
    size_t n = 1u;
    for (uint8_t i = 0; i < t->dims_count; ++i) {
        n *= (size_t)t->dims[i];
    }
    return n;
}

// Initialise strides for a contiguous row-major tensor (last dim contiguous)
static inline void tensor_init_strides_contiguous(Tensor *t) {
    if (t->dims_count == 0) return;
    tensor_stride_t s = 1;
    for (int i = (int)t->dims_count - 1; i >= 0; --i) {
        t->strides[i] = s;
        s = (tensor_stride_t)((size_t)s * (size_t)t->dims[i]);
    }
}

// Compute byte offset for an index tuple (assumes strides initialised in elements)
static inline size_t tensor_offset_bytes(const Tensor *t, const tensor_index_t *indices) {
    size_t off_elems = 0u;
    for (uint8_t i = 0; i < t->dims_count; ++i) {
        off_elems += (size_t)indices[i] * (size_t)t->strides[i];
    }
    return off_elems * tensor_element_size(t);
}

// Convenience: return typed pointer at index (bounds not checked)
static inline void* tensor_ptr_at(const Tensor *t, const tensor_index_t *indices) {
    return (void*)((uintptr_t)t->data + tensor_offset_bytes(t, indices));
}

