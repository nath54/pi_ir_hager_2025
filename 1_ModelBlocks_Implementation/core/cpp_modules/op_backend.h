#pragma once
// Portable C operator backend interface with pluggable implementations.
// Backends: CMSIS-NN (int8/fp32), Fastor (fp32), reference C (int8/fp32).

#include "tensor_header.h"

#include <stdint.h>
#include <stddef.h>

// Convolution/pooling parameters
typedef struct {
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
} op_conv2d_params_t;

typedef struct {
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;
    int32_t stride_h;
    int32_t stride_w;
    int32_t kernel_h;
    int32_t kernel_w;
} op_pool2d_params_t;

// Status codes
typedef enum {
    OP_OK = 0,
    OP_UNSUPPORTED = 1,
    OP_INVALID_ARG = 2,
    OP_NO_MEMORY = 3,
    OP_RUNTIME_ERR = 4
} op_status_t;

struct OpBackend;

// Function table for all supported ops. First argument is backend for scratch/config.
typedef struct {
    // Memory/scratch configuration (optional)
    op_status_t (*set_scratch)(struct OpBackend *backend, void *scratch, size_t scratch_size);

    // Fully connected / linear: X[N, in], W[out, in] (row-major), B[out] optional, Y[N, out]
    op_status_t (*linear)(struct OpBackend *backend, const Tensor *X, const Tensor *W, const Tensor *B, Tensor *Y);

    // Convolution 2D, NHWC by default (follow Tensor.layout). W layout: [KH, KW, Cin, Cout] for NHWC
    op_status_t (*conv2d)(struct OpBackend *backend, const Tensor *X, const Tensor *W, const Tensor *B,
                          const op_conv2d_params_t *p, Tensor *Y);

    // Depthwise conv2d, NHWC, weights [KH, KW, Cin, depth_multiplier]
    op_status_t (*depthwise_conv2d)(struct OpBackend *backend, const Tensor *X, const Tensor *W, const Tensor *B,
                                    const op_conv2d_params_t *p, Tensor *Y);

    // Pooling 2D: max/avg controlled by mode (0=max,1=avg)
    op_status_t (*max_pool2d)(struct OpBackend *backend, const Tensor *X, const op_pool2d_params_t *p, Tensor *Y);
    op_status_t (*avg_pool2d)(struct OpBackend *backend, const Tensor *X, const op_pool2d_params_t *p, Tensor *Y);

    // Elementwise ops (broadcast rules: trailing dims match or size 1)
    op_status_t (*add)(struct OpBackend *backend, const Tensor *A, const Tensor *B, Tensor *C);
    op_status_t (*mul)(struct OpBackend *backend, const Tensor *A, const Tensor *B, Tensor *C);
    op_status_t (*relu)(struct OpBackend *backend, const Tensor *X, Tensor *Y);

    // Softmax along last dimension
    op_status_t (*softmax)(struct OpBackend *backend, const Tensor *X, Tensor *Y);

    // Requantize: apply scale/zero-point change for int8 tensors (per-tensor or per-channel)
    op_status_t (*requantize)(struct OpBackend *backend, const Tensor *X, const tensor_quant_params_t *to_qparams, Tensor *Y);

    // Permute (fallback for non-native layouts)
    op_status_t (*transpose)(struct OpBackend *backend, const Tensor *X, const uint8_t *perm, uint8_t perm_len, Tensor *Y);
} tensor_ops_vtable_t;

typedef struct OpBackend {
    const tensor_ops_vtable_t *vtable;
    void *scratch;
    size_t scratch_size;
    // Backend-specific configuration hook (may be NULL)
    void *user_ctx;
} OpBackend;

// Inline wrappers to simplify call sites
static inline op_status_t op_set_scratch(OpBackend *b, void *buf, size_t n) {
    return b->vtable->set_scratch ? b->vtable->set_scratch(b, buf, n) : OP_OK;
}

static inline op_status_t op_linear(OpBackend *b, const Tensor *X, const Tensor *W, const Tensor *B, Tensor *Y) {
    return b->vtable->linear(b, X, W, B, Y);
}

static inline op_status_t op_conv2d(OpBackend *b, const Tensor *X, const Tensor *W, const Tensor *B, const op_conv2d_params_t *p, Tensor *Y) {
    return b->vtable->conv2d(b, X, W, B, p, Y);
}

static inline op_status_t op_depthwise_conv2d(OpBackend *b, const Tensor *X, const Tensor *W, const Tensor *B, const op_conv2d_params_t *p, Tensor *Y) {
    return b->vtable->depthwise_conv2d(b, X, W, B, p, Y);
}

static inline op_status_t op_max_pool2d(OpBackend *b, const Tensor *X, const op_pool2d_params_t *p, Tensor *Y) {
    return b->vtable->max_pool2d(b, X, p, Y);
}

static inline op_status_t op_avg_pool2d(OpBackend *b, const Tensor *X, const op_pool2d_params_t *p, Tensor *Y) {
    return b->vtable->avg_pool2d(b, X, p, Y);
}

static inline op_status_t op_add(OpBackend *b, const Tensor *A, const Tensor *B, Tensor *C) {
    return b->vtable->add(b, A, B, C);
}

static inline op_status_t op_mul(OpBackend *b, const Tensor *A, const Tensor *B, Tensor *C) {
    return b->vtable->mul(b, A, B, C);
}

static inline op_status_t op_relu(OpBackend *b, const Tensor *X, Tensor *Y) {
    return b->vtable->relu(b, X, Y);
}

static inline op_status_t op_softmax(OpBackend *b, const Tensor *X, Tensor *Y) {
    return b->vtable->softmax(b, X, Y);
}

static inline op_status_t op_requantize(OpBackend *b, const Tensor *X, const tensor_quant_params_t *to_qparams, Tensor *Y) {
    return b->vtable->requantize(b, X, to_qparams, Y);
}

static inline op_status_t op_transpose(OpBackend *b, const Tensor *X, const uint8_t *perm, uint8_t perm_len, Tensor *Y) {
    return b->vtable->transpose(b, X, perm, perm_len, Y);
}


