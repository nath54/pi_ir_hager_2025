// =============================================================================
// STM32 AI Inference Firmware
// =============================================================================
// Features:
// - Multi-device support (NUCLEO-H723ZG, NUCLEO-U545RE-Q)
// - Configurable CPU frequency via TARGET_SYSCLK_MHZ
// - Debug mode with UART logging (DEBUG build)
// - Release mode optimized for performance (default)
// - Alternating signal output for oscilloscope timing
// =============================================================================

#include "device_config.h"
#include "init_config.h"
#include "utils.h"

#include "network.h"
#include "network_data.h"

#include <stdio.h>
#include <string.h>

// =============================================================================
// AI NETWORK BUFFERS
// =============================================================================

STAI_ALIGNED(STAI_NETWORK_CONTEXT_ALIGNMENT)
static stai_network network[STAI_NETWORK_CONTEXT_SIZE];

STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations[STAI_NETWORK_ACTIVATIONS_SIZE];

// Input/output buffers - type depends on quantization
#ifdef QUANTIZED_INT8
static int8_t input_data[STAI_NETWORK_IN_1_SIZE];
static int8_t output_data[STAI_NETWORK_OUT_1_SIZE];
#else
static float input_data[STAI_NETWORK_IN_1_SIZE];
static float output_data[STAI_NETWORK_OUT_1_SIZE];
#endif

// =============================================================================
// INPUT DATA GENERATION
// =============================================================================

static int inference_counter = 0;

static void prepare_input_data(void) {
    // Generate pseudo-random input data based on time and counter
    for (int i = 0; i < STAI_NETWORK_IN_1_SIZE; i++) {
#ifdef QUANTIZED_INT8
        input_data[i] = (int8_t)((system_millis * 13 + inference_counter * 7 + i * 3) % 256 - 128);
#else
        input_data[i] = (float)((system_millis * 13 + inference_counter * 7 + i * 3) % 100) / 100.0f;
#endif
    }
}

// =============================================================================
// AI NETWORK INITIALIZATION
// =============================================================================

static void ai_network_init(void) {
    debug_printf("Initializing AI network...\n");

    stai_return_code err = stai_runtime_init();
    if (err != STAI_SUCCESS) {
        error_handler("Runtime Init", err);
    }

    err = stai_network_init(network);
    if (err != STAI_SUCCESS) {
        error_handler("Network Init", err);
    }

    const stai_ptr act_ptrs[] = {(stai_ptr)activations};
    err = stai_network_set_activations(network, act_ptrs, 1);
    if (err != STAI_SUCCESS) {
        error_handler("Set Activations", err);
    }

    const stai_ptr wgt_ptrs[] = {(stai_ptr)g_network_weights_array};
    err = stai_network_set_weights(network, wgt_ptrs, 1);
    if (err != STAI_SUCCESS) {
        error_handler("Set Weights", err);
    }

    const stai_ptr in_ptrs[] = {(stai_ptr)input_data};
    err = stai_network_set_inputs(network, in_ptrs, 1);
    if (err != STAI_SUCCESS) {
        error_handler("Set Inputs", err);
    }

    const stai_ptr out_ptrs[] = {(stai_ptr)output_data};
    err = stai_network_set_outputs(network, out_ptrs, 1);
    if (err != STAI_SUCCESS) {
        error_handler("Set Outputs", err);
    }

    debug_printf("Network ready: %s\n", STAI_NETWORK_MODEL_NAME);
}

// =============================================================================
// MAIN
// =============================================================================

int main(void) {
    // System initialization (clock, cache, MPU, GPIO, UART)
    system_init();

    // Initialize signal output for oscilloscope
    signal_gpio_init();

    // Startup message
    debug_printf("\n\n====================================\n");
    debug_printf("STM32 AI Inference - %s\n", DEVICE_NAME);
    debug_printf("Clock: %d MHz | Lib: %s\n", TARGET_SYSCLK_MHZ, LIB_NAME);
    debug_printf("====================================\n");

    // Initialize AI network
    ai_network_init();

    // Main inference loop
    for (;;) {
        // Clear LEDs
        led_clear_all();

        // Prepare input data
        led_set(LED_YELLOW);
        prepare_input_data();
        led_clear(LED_YELLOW);

        // Run inference with timing signal
        led_set(LED_RED);
        signal_inference_start();  // Signal goes HIGH or LOW (alternating)

        uint32_t t_start = millis();
        stai_return_code err = stai_network_run(network, STAI_MODE_SYNC);
        uint32_t t_elapsed = millis() - t_start;

        signal_inference_end();  // Mark end (signal stays for next toggle)
        led_clear(LED_RED);

        if (err != STAI_SUCCESS) {
            debug_printf("Inference FAILED (0x%X)\n", err);
            for (int i = 0; i < 6; i++) {
                led_toggle(LED_RED);
                delay_ms(50);
            }
        } else {
            // Process output
#ifdef QUANTIZED_INT8
            uint8_t out_byte = (uint8_t)output_data[0];
            debug_printf("OK %lums | Out: %d | INT8\n", t_elapsed, (int)output_data[0]);
#else
            uint8_t out_byte = (uint8_t)(output_data[0] * 100);
            debug_printf("OK %lums | Out: %.4f | F32\n", t_elapsed, (double)output_data[0]);
#endif

            // Emit output data on parallel bus (if available)
            signal_emit_data(out_byte);

            // Success indication
            led_set(LED_GREEN);
#ifndef NO_SLEEP
            delay_ms(100);
#endif
        }

        // Counter management
        inference_counter++;
        if (system_millis > 10000 || inference_counter > 10000) {
            system_millis = 0;
            inference_counter = 0;
            debug_printf("--- COUNTER RESET ---\n");
        }

        led_clear(LED_GREEN);

#ifndef NO_SLEEP
        delay_ms(400);
#endif
    }
}
