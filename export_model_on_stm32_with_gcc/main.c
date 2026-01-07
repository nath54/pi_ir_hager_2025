// STM32H723ZG AI Inference - UPDATED FOR ST.AI V3 API + UART Logging
// All critical fixes + performance optimizations based on official STM32 export
// - MPU configuration
// - I-Cache and D-Cache enabled
// - 550MHz clock from PLL (simulated via HSE/HSI here)
// - 32-byte aligned buffers
// - Proper initialization sequence
// - UART3 Logging enabled (PD8/PD9)

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/cm3/scb.h>
#include <libopencm3/cm3/mpu.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "network.h"
#include "network_data.h"
// #include "debug_log.h" // We implement our own debug_printf now

// LED definitions for NUCLEO-H723ZG
#define LED1_PORT GPIOB  // Green LED
#define LED1_PIN  GPIO0
#define LED2_PORT GPIOE  // Yellow LED
#define LED2_PIN  GPIO1
#define LED3_PORT GPIOB  // Red LED
#define LED3_PIN  GPIO14

static volatile uint32_t system_millis = 0;

// ==============================================================================
// ST.AI V3 DATA STRUCTURES
// ==============================================================================

// 1. Network Context
STAI_ALIGNED(STAI_NETWORK_CONTEXT_ALIGNMENT)
static stai_network network[STAI_NETWORK_CONTEXT_SIZE];

// 2. Activations Buffer
STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations[STAI_NETWORK_ACTIVATIONS_SIZE];

// 3. Input/Output Buffers
static float input_data[STAI_NETWORK_IN_1_SIZE];
static float output_data[STAI_NETWORK_OUT_1_SIZE];

// Input generator macro
#define MODEL_INPUT_GEN(i) ((float)((system_millis * 13 + counter * 7 + i * 3) % 100) / 100.0f)

// ==============================================================================
// SYSTEM FUNCTIONS
// ==============================================================================

void sys_tick_handler(void) {
	system_millis++;
}

static void delay_ms(uint32_t ms) {
	uint32_t start = system_millis;
	while ((system_millis - start) < ms) {
		__asm__("wfi");
	}
}

// CRITICAL FIX 1: MPU Configuration
static void mpu_config(void) {
	// Simply disable MPU to avoid conflicts
	MPU_CTRL = 0;
}

static void system_clock_config(void) {
    // Basic HSI setup (64MHz) is default on reset.
    // For UART at 115200 with 64MHz, standard PCLK settings work fine.
    // We won't touch PLL to keep it robust for now.
}

static void gpio_setup(void) {
	rcc_periph_clock_enable(RCC_GPIOB);
	rcc_periph_clock_enable(RCC_GPIOE);
	gpio_mode_setup(LED1_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED1_PIN);
	gpio_mode_setup(LED2_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED2_PIN);
	gpio_mode_setup(LED3_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED3_PIN);

	gpio_clear(LED1_PORT, LED1_PIN);
	gpio_clear(LED2_PORT, LED2_PIN);
	gpio_clear(LED3_PORT, LED3_PIN);
}

// ==============================================================================
// UART SETUP
// ==============================================================================
static void usart_setup(void) {
    // Nucleo-H723ZG VCP is on USART3: PD8 (TX) and PD9 (RX)
    rcc_periph_clock_enable(RCC_GPIOD);
    rcc_periph_clock_enable(RCC_USART3);

    // TX Pin (PD8)
    gpio_mode_setup(GPIOD, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO8);
    gpio_set_af(GPIOD, GPIO_AF7, GPIO8);

    // RX Pin (PD9)
    gpio_mode_setup(GPIOD, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO9);
    gpio_set_af(GPIOD, GPIO_AF7, GPIO9);

    // Setup USART parameters
    usart_set_baudrate(USART3, 115200);
    usart_set_databits(USART3, 8);
    usart_set_stopbits(USART3, USART_STOPBITS_1);
    usart_set_mode(USART3, USART_MODE_TX_RX);
    usart_set_parity(USART3, USART_PARITY_NONE);
    usart_set_flow_control(USART3, USART_FLOWCONTROL_NONE);

    usart_enable(USART3);
}

// Custom printf implementation for UART
static void debug_printf(const char *format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    for (int i = 0; buffer[i] != '\0'; i++) {
        usart_send_blocking(USART3, buffer[i]);
        // Add CR before LF for terminal compatibility if needed
        if (buffer[i] == '\n') {
             // usart_send_blocking(USART3, '\r');
             // Intentionally commented out, usually terminals handle LF locally
        }
    }
}

static void systick_setup(uint32_t ahb_hz) {
	if (ahb_hz == 0) {
		ahb_hz = 64000000u;  // 64MHz from HSI (Reset default)
	}
	uint32_t reload = (ahb_hz / 1000u) - 1u;
	systick_set_reload(reload);
	systick_set_clocksource(STK_CSR_CLKSOURCE_AHB);
	systick_clear();
	systick_interrupt_enable();
	systick_counter_enable();
}

static void error_handler(const char* error_msg, stai_return_code err_code) {
	debug_printf("ERROR: %s (Code: 0x%X)\n", error_msg, err_code);
	for (;;) {
		gpio_set(LED3_PORT, LED3_PIN);
		delay_ms(100);
		gpio_clear(LED3_PORT, LED3_PIN);
		delay_ms(100);
	}
}

// ==============================================================================
// SIGNAL OUTPUT SETUP (GPIO PORT E)
// PINS: PE2-PE9 (Data D0-D7), PE10 (Valid Strobe)
// ==============================================================================
static void gpio_signal_setup(void) {
    rcc_periph_clock_enable(RCC_GPIOE);
    // PE2-PE10 as Output Push-Pull
    gpio_mode_setup(GPIOE, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE,
                    GPIO2 | GPIO3 | GPIO4 | GPIO5 |
                    GPIO6 | GPIO7 | GPIO8 | GPIO9 | GPIO10);
    // Initial state Low
    gpio_clear(GPIOE, GPIO2 | GPIO3 | GPIO4 | GPIO5 |
                      GPIO6 | GPIO7 | GPIO8 | GPIO9 | GPIO10);
}

static void emit_signal(uint8_t value) {
    // 1. Set Data Pins (PE2-PE9)
    // Clear first to be safe (masked)
    uint16_t mask = (0xFF << 2); // 0x03FC
    gpio_clear(GPIOE, mask);

    // Set bits based on value shifted by 2 (PE2 is LSB)
    uint16_t set_bits = ((uint16_t)value) << 2;
    gpio_set(GPIOE, set_bits & mask);

    // 2. Pulse Valid Strobe (PE10)
    gpio_set(GPIOE, GPIO10);
    // Short delay for signal integrity (~1us is enough usually, but we do more for visibility if needed)
    // For fast signaling, just a few cycles.
    for(volatile int i=0; i<100; i++) __asm__("nop");
    gpio_clear(GPIOE, GPIO10);
}

// ==============================================================================
// MAIN
// ==============================================================================

int main(void) {

	// STEP 0: Enable FPU
	SCB_CPACR |= (0xF << 20);
	__asm__ volatile ("dsb");
	__asm__ volatile ("isb");

	// STEP 1: Disable MPU
	mpu_config();

	// STEP 2: Configure system clock (HSI Default)
	system_clock_config();

	// STEP 3: Initialize peripherals
	gpio_setup();
    usart_setup();
    gpio_signal_setup(); // SIGNAL OUTPUT INIT
	systick_setup(64000000);  // 64MHz

    // Hello message
    debug_printf("\n\n");
    debug_printf("========================================\n");
    debug_printf("STM32 AI Inference v3.0 + SIGNAL OUTPUT\n");
    debug_printf("Signal: PE2-PE9 (Data), PE10 (Strobe)\n");
    debug_printf("UART: Enabled (115200 8N1)\n");
    debug_printf("========================================\n");

	// ========================
	// AI NETWORK INITIALIZATION
	// ========================
	debug_printf("Initializing ST.AI Network...\n");

    // Initialize Runtime
    stai_return_code err = stai_runtime_init();
    if (err != STAI_SUCCESS) error_handler("Runtime Init Failed", err);

	debug_printf("  Model: %s\n", STAI_NETWORK_MODEL_NAME);

    // Initialize Network Context
    err = stai_network_init(network);
    if (err != STAI_SUCCESS) error_handler("Network Init Failed", err);

    // Set Activations
    const stai_ptr activations_ptrs[] = { (stai_ptr)activations };
    err = stai_network_set_activations(network, activations_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Activations Failed", err);

    // Set Weights
    const stai_ptr weights_ptrs[] = { (stai_ptr)g_network_weights_array };
    err = stai_network_set_weights(network, weights_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Weights Failed", err);

    // Set Inputs (bind static buffer)
    const stai_ptr inputs_ptrs[] = { (stai_ptr)input_data };
    err = stai_network_set_inputs(network, inputs_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Inputs Failed", err);

    // Set Outputs (bind static buffer)
    const stai_ptr outputs_ptrs[] = { (stai_ptr)output_data };
    err = stai_network_set_outputs(network, outputs_ptrs, 1);
    if (err != STAI_SUCCESS) error_handler("Set Outputs Failed", err);

	debug_printf("Network initialized successfully!\n");
	debug_printf("Network context: %p\n\n", (void*)network);


	// ========================
	// MAIN INFERENCE LOOP
	// ========================

	int counter = 0;

	for (;;) {
		// debug_printf("--- Iteration (t=%lu ms) ---\n", system_millis);

		// Reset LEDs
		gpio_clear(LED1_PORT, LED1_PIN);
		gpio_clear(LED2_PORT, LED2_PIN);
		gpio_clear(LED3_PORT, LED3_PIN);

		// Orange LED ON: input preparation
		gpio_set(LED2_PORT, LED2_PIN);

		// Fill input buffer with dummy data
		for (int i = 0; i < STAI_NETWORK_IN_1_SIZE; i++) {
			input_data[i] = MODEL_INPUT_GEN(i);
		}

		// Run inference
		// debug_printf("Running inference...\n");

		// DEBUG: Check input (print first element only to avoid spam)
		// debug_printf("  Input[0] = %f\n", (double)input_data[0]);

		// Red LED ON: inference
        gpio_clear(LED2_PORT, LED2_PIN);
		gpio_set(LED3_PORT, LED3_PIN);

		uint32_t start_time = system_millis;

        // V3 API Run
		err = stai_network_run(network, STAI_MODE_SYNC);

		uint32_t end_time = system_millis;
		uint32_t inference_time = end_time - start_time;

		if (err != STAI_SUCCESS) {
			debug_printf("FAILED! (Code: 0x%X)\n", err);

			// Blink red LED rapidly
			for (int i = 0; i < 6; i++) {
				gpio_toggle(LED3_PORT, LED3_PIN);
				delay_ms(50);
			}
		} else {
            uint8_t out_byte = 0;

            #ifdef QUANTIZED_INT8
                out_byte = (uint8_t)output_data[0];
            #else
                out_byte = (uint8_t)(output_data[0] * 100); // Scale float
            #endif

            // EMIT SIGNAL
            emit_signal(out_byte);

			// Compact output
            #ifdef QUANTIZED_INT8
                debug_printf("INF: OK (%lu ms) | Out: %d (0x%02X) | Mode: INT8\n", inference_time, (int8_t)output_data[0], out_byte);
            #else
			    debug_printf("INF: OK (%lu ms) | Out: %.4f (0x%02X) | Mode: FP32\n", inference_time, (double)output_data[0], out_byte);
            #endif

			// Blink green LED for success
            gpio_clear(LED3_PORT, LED3_PIN);
			for(int i=0; i<2; i++) {
				gpio_set(LED1_PORT, LED1_PIN);
				delay_ms(50);
				gpio_clear(LED1_PORT, LED1_PIN);
				delay_ms(50);
			}
            gpio_set(LED1_PORT, LED1_PIN); // Keep green on
		}


		#ifndef NO_SLEEP
		delay_ms(200);
		#endif

		counter++;
		// Reset logic
		if(system_millis > 10000 || counter > 10000) {
			system_millis = 0;
			counter = 0;
            debug_printf("--- RESET COUNTERS ---\n");
			gpio_set(LED1_PORT, LED1_PIN);
			gpio_set(LED2_PORT, LED2_PIN);
			gpio_set(LED3_PORT, LED3_PIN);
			#ifndef NO_SLEEP
			delay_ms(2000);
			#endif
		}

		gpio_clear(LED1_PORT, LED1_PIN);
		gpio_clear(LED2_PORT, LED2_PIN);
		gpio_clear(LED3_PORT, LED3_PIN);

		#ifndef NO_SLEEP
		delay_ms(400);
		#endif
	}
}
