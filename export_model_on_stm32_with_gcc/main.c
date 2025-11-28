// Simple LED binary counter for NUCLEO-H723ZG using libopencm3
// Counts 0..7 each second and shows value on LD1/LD2/LD3.

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/cm3/systick.h>
#include <stdio.h>

#include "network.h"
#include "network_data.h"
#include "debug_log.h"

// NOTE: Many Nucleo-144 boards map LD1/LD2/LD3 to PB0, PB7, PB14 respectively.
// If your board uses a different mapping, update the port/pin defines below.
#define LED1_PORT GPIOB
#define LED1_PIN  GPIO0
#define LED2_PORT GPIOE
#define LED2_PIN  GPIO1
#define LED3_PORT GPIOB
#define LED3_PIN  GPIO14

static volatile uint32_t system_millis = 0;

// Neural network variables
static ai_handle network = AI_HANDLE_NULL;
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATION_1_SIZE];
static ai_i8 input_data[AI_NETWORK_IN_1_SIZE];   // 300 bytes: 30x10 matrix
static ai_i8 output_data[AI_NETWORK_OUT_1_SIZE]; // 1 byte output

void sys_tick_handler(void) {
	system_millis++;
}

static void delay_ms(uint32_t ms) {
	uint32_t start = system_millis;
	while ((system_millis - start) < ms) {
		__asm__("wfi");
	}
}

static void clock_setup(void) {
	// Use internal clock defaults; user code can adjust later if needed.
	// We only need GPIO clock and SysTick here.
}

static void gpio_setup(void) {
	rcc_periph_clock_enable(RCC_GPIOB);
	rcc_periph_clock_enable(RCC_GPIOE);
	gpio_mode_setup(LED1_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED1_PIN);
	gpio_mode_setup(LED2_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED2_PIN);
	gpio_mode_setup(LED3_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED3_PIN);

	// Start with LEDs off
	gpio_clear(LED1_PORT, LED1_PIN);
	gpio_clear(LED2_PORT, LED2_PIN);
	gpio_clear(LED3_PORT, LED3_PIN);
}

static void systick_setup(uint32_t ahb_hz) {
	// Configure SysTick to 1ms tick from AHB clock
	// If ahb_hz is 0, fall back to 200 MHz assumption (typical H7 default after reset is HSI)
	if (ahb_hz == 0) {
		ahb_hz = 200000000u; // conservative default; timing is approximate
	}
	uint32_t reload = (ahb_hz / 1000u) - 1u;
	systick_set_reload(reload);
	systick_set_clocksource(STK_CSR_CLKSOURCE_AHB);
	systick_clear();
	systick_interrupt_enable();
	systick_counter_enable();
}

static void error_blink(uint32_t ms, const char* error_msg) {
	debug_printf("ERROR: %s\n", error_msg);
	for (;;) {
		gpio_set(LED3_PORT, LED3_PIN);
		delay_ms(ms);
		gpio_clear(LED3_PORT, LED3_PIN);
		delay_ms(ms);
	}
}

int main(void) {

	clock_setup();
	gpio_setup();
	systick_setup(0);

	debug_printf("\n\n=== STM32 AI Network Debug Log ===\n");
	debug_printf("System initialized\n");

	// Initialize the neural network using the helper function
	// Prepare activation and weight arrays
	debug_printf("Preparing network buffers...\n");
	ai_handle activations_table[] = { activations };
	ai_handle weights_table[] = { ai_network_data_weights_get() };

	// Create and initialize the network in one call
	debug_printf("Creating and initializing AI network...\n");
	ai_error err = ai_network_create_and_init(&network, activations_table, weights_table);
	if (err.type != AI_ERROR_NONE) {
		// Error: blink red LED to indicate initialization failure
		debug_printf("Network init failed! Error type: %d, code: %d\n", err.type, err.code);
		error_blink(100, "AI network initialization failed");
	}
	debug_printf("Network initialized successfully!\n");

	uint8_t counter = 0;

	debug_printf("Entering main loop...\n");
    //
	for (;;) {
		debug_printf("\n--- Iteration %d (time=%lu ms) ---\n", counter, system_millis);
		
        // FIRST STEP: CLEAR ALL LEDS
		gpio_clear(LED1_PORT, LED1_PIN);
		gpio_clear(LED2_PORT, LED2_PIN);
		gpio_clear(LED3_PORT, LED3_PIN);

        // ON ALLUME LA LED ROUGE POUR DIRE QU'ON COMMENCE LE CALCUL
        gpio_set(LED3_PORT, LED3_PIN);
        debug_printf("Red LED ON - starting computation\n");

        // Generate a pseudo-random input vector (30x10 = 300 int8 values)
        debug_printf("Generating input data...\n");
        for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
            // Simple PRNG using system time and counter
            input_data[i] = (ai_i8)((system_millis * 13 + counter * 7 + i * 3) % 256 - 128);
        }

        // Prepare input buffer
        ai_buffer ai_input[AI_NETWORK_IN_NUM] = {
            AI_BUFFER_INIT(
                AI_FLAG_NONE,
                AI_BUFFER_FORMAT_S8,
                AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, AI_NETWORK_IN_1_CHANNEL, 1, AI_NETWORK_IN_1_HEIGHT),
                AI_NETWORK_IN_1_SIZE,
                NULL,
                input_data
            )
        };

        // Prepare output buffer
        ai_buffer ai_output[AI_NETWORK_OUT_NUM] = {
            AI_BUFFER_INIT(
                AI_FLAG_NONE,
                AI_BUFFER_FORMAT_S8,
                AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, AI_NETWORK_OUT_1_CHANNEL, 1, 1),
                AI_NETWORK_OUT_1_SIZE,
                NULL,
                output_data
            )
        };

        // Run the neural network inference
        debug_printf("Running inference...\n");
        ai_i32 batch = ai_network_run(network, ai_input, ai_output);
        debug_printf("Inference complete. Batch result: %ld\n", (long)batch);
        
        if (batch != 1) {
            // Inference failed - blink red LED twice rapidly
            debug_printf("WARNING: Inference failed!\n");
            for (int i = 0; i < 4; i++) {
                gpio_toggle(LED3_PORT, LED3_PIN);
                delay_ms(50);
            }
        }

        // Get the output value (available in output_data[0])
        ai_i8 result = output_data[0];
        debug_printf("Output value: %d\n", result);

        // ON ALLUME LA LED VERTE POUR DIRE QU'ON A FINI LE CALCUL
        gpio_clear(LED3_PORT, LED3_PIN);
        gpio_set(LED1_PORT, LED1_PIN);
        debug_printf("Green LED ON - computation finished\n");

        // ON ATTENDS UN PEU POUR OBSERVER LA FIN DU CALCUL
		delay_ms(100);

        counter++;
        //
		// counter = (uint8_t)((counter + 1) & 0x07);
	}
}


