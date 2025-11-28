// Simple LED binary counter for NUCLEO-H723ZG using libopencm3
// Counts 0..7 each second and shows value on LD1/LD2/LD3.

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/cm3/systick.h>

#include "network.h"
#include "network_data.h"

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

int main(void) {

	clock_setup();
	gpio_setup();
	systick_setup(0);

	// Initialize the neural network model
	ai_network_params params = {
		.params = AI_HANDLE_NULL,
		.activations = AI_HANDLE_NULL
	};

	// Get the default params (includes weights)
	if (!ai_network_data_params_get(&params)) {
		// Error: blink red LED rapidly
		for (;;) {
			gpio_toggle(LED3_PORT, LED3_PIN);
			delay_ms(100);
		}
	}

	// Create the network
	ai_error err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
	if (err.type != AI_ERROR_NONE) {
		// Error: blink red LED
		for (;;) {
			gpio_toggle(LED3_PORT, LED3_PIN);
			delay_ms(200);
		}
	}

	// Initialize the network with activations buffer
	if (!ai_network_init(network, &params)) {
		// Error: blink red LED slowly
		for (;;) {
			gpio_toggle(LED3_PORT, LED3_PIN);
			delay_ms(500);
		}
	}

	uint8_t counter = 0;

    //
	for (;;) {
		
        // FIRST STEP: CLEAR ALL LEDS
		gpio_clear(LED1_PORT, LED1_PIN);
		gpio_clear(LED2_PORT, LED2_PIN);
		gpio_clear(LED3_PORT, LED3_PIN);

        // ON ALLUME LA LED ROUGE POUR DIRE QU'ON COMMENCE LE CALCUL
        gpio_set(LED3_PORT, LED3_PIN);

        // Generate a pseudo-random input vector (30x10 = 300 int8 values)
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
        ai_i32 batch = ai_network_run(network, ai_input, ai_output);
        
        if (batch != 1) {
            // Inference failed - blink red LED twice rapidly
            for (int i = 0; i < 4; i++) {
                gpio_toggle(LED3_PORT, LED3_PIN);
                delay_ms(50);
            }
        }

        // Get the output value (available in output_data[0])
        // ai_i8 result = output_data[0];
        // TODO: Use result for GPIO output or other purposes in the future

        // ON ALLUME LA LED VERTE POUR DIRE QU'ON A FINI LE CALCUL
        gpio_clear(LED3_PORT, LED3_PIN);
        gpio_set(LED1_PORT, LED1_PIN);

        // ON ATTENDS UN PEU POUR OBSERVER LA FIN DU CALCUL
		delay_ms(100);

        //
		// counter = (uint8_t)((counter + 1) & 0x07);
	}
}


