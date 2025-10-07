// Simple LED binary counter for NUCLEO-H723ZG using libopencm3
// Counts 0..7 each second and shows value on LD1/LD2/LD3.

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/cm3/systick.h>

// NOTE: Many Nucleo-144 boards map LD1/LD2/LD3 to PB0, PB7, PB14 respectively.
// If your board uses a different mapping, update the port/pin defines below.
#define LED1_PORT GPIOB
#define LED1_PIN  GPIO0
#define LED2_PORT GPIOE
#define LED2_PIN  GPIO1
#define LED3_PORT GPIOB
#define LED3_PIN  GPIO14

static volatile uint32_t system_millis = 0;

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

	uint8_t counter = 0;
	for (;;) {
		// Update LEDs to show bits 0..2 of counter
		gpio_clear(LED1_PORT, LED1_PIN);
		gpio_clear(LED2_PORT, LED2_PIN);
		gpio_clear(LED3_PORT, LED3_PIN);
		if (counter & 0x01) gpio_set(LED1_PORT, LED1_PIN);
		if (counter & 0x02) gpio_set(LED2_PORT, LED2_PIN);
		if (counter & 0x04) gpio_set(LED3_PORT, LED3_PIN);

		delay_ms(100);
		counter = (uint8_t)((counter + 1) & 0x07);
	}
}


