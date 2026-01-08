PROJECT = blink
BUILD_DIR = build
LIBOPENCM3_DIR = libopencm3

# Allow selection of main source file
# Usage: make MAIN_SRC=main_minimal_test.c
# Default: main.c
MAIN_SRC ?= main.c

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy

CFLAGS = -Os -g -Wall -Wextra -std=c11 \
	-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
	-ffunction-sections -fdata-sections \
	-I$(LIBOPENCM3_DIR)/include -IInc -DSTM32H7


# Arch/ABI flags must be present at link time too to match libopencm3
LDFLAGS = -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
	-Wl,--gc-sections -nostartfiles \
	-L$(LIBOPENCM3_DIR)/lib -lopencm3_stm32h7 \
	-LLib -l:NetworkRuntime1100_CM7_GCC.a \
	-Tstm32h723xg.ld

# Debug / Semihosting configuration
# Usage: make DEBUG=1
ifdef DEBUG
    CFLAGS += -DDEBUG_SEMIHOSTING -g
    # Use rdimon.specs for semihosting
    LDFLAGS += --specs=rdimon.specs -Wl,--start-group -lc -lgcc -lrdimon -Wl,--end-group
else
    # Default: Release mode (no semihosting, smaller binary)
    # Use nosys.specs for bare metal without semihosting
    LDFLAGS += --specs=nosys.specs -Wl,--start-group -lc -lgcc -lnosys -Wl,--end-group

endif

# Quantization configuration
# Usage: make INT_QUANTIZATION=1
ifneq ($(INT_QUANTIZATION),)
    CFLAGS += -DQUANTIZED_INT8
endif

# Fast Mode configuration (No Sleep)
# Usage: make FAST_MODE=1
ifneq ($(FAST_MODE),)
    CFLAGS += -DNO_SLEEP
endif

# CPU Frequency configuration
# Usage: make SYSCLK=240 (options: 64, 120, 240, 480)
# Default: 64 MHz (safe HSI mode without PLL)
ifdef SYSCLK
    CFLAGS += -DTARGET_SYSCLK_MHZ=$(SYSCLK)
endif

# Source files - main.c gets replaced by MAIN_SRC
SRCS = $(MAIN_SRC) network.c network_data.c network_data_params.c
OBJS = $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.c=.o)))

.PHONY: all clean flash bin hex

all: $(BUILD_DIR)/$(PROJECT).elf $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex

$(BUILD_DIR):
	mkdir -p $@

# Build rule for all .c files (including main.c and main_minimal_test.c)
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(PROJECT).elf: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

bin: $(BUILD_DIR)/$(PROJECT).bin
hex: $(BUILD_DIR)/$(PROJECT).hex

$(BUILD_DIR)/$(PROJECT).bin: $(BUILD_DIR)/$(PROJECT).elf
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/$(PROJECT).hex: $(BUILD_DIR)/$(PROJECT).elf
	$(OBJCOPY) -O ihex $< $@

flash: $(BUILD_DIR)/$(PROJECT).bin
	st-flash write $< 0x08000000

clean:
	rm -rf $(BUILD_DIR)


