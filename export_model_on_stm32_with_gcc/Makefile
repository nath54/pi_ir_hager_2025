PROJECT = blink
BUILD_DIR = build
LIBOPENCM3_DIR = libopencm3

# Allow selection of main source file
# Usage: make MAIN_SRC=main_minimal_test.c
# Default: main.c
MAIN_SRC ?= main.c

# Device selection (default: H723ZG)
# Usage: make DEVICE=H723ZG or make DEVICE=U545RE
DEVICE ?= H723ZG

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy

# Common CFLAGS
CFLAGS = -Os -g -Wall -Wextra -std=c11 \
	-ffunction-sections -fdata-sections \
	-I$(LIBOPENCM3_DIR)/include -IInc -I.

# Common LDFLAGS (device-specific added below)
LDFLAGS_COMMON = -Wl,--gc-sections -nostartfiles \
	-L$(LIBOPENCM3_DIR)/lib \
	-LLib

# =============================================================================
# DEVICE-SPECIFIC CONFIGURATION
# =============================================================================

ifeq ($(DEVICE), H723ZG)
    # NUCLEO-H723ZG: Cortex-M7 with FPU, D-cache, I-cache
    CFLAGS += -DDEVICE_STM32H723ZG -DSTM32H7 \
              -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard
    LDFLAGS_DEVICE = -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
                     -lopencm3_stm32h7 \
                     -l:NetworkRuntime1100_CM7_GCC.a \
                     -Tstm32h723xg.ld
else ifeq ($(DEVICE), U545RE)
    # NUCLEO-U545RE-Q: Cortex-M33 with FPU, I-cache only
    CFLAGS += -DDEVICE_STM32U545RE -DSTM32U5 \
              -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
    LDFLAGS_DEVICE = -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard \
                     -lopencm3_stm32u5 \
                     -l:NetworkRuntime1100_CM33_GCC.a \
                     -Tstm32u545re.ld
else
    $(error Unknown DEVICE: $(DEVICE). Use H723ZG or U545RE)
endif

# Combine LDFLAGS
LDFLAGS = $(LDFLAGS_COMMON) $(LDFLAGS_DEVICE)

# =============================================================================
# BUILD OPTIONS
# =============================================================================

# Debug / Semihosting configuration
# Usage: make DEBUG=1
ifdef DEBUG
    CFLAGS += -DDEBUG_SEMIHOSTING -DDEBUG_UART -g
    LDFLAGS += --specs=rdimon.specs -Wl,--start-group -lc -lgcc -lrdimon -Wl,--end-group
else
    # Default: Release mode (no semihosting, smaller binary)
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
# Usage: make SYSCLK=550 (H723ZG: 64, 120, 240, 480, 550; U545RE: 16, 80, 160)
ifdef SYSCLK
    CFLAGS += -DTARGET_SYSCLK_MHZ=$(SYSCLK)
endif

# Library selection (future: HAL support)
# Usage: make LIB=hal (not yet implemented)
ifdef LIB
    ifeq ($(LIB), hal)
        CFLAGS += -DUSE_HAL_LIBRARY
        $(info HAL library support not yet implemented. Using libopencm3.)
    endif
endif

# =============================================================================
# SOURCE FILES
# =============================================================================

# Core source files
SRCS = $(MAIN_SRC) init_config.c utils.c network.c network_data.c network_data_params.c
OBJS = $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.c=.o)))

# =============================================================================
# BUILD RULES
# =============================================================================

.PHONY: all clean flash bin hex info

all: $(BUILD_DIR)/$(PROJECT).elf $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex
	@echo ""
	@echo "Build complete for $(DEVICE) @ $(if $(SYSCLK),$(SYSCLK),default) MHz"

$(BUILD_DIR):
	mkdir -p $@

# Build rule for all .c files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(PROJECT).elf: $(OBJS)
	@echo "Linking..."
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

# Info target - show configuration
info:
	@echo "Device:       $(DEVICE)"
	@echo "CPU Freq:     $(if $(SYSCLK),$(SYSCLK) MHz,default)"
	@echo "Debug:        $(if $(DEBUG),enabled,disabled)"
	@echo "Quantization: $(if $(INT_QUANTIZATION),INT8,F32)"
	@echo "Fast Mode:    $(if $(FAST_MODE),enabled,disabled)"
	@echo "Source:       $(MAIN_SRC)"
