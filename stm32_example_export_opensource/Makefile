PROJECT = blink
BUILD_DIR = build
LIBOPENCM3_DIR = libopencm3

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy

CFLAGS = -Os -g -Wall -Wextra -std=c11 \
	-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
	-ffunction-sections -fdata-sections \
	-I$(LIBOPENCM3_DIR)/include -DSTM32H7


# Arch/ABI flags must be present at link time too to match libopencm3 build
LDFLAGS = -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
	-Wl,--gc-sections -nostartfiles \
	-L$(LIBOPENCM3_DIR)/lib -lopencm3_stm32h7 \
	-Tstm32h723xg.ld \
	-Wl,--start-group -lc -lgcc -lnosys -Wl,--end-group

SRCS = blink.c
OBJS = $(addprefix $(BUILD_DIR)/, $(SRCS:.c=.o))

.PHONY: all clean flash bin hex

all: $(BUILD_DIR)/$(PROJECT).elf $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
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


