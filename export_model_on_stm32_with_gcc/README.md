# STM32H723ZG AI Inference Project

This project runs an AI model on the STM32H723ZG Nucleo board.

## Build System

The build system supports two modes: **Release** (default) and **Debug**.

### 1. Release Mode (Default)
- **Optimized for performance**
- **No Semihosting** (runs standalone without debugger)
- **No Debug Output** (printf is disabled)
- Uses `nosys.specs`

**Build & Flash:**
```bash
./build.sh
./flash.sh
```

### 2. Debug Mode
- **Semihosting Enabled** (requires debugger connection)
- **Debug Output** (printf works via OpenOCD)
- Uses `rdimon.specs`

**Build, Flash & Run:**
```bash
./build.sh debug
./flash.sh
./run.sh
```

## LED Status Indicators

| LED | Color | Status |
|-----|-------|--------|
| **LED1** | Green | **Success** / Idle (Blinking slowly = All good) |
| **LED2** | Yellow | **Input Preparation** / Initialization |
| **LED3** | Red | **Inference Running** (Fast blink = Error) |

## Project Structure

- `main.c`: Main application with AI inference loop.
- `network.c` / `network_data.c`: AI model implementation (generated).
- `debug_log.h`: Macros for debug printing (maps to printf or empty).
- `build.sh`: Build script wrapper around Make.
- `run.sh`: Script to launch OpenOCD and capture semihosting output.
