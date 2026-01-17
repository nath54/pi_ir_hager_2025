# STM32 AI Inference Project

Run ONNX models on STM32 microcontrollers using libopencm3 (open-source) or ST HAL (optional).

## Supported Devices

| Device              | Core       | Max Clock | Features                |
| ------------------- | ---------- | --------- | ----------------------- |
| **NUCLEO-H723ZG**   | Cortex-M7  | 550 MHz   | D-cache, I-cache, FPU   |
| **NUCLEO-U545RE-Q** | Cortex-M33 | 160 MHz   | I-cache, FPU, TrustZone |

## Quick Start

```bash
# 1. Load a model
python3 manage_models.py
# Select option 1 → choose a model

# 2. Build
make DEVICE=H723ZG SYSCLK=240

# 3. Flash
./flash.sh
```

## Model Manager

Interactive menu for managing AI models:

```bash
python3 manage_models.py
```

Features:
- **Auto-detect device** from model name (H723ZG, U545RE)
- **Auto-detect type** (F32 or INT8 quantized)
- **Clock speed selection** per device
- **Debug mode** with UART output

## Build Options

```bash
# Basic build (default: H723ZG @ 64MHz, Release mode)
make

# Full performance (H723ZG @ 550MHz, max speed)
make DEVICE=H723ZG SYSCLK=550 FAST_MODE=1

# Debug with semihosting
make DEVICE=H723ZG SYSCLK=64 DEBUG=1

# INT8 quantized model
make DEVICE=H723ZG INT_QUANTIZATION=1

# Show build info
make info
```

| Option                      | Description                   |
| --------------------------- | ----------------------------- |
| `DEVICE=H723ZG` or `U545RE` | Target board                  |
| `SYSCLK=64/120/240/480/550` | CPU frequency (H723ZG)        |
| `SYSCLK=16/80/160`          | CPU frequency (U545RE)        |
| `INT_QUANTIZATION=1`        | Enable INT8 mode              |
| `FAST_MODE=1`               | No delays, max inference rate |
| `DEBUG=1`                   | Enable UART debug output      |

## Signal Output (Oscilloscope)

The firmware outputs an **alternating signal** for precise timing:

- **Pin:** PE10 (H723ZG) / PA8 (U545RE)
- **Pattern:** Toggles HIGH↔LOW at each inference start
- **Benefit:** Clear square wave shows exact inference duration

```
Signal:  _____|‾‾‾‾‾|_____|‾‾‾‾‾|_____
              ^     ^     ^     ^
           Start  End  Start  End
           Inf1      Inf2      Inf3
```

## Project Structure

```
├── main.c              # Main application loop
├── device_config.h     # Device-specific definitions
├── init_config.c/h     # Clock, cache, MPU initialization
├── utils.c/h           # LED control, signal output, delays
├── network.c/h         # AI model (copied from models/)
├── manage_models.py    # Model manager (interactive)
├── Makefile            # Build system
├── models/             # Exported AI models from ST Edge AI
└── STM32_PROJECTS/     # Reference CUBE IDE projects
```

## Performance Notes

### Recommended Settings for Benchmarking

```bash
# Maximum performance on H723ZG
make DEVICE=H723ZG SYSCLK=550 FAST_MODE=1
```

This enables:
- 550 MHz clock (VOS0 + HSE)
- I-cache and D-cache with proper initialization
- MPU with privileged default mode
- No inter-inference delays

## LED Indicators

| LED  | Color  | Status                                 |
| ---- | ------ | -------------------------------------- |
| LED1 | Green  | Success / Idle                         |
| LED2 | Yellow | Input preparation                      |
| LED3 | Red    | Inference running (fast blink = error) |

## Adding New Models

1. Export from [ST Edge AI Developer Cloud](https://stedgeai-dc.st.com)
2. Select target device and output type
3. Place in `models/` with appropriate naming:
   - `model_name.onnx-NUCLEO-H723ZG-code_F32`
   - `model_name.onnx-NUCLEO-H723ZG-code_INT8`
4. Load via `manage_models.py`
