# PI IR Hager 2025: Implementation and Optimization of Neural Networks for Microcontrollers

[![Project Status](https://img.shields.io/badge/Status-Work_in_progress-orange.svg)](https://gitlab.unistra.fr/cerisara/pi_hager_2025)
[![Target](https://img.shields.io/badge/Target-STM32H7%20/%20U5-blue.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32h7-series.html)

This repository contains the work carried out as part of the **Engineering Project (PI)** for the **IR (Informatique et Réseaux)** specialty at Telecom Physique Strasbourg, in collaboration with **Hager Group**.

The project focuses on the transformation of Deep Learning models (developed in PyTorch) into optimized C/C++ code for low-power microcontrollers, specifically the STM32 family.

## Members
- **Nathan Cerisara** ([nathan.cerisara@etu.unistra.fr](mailto:nathan.cerisara@etu.unistra.fr))
- **Clément Desberg** ([clement.desberg@etu.unistra.fr](mailto:clement.desberg@etu.unistra.fr))
- **Lucas Levy** ([lucas.levy2@etu.unistra.fr](mailto:lucas.levy2@etu.unistra.fr))
- **Ahmed Amine Jadi** ([ahmed-amine.jadi@etu.unistra.fr](mailto:ahmed-amine.jadi@etu.unistra.fr))

**Contacts (Hager Group):** Hien Duc VU, Nicolas BRITSCH

---

## Project Evolution

The project followed a multi-stage evolution, documented in detailed reports located in the [`doc/`](./doc/) directory.

### Phase 1: Conceptualization (Report [R1](./doc/R1_Projet_Ingenieur.md))
- **Goal**: Research tools (TensorFlow Lite, PyTorch Mobile) and define an intermediate language.
- **Key Decision**: Creation of a JSON-based Intermediate Representation (IR) to bridge high-level Python models and low-level code.

### Phase 2: Optimization Research (Report [R2](./doc/R2_Projet_Ingenieur.md))
- **Focus**: Weight quantization (INT8), pruning, and knowledge distillation.
- **Architecture**: Choice of **C++** with the **Fastor** library for high-performance tensor operations on microcontrollers.

### Phase 3: State of the Art & Pivot (Report [R3](./doc/R3_Projet_Ingenieur.md))
- **Exploration**: Evaluated `Executorch`, `LiteRT`, and `TorchScript`. Most were rejected due to dependency hell or lack of maintenance.
- **Pivot**: Introduction of the **ST Edge AI Developer Cloud** as a primary industry-standard path, while maintaining the manual "ModelBlocks" path for research.

### Phase 4: Dual-Path Implementation (Report [R4](./doc/R4_Projet_Ingenieur.md))
- **Sub-group 1 (ModelBlocks)**: Implemented an AST-based PyTorch parser and a NumPy-backed interpreter to validate the Intermediate Representation.
- **Sub-group 2 (ST Cloud)**: Benchmarking of various architectures (CNN, RNN, Transformers) using the ST proprietary pipeline.

### Phase 5: Deployment & Benchmarking (Report [R5](./doc/R5_Projet_Ingenieur.md))
- **Current Process**: PyTorch -> ONNX -> ST Edge AI Cloud -> Optimized C Code -> STM32 Flash.
- **Findings**: Detailed analysis of memory vs. latency trade-offs for different model families (Linear, CNN, RNN, Transformers, Hybrids).

### Phase 6: Physical benchmarking and pipeline finalization
- **Physical benchmarking**: The obtained results on a fiew models on the STM32U545RE-Q board were not as expected. We did our best but could not reach the expected performance.
- **Pipeline finalization**: Our final pipeline is the one detailed a little later in this document.

---

## Technical Architecture

During this project, we explored multiple paths to convert PyTorch models into optimized C code for microcontrollers.

### Developed from scratch: NeuralBlocks (Open-Source)

The objective of this path was to develop a custom solution to convert PyTorch models into optimized C code for microcontrollers. The aim was to have a non-proprietary solution that could be used to convert any PyTorch model into optimized C code for microcontrollers (like ST Edge AI Developer Cloud).

We successfully implemented a parser for PyTorch models and an interpreter for the Intermediate Representation (IR) to validate the IR. But we didn't had the time to implement the conversion from IR to C code before it was asked to us to pivot to the ST Edge AI Developer Cloud.

The **NeuralBlocks** sub-project has been moved to its own repository: [Pytorch_ModelBlock_IntermediateLayer_Conversion](https://github.com/nath54/Pytorch_ModelBlock_IntermediateLayer_Conversion).

### Custom PyTorch to ONNX + Pre-Benchmark

The local directory [`onnx_models_conversion_and_benchmark/`](./onnx_models_conversion_and_benchmark/) now serves as the **first stage of the Main Pipeline**:

- **Library of Models**: instead of a single model, we host a library of models to be tested.
- **ONNX Conversion**: converts PyTorch models from the library into ONNX format.
- **Pre-Benchmark (Optional)**: allows running a preliminary benchmark on the computer to test models before deploying to the STM32.

### STM32-H723ZG (libopencm3 + arm-none-eabi-gcc)

Located in [`export_model_on_stm32_with_gcc/`](./export_model_on_stm32_with_gcc/), we first worked on the STM32-H723ZG board using the libopencm3 library and the arm-none-eabi-gcc compiler.

- **Pipeline**: Converts ONNX models to optimized C headers using ST's proprietary kernels.
- **Automation**: Includes `manage_models.py`, a script to easily select models, switch between INT8/F32 quantization, build, and flash.
- **Embedded Env**: Integration with `libopencm3` and `arm-none-eabi-gcc` for a purely open-source build system (no dependence on heavy IDEs for production).

### STM32U5 HAL (CMake + STM32CubeMX + arm-none-eabi-gcc)
Located in [`export_model_on_stm32_with_hal_gcc/`](./export_model_on_stm32_with_hal_gcc/), we then pivoted to the STM32U5 HAL, targeting the **NUCLEO-U545RE-Q** board using the official STM32 HAL.

- **Build System**: Uses **CMake** for a modern, standard build workflow.
- **Automation**: Includes `manage_models.py`, a script to easily select models, switch between INT8/F32 quantization, build, and flash.
- **HAL**: leveraged for peripheral access (GPIO, UART, etc.), offering a more standardized approach than libopencm3 for newer chips.

---

## Repository Structure

```text
.
├── doc/                                        # Project reports (R1-R5) and format guides
├── export_model_on_stm32_with_gcc/             # STM32 C implementation and build system
│   └── manage_models.py                        # [SCRIPT] Automation script for model loading, build, flashing and debugging
├── export_model_on_stm32_with_hal_gcc/         # STM32U5 HAL + CMake implementation (NUCLEO-U545RE-Q)
│   └── manage_models.py                        # [SCRIPT] Automation script for model loading, build, flashing and debugging
├── onnx_models_conversion_and_benchmark/       # PyTorch -> ONNX conversion and pre-benchmarking
│   └── main_convert_to_onnx_and_measures.py    # [SCRIPT] Script to convert all the models pytorch into onnx and run a local benchmark
└── requirements.txt                            # List of the python dependencies
```

---

## Main Pipeline (PyTorch -> STM32) Instructions & Usage

### Prerequisites
1. **An Ubuntu Computer** (it works on windows but it is more tricky)
2. **Python 3.10+** (3.10 recommended for some ST tools compatibility).
3. **CMake**: Tool for the compilation
4. **ARM GNU Toolchain**: `arm-none-eabi-gcc` for compiling for STM32.
5. **st-link**: To flash the binary onto the boards.

### Commands

```bash
# Clone the repository
git clone https://github.com/nath54/pi_ir_hager_2025
cd pi_ir_hager_2025

# Install the dependencies
pip install -r requirements.txt

# Pipeline Step 1: Convert PyTorch models to ONNX
python onnx_models_conversion_and_benchmark/main_convert_to_onnx_and_measures.py --skip_measures

# Pipeline Step 2:
#  - Go to [ST Edge AI Cloud](https://stm32ai-cs.st.com/)
#  - Login / Create an account
#  - Upload the ONNX models
#  - Select the "STM32 MCUs" platform
#  - Activate or not the model quantization
#  - Optimize for inference time
#  - Go directly into the Generate page
#  - Choose the "NUCLEO-U545RE-Q" board
#  - Download the C code
#  - This will generate a zip file for each ONNX model you process
#  - Then extract theses zip files into the `stm32_ai_models/` directory
#    **IMPORTANT:** The name of the subdirectories must be the exact name of the zip file

# Pipeline Step 3: Build and flash the STM32 project
cd export_model_on_stm32_with_hal_gcc
python manage_models.py
#  - Then select the option [1] to select the model you want to build and flash
#  - Then select the option [7] to clean the build directory
#  - Then select the option [3] to build the project
#  - Then select the option [6] to flash the project
```

Theses steps have been tested on Ubuntu and Arch-Linux computers.

For windows, the workaround is to use WSL (Windows Subsystem for Linux). But there is a catch: the STM32 USB plugged is not detected via the WSL. So the next workaround is to use the STM32_Programmer_CLI tool on windows directly **after the build phase on the wsl `manage_models.py` script** on the `export_model_on_stm32_with_hal_gcc/build/Release/stm32_hal_ai.bin` result file.

---

## Benchmarking Instructions

If you want to run some benchmarking, you can do:

- **To run Pre-Benchmark on a computer (ONNXRUNTIME on CPU):**

```bash
cd onnx_models_conversion_and_benchmark
python main_convert_to_onnx_and_measures.py
```

*Note: It is the same command as the Pipeline Step 1, but without the `--skip_measures` flag.*

- **To run the benchmarks on ST Edge AI Developer Cloud:** You have to do the same steps as the Pipeline Step 2, but instead of skipping and going directly into the "Generate" page, you have to go into the "Benchmark" page. Then there are not the exact STM32 MCU that we had (STM32 H723ZG and STM32 U545RE-Q). So we used the closest equivalents that we found: *NUCLEO-G474RE* and *B-U585I-IOT02A*.

- **To benchmark the models physically on the STM32 boards:** You have to execute the full final Pipeline, then after the flash on the STM32 board, we used an oscilloscope on a PIN that was connected to the LD2 LED: `C5 / PIN 6`, because we couldn't get the `IO D7` (C9 / D7) PIN working. The signal was set to toggle between each inference, so we could measure the inference time precisely by measuring multiple oscillations and then divide the total time by the number of oscillations.

---

## Useful Links

- [ST Edge AI Cloud](https://stm32ai-cs.st.com/)

---
© 2025-2029 PI IR Hager Project Team.
