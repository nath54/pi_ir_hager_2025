# PI IR Hager 2025: Implementation and Optimization of Neural Networks for Microcontrollers

[![Project Status](https://img.shields.io/badge/Status-Work_in_progress-orange.svg)](https://gitlab.unistra.fr/cerisara/pi_hager_2025)
[![Target](https://img.shields.io/badge/Target-STM32H7%20/%20U5-blue.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32h7-series.html)

This repository contains the work carried out as part of the **Engineering Project (PI)** for the **IR (Informatique et Réseaux)** specialty at Telecom Physique Strasbourg, in collaboration with **Hager Group**.

The project focuses on the transformation of Deep Learning models (developed in PyTorch) into optimized C/C++ code for low-power microcontrollers, specifically the STM32 family.

## 👥 Members
- **Nathan Cerisara** ([nathan.cerisara@etu.unistra.fr](mailto:nathan.cerisara@etu.unistra.fr))
- **Clément Desberg** ([clement.desberg@etu.unistra.fr](mailto:clement.desberg@etu.unistra.fr))
- **Lucas Levy** ([lucas.levy2@etu.unistra.fr](mailto:lucas.levy2@etu.unistra.fr))
- **Ahmed Amine Jadi** ([ahmed-amine.jadi@etu.unistra.fr](mailto:ahmed-amine.jadi@etu.unistra.fr))

**Contacts (Hager Group):** Hien Duc VU, Nicolas BRITSCH

---

## 📜 Project Evolution

The project followed a multi-stage evolution, documented in detailed reports located in the [`doc/`](./doc/) directory.

### 🏁 Phase 1: Conceptualization (Report [R1](./doc/R1_Projet_Ingenieur.md))
- **Goal**: Research tools (TensorFlow Lite, PyTorch Mobile) and define an intermediate language.
- **Key Decision**: Creation of a JSON-based Intermediate Representation (IR) to bridge high-level Python models and low-level code.

### 🧪 Phase 2: Optimization Research (Report [R2](./doc/R2_Projet_Ingenieur.md))
- **Focus**: Weight quantization (INT8), pruning, and knowledge distillation.
- **Architecture**: Choice of **C++** with the **Fastor** library for high-performance tensor operations on microcontrollers.

### 🔍 Phase 3: State of the Art & Pivot (Report [R3](./doc/R3_Projet_Ingenieur.md))
- **Exploration**: Evaluated `Executorch`, `LiteRT`, and `TorchScript`. Most were rejected due to dependency hell or lack of maintenance.
- **Pivot**: Introduction of the **ST Edge AI Developer Cloud** as a primary industry-standard path, while maintaining the manual "ModelBlocks" path for research.

### 🛠️ Phase 4: Dual-Path Implementation (Report [R4](./doc/R4_Projet_Ingenieur.md))
- **Sub-group 1 (ModelBlocks)**: Implemented an AST-based PyTorch parser and a NumPy-backed interpreter to validate the Intermediate Representation.
- **Sub-group 2 (ST Cloud)**: Benchmarking of various architectures (CNN, RNN, Transformers) using the ST proprietary pipeline.

### 🚀 Phase 5: Deployment & Benchmarking (Report [R5](./doc/R5_Projet_Ingenieur.md))
- **Current Process**: PyTorch -> ONNX -> ST Edge AI Cloud -> Optimized C Code -> STM32 Flash.
- **Findings**: Detailed analysis of memory vs. latency trade-offs for different model families (Linear, CNN, RNN, Transformers, Hybrids).

---

## 🏗️ Technical Architecture

The project is split into two complementary approaches:

### 1. The Expert Path: ModelBlocks (Open-Source)
The **ModelBlock** path has been moved to its own repository: [Pytorch_ModelBlock_IntermediateLayer_Conversion](https://github.com/nath54/Pytorch_ModelBlock_IntermediateLayer_Conversion).

The local directory [`onnx_models_conversion_and_benchmark/`](./onnx_models_conversion_and_benchmark/) now serves as the **first stage of the Main Pipeline**:
- **Library of Models**: instead of a single model, we host a library of models to be tested.
- **ONNX Conversion**: converts PyTorch models from the library into ONNX format.
- **Pre-Benchmark**: allows running a preliminary benchmark on the computer to test models before deploying to the STM32.

### 2. The Industrial Path: ST Edge AI Cloud
Located in [`export_model_on_stm32_with_gcc/`](./export_model_on_stm32_with_gcc/), this path uses professional tools for deployment.
- **Pipeline**: Converts ONNX models to optimized C headers using ST's proprietary kernels.
- **Embedded Env**: Integration with `libopencm3` and `arm-none-eabi-gcc` for a purely open-source build system (no dependence on heavy IDEs for production).

### 3. The Modern Path: STM32U5 HAL (CMake)
Located in [`export_model_on_stm32_with_hal_gcc/`](./export_model_on_stm32_with_hal_gcc/), this path targets the **NUCLEO-U545RE-Q** board using the official STM32 HAL.
- **Build System**: Uses **CMake** for a modern, standard build workflow.
- **Automation**: Includes `manage_models.py`, a script to easily select models, switch between INT8/F32 quantization, build, and flash.
- **HAL**: leveraged for peripheral access (GPIO, UART, etc.), offering a more standardized approach than libopencm3 for newer chips.

---

## 📂 Repository Structure

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

## 🔧 Installation & Setup

### Prerequisites
1. **An Ubuntu Computer**
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

## 🔗 Useful Links
- [ST Edge AI Cloud](https://stm32ai-cs.st.com/)

---
© 2024-2025 PI IR Hager Project Team.
