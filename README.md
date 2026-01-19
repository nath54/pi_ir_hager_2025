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

---

## 📂 Repository Structure

```text
.
├── doc/                                # Project reports (R1-R5) and format guides
├── export_model_on_stm32_with_gcc/     # STM32 C implementation and build system
│   ├── main.c                          # Main entry point for the MCU
│   ├── manage_models.py                # Automation script for flashing/debugging
│   ├── Makefile                        # ARM GCC build configuration
│   └── network.c                       # Generated model code (from ST Cloud)
├── onnx_models_conversion_and_benchmark/ # PyTorch -> ONNX conversion and pre-benchmarking
│   ├── lib_onnx_convert.py             # PyTorch -> ONNX utility
│   ├── lib_model_loader.py             # Architecture loading
│   └── script_test_and_measures.py     # Local benchmarking script
└── export_model_on_stm32_with_hal_gcc/ # Alternative HAL-based implementation
```

---

## 🔧 Installation & Setup

### Prerequisites
1. **Python 3.10+** (3.10 recommended for some ST tools compatibility).
2. **ARM GNU Toolchain**: `arm-none-eabi-gcc` for compiling for STM32.
3. **st-link**: To flash the binary onto the boards.
4. **libopencm3**: Included as a submodule.

### Commands
```bash
# Clone the repository
git clone https://gitlab.unistra.fr/cerisara/pi_hager_2025.git
cd pi_hager_2025

# Setup the STM32 build environment
cd export_model_on_stm32_with_gcc
bash build.sh

# Flash a model
bash flash.sh
```

---

## 📊 Benchmarking Highlights

The project evaluated several model families on **STM32H7** and **U5** boards.
- **Linear Models**: < 1ms inference, very low memory (ideal for simple sensors).
- **CNNs**: Balanced performance, but memory intensive for high filter counts.
- **RNNs/LSTMs**: Good temporal modeling, slightly higher latency in conversion.
- **Transformers/Attention**: High representational power, currently at the limit of embedded deployment (high RAM requirements).

---

## 🔗 Useful Links
- [ST Edge AI Cloud](https://stm32ai-cs.st.com/)
- [libopencm3 Documentation](http://libopencm3.org/)
- [Fastor Library](https://github.com/romeric/Fastor)

---
© 2024-2025 PI IR Hager Project Team.
