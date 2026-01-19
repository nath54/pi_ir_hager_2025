# Engineering Project : HAGER Group

## Implementation and Optimization of Neural Networks for Microcontrollers

**Members**
Nathan Cerisara : [nathan.cerisara@etu.unistra.fr](mailto:nathan.cerisara@etu.unistra.fr)
Clément Desberg : [clement.desberg@etu.unistra.fr](mailto:clement.desberg@etu.unistra.fr)
Lucas Levy: [lucas.levy2@etu.unistra.fr](mailto:lucas.levy2@etu.unistra.fr)
Ahmed Amine Jadi : [ahmed-amine.jadi@etu.unistra.fr](mailto:ahmed-amine.jadi@etu.unistra.fr)

**Contacts**
Hien Duc Vu : [hienduc.vu@hagergroup.com](mailto:hienduc.vu@hagergroup.com)
Nicolas Britsch : [nicolas.britsch@hagergroup.com](mailto:nicolas.britsch@hagergroup.com)

# **Table of Contents** {#table-of-contents}

[**Table of Contents	2**](#table-of-contents)

[**1\. Introduction	3**](#1.-introduction)

[1.1. Summary	3](#1.1.-summary)

[1.2. Project objectives and history	3](#1.2.-project-objectives-and-history)

[1.2.1. Study of Code Generation Platforms and Tools	3](#1.2.1.-study-of-code-generation-platforms-and-tools)

[1.2.2. Transformation of Python Models into Embedded Code	3](#1.2.2.-transformation-of-python-models-into-embedded-code)

[1.2.3. Simulation of Embedded Models \- ST Edge AI Developer Cloud	3](#1.2.3.-simulation-of-embedded-models---st-edge-ai-developer-cloud)

[1.2.4. Evaluation and Deployment	3](#1.2.4.-evaluation-and-deployment)

[**2\. Current Process	4**](#2.-current-process)

[2.1. Global Scheme	4](#2.1.-global-scheme)

[2.2. Pytorch to ONNX Conversion	4](#2.2.-pytorch-to-onnx-conversion)

[2.3. ST Edge AI Developer Cloud process	4](#2.3.-st-edge-ai-developer-cloud-process)

[2.4. C Code Project	5](#2.4.-c-code-project)

[**3\. Benchmarking Models	6**](#3.-benchmarking-models)

[3.1. Issue with the given model from the client	6](#3.1.-issue-with-the-given-model-from-the-client)

[3.2. New models tested	6](#3.2.-new-models-tested)

[Models generated and tested can be grouped into five categories:	6](#models-generated-and-tested-can-be-grouped-into-five-categories:)

[1\) Linear Models, MLPs and Global-Statistics Architectures	6](#1\)-linear-models,-mlps-and-global-statistics-architectures)

[2\) Convolution-based Architectures (CNNs)	7](#2\)-convolution-based-architectures-\(cnns\))

[3\) Recurrent Networks (RNN / GRU / LSTM)	7](#3\)-recurrent-networks-\(rnn-/-gru-/-lstm\))

[4\) Attention-based and Transformer Models	7](#4\)-attention-based-and-transformer-models)

[5\) Hybrid Architectures	8](#5\)-hybrid-architectures)

[3.3. Benchmarking on STM32 Edge AI developer Cloud	8](#3.3.-benchmarking-on-stm32-edge-ai-developer-cloud)

[This benchmarking step provided:	9](#this-benchmarking-step-provided:)

[**4\. Future and final objectives	10**](#4.-future-and-final-objectives)

[4.1. Measuring inference time with an oscilloscope	10](#4.1.-measuring-inference-time-with-an-oscilloscope)

[4.2. Benchmarking all the models types and getting the limits	10](#4.2.-benchmarking-all-the-models-types-and-getting-the-limits)

[4.3. Testing the capabilities of each models families with a small training set	10](#heading=h.yna68um8b25h)

[**5\. Conclusion	10**](#5.-conclusion)

# **1\. Introduction** {#1.-introduction}

## 1.1. Summary {#1.1.-summary}

Our project explores and implements various types and architectures of neural networks on ST microcontrollers, with a particular focus on optimization and the generation of embedded code. The goal is to transform Python models developed on PyTorch into efficient code for microcontrollers, while leveraging hardware accelerators provided by manufacturers. The objective is also to address simulation to test models before hardware deployment, as well as the evaluation of performance, cost, and energy consumption of the deployed solutions.

## 1.2. Project objectives and history {#1.2.-project-objectives-and-history}

### 1.2.1. Study of Code Generation Platforms and Tools {#1.2.1.-study-of-code-generation-platforms-and-tools}

- Exploration of conversion tools (such as TensorFlow Lite, PyTorch Mobile, etc.).
- Understanding of manufacturer-specific deployment libraries.
- Already done, with unsatisfying results.

### 1.2.2. Transformation of Python Models into Embedded Code {#1.2.2.-transformation-of-python-models-into-embedded-code}

- Conversion of machine learning models into optimized code for microcontrollers.
- Management of specific constraints (quantization, limited memory).
- Done with two steps : transformation of python model to C++ code with ST Edge AI Cloud, compilation of C++ code for the STM32 microcontroller done manually

### 1.2.3. Simulation of Embedded Models \- ST Edge AI Developer Cloud {#1.2.3.-simulation-of-embedded-models---st-edge-ai-developer-cloud}

- Use of hardware simulators to predict performance before deployment.
- Functional validation of models on virtual test benches.

### 1.2.4. Evaluation and Deployment {#1.2.4.-evaluation-and-deployment}

- Analysis of performance (accuracy, latency, energy consumption).
- Comparison of results across different hardware architectures.

# **2\. Current Process** {#2.-current-process}

## 2.1. Global Scheme {#2.1.-global-scheme}

Our current global process is organized as following:

**Step 1:** Convert the pytorch code into ONNX
**Step 2:** Upload the ONNX on the ST Edge AI Developer Cloud app
**Step 3:** Apply all the optimisations and optionally apply the quantization method on the model
**Step 4:** Export the C code from the ST Edge AI Developer Cloud app
**Step 5:** Include the C code into our C project
**Step 6:** Compile the project, flash it into the board and run it

## 2.2. Pytorch to ONNX Conversion {#2.2.-pytorch-to-onnx-conversion}

With the use of the ONNX library in Python, we can convert pytorch model into onnx. We choose to not use the latest onnx version for better support with ST Edge AI Developer Cloud to solve previously encountered compatibility issues.

## 2.3. ST Edge AI Developer Cloud process {#2.3.-st-edge-ai-developer-cloud-process}

After converting the PyTorch model into ONNX format, the next step of the workflow consist of processing the exported model using **ST Edge AI Developer Cloud**, a tool specifically designed for embedded deployment. This platform enabled evaluation, optimization, quantization, and automatic C-code generation tailored for STM32 microcontrollers.

The workflow carried out during this project follows the sequence:

1. **Upload ONNX Model**
    The converted ONNX model is imported into the ST Edge AI Cloud interface, where the tool automatically parses network structure, input shape, operators and dependency layers.

2. **Platform Compatibility Check**
    The cloud tool verifies whether the model could run on STM32-class hardware, identifying unsupported operators and estimating the required Flash and RAM consumption.

3. **Optimization & Quantization**
    Several techniques are applied to reduce model size and enable faster inference, including parameter compression and conversion to **INT8 quantized format**, significantly lowering memory usage compared to FP32.

4. **Benchmark Simulation**
    Before deployment, the platform internally profiles the model to predict latency, memory footprint, and throughput for different microcontroller targets, which guides the selection of deployable architectures.

5. **C Code Export**
    Once validated, the platform generated optimized C code, including the model weights, inference kernels, and execution functions in a format directly compilable within STM32CubeIDE or open-source arm-gcc compiler.

The use of ST Edge AI Developer Cloud has proved to be essential in the scalability of this project, enabling fast iteration between models and efficient testing.

## 2.4. C Code Project {#2.4.-c-code-project}

Once the optimized code was generated by ST Edge AI Cloud, the final step of the deployment pipeline was the creation of a fully functional C-based firmware capable of performing on-device inference.

We have two ways of compiling and flashing the exported model C code into a STM32:

- Via STM32 CUBE IDE
- Via arm-gcc and libopencm3

The procedure implemented during this step was structured as follows:

1. **Code Integration into STM32 Project Structure**
    The generated model files (headers, static arrays, inference engine modules) were inserted into a dedicated directory inside the STM32CubeIDE project. The wrapper functions provided by ST were included to handle tensor I/O and model execution.

2. **Compilation and Build Configuration**
    The firmware was compiled with optimization flags enabled (-Os to reduce the binary size), ensuring reduced Flash usage.

3. **Deployment on STM32 Hardware**
    The binary firmware was flashed onto the microcontroller. The model was executed using randomly generated inputs, enabling the measurement of inference duration.

The C code project stage therefore represents the convergence of the entire workflow, transforming a high-level neural network into executable embedded code running autonomously on constrained hardware.

# **3\. Benchmarking Models** {#3.-benchmarking-models}

## 3.1. Issue with the given model from the client {#3.1.-issue-with-the-given-model-from-the-client}

The first model provided by the client could not be deployed on the STM32 target for two main reasons:

1. **Model size was significantly above available Flash and RAM limits**
    The network contained a large number of parameters, and the exported binary exceeded the typical memory capacity of STM32 MCUs, making deployment infeasible without aggressive compression or pruning.

2. **Layer compatibility problems were encountered**
    Some of the layers (non-standard convolution blocks, activation functions or tensor reshaping operations) were not supported by the ST Edge AI conversion pipeline. As a result, inference code could not be generated, preventing full execution on the microcontroller.

This step confirmed that not all desktop-grade architectures can be directly transferred to embedded systems, and justified the need to design and evaluate alternative model structures.

## 3.2. New models tested {#3.2.-new-models-tested}

After determining that the original client-provided network could not be deployed due to memory size and operator incompatibility, several alternative architectures were designed and evaluated. These new models focus on reduced parameter count, simplified topology, and structural compatibility with STM32 Edge AI acceleration kernels.

The goal of this exploration was not to train models for accuracy, but to **produce deployable architectures** that could pass conversion, benchmarking, and quantization steps reliably.
 Each architecture variant was intentionally generated with incremental scaling in depth, width, and structural complexity to progressively determine the limits of STM32 deployability — forming the comparative basis for the benchmarking stage shown later.

### **Models generated and tested can be grouped into five categories:** {#models-generated-and-tested-can-be-grouped-into-five-categories:}

### **1\) Linear Models, MLPs and Global-Statistics Architectures** {#1)-linear-models,-mlps-and-global-statistics-architectures}

This first family includes all non-convolutional and non-recurrent architectures, and was used as the experimental baseline.
 It can itself be divided into three sub-groups:

* **Simple Fully Connected Baselines**
   *simple\_lin\_1, simple\_lin\_2, simple\_lin\_N, single\_layer\_perceptron*
   → extremely lightweight, sub-millisecond runtime, ideal for microcontrollers.

* **Deeper MLPs with Flatten Stages**
   *mlp\_flatten\_first, factorized, parallel\_feature\_extractors*
   → increased representational power at the cost of a larger parameter count.

* **Global-Pooling / Statistical Feature Compression Models**
   *global\_statistics\_extractor, global\_avg\_pooling, max\_pooling, mixed\_pooling*
   → reduce input dimensionality, preserving performance while minimizing memory load.

These models provide an excellent performance-to-size ratio and are highly suitable for embedded deployment.

### **2\) Convolution-based Architectures (CNNs)** {#2)-convolution-based-architectures-(cnns)}

This family covers both 1D and 2D convolutional models – from standard convolution to depthwise-separable and multi-scale variations:

*conv1d, conv1d\_features, conv2d\_standard, conv2d\_depthwise\_sep, multi\_scale\_cnn, stacked\_conv2d, residual\_cnn.*

CNNs offer a strong balance between accuracy and efficiency, although scaling up the number of filters can quickly increase memory and compute requirements.

### **3\) Recurrent Networks (RNN / GRU / LSTM)** {#3)-recurrent-networks-(rnn-/-gru-/-lstm)}

These architectures target sequential and temporal processing:

*simple\_rnn, lstm, gru, bidirectionnal\_lstm, bidirectionnal\_gru.*

They model time dependencies effectively, but consume more RAM/Flash than purely linear or convolutional models.

### **4\) Attention-based and Transformer Models** {#4)-attention-based-and-transformer-models}

Modern architectures leveraging attention mechanisms fall into this category:

*self\_attention, lightweight\_transformer, vision\_transformer.*

These models are highly expressive and generalizable, but attention layers remain challenging to deploy on low-resource devices.

### **5\) Hybrid Architectures** {#5)-hybrid-architectures}

This family combines multiple paradigms – convolution, recurrence and attention – to maximize representational power while retaining computational efficiency:

*cnn\_rnn\_hybrid, cnn\_lstm\_hybrid, cnn\_attention, temporal\_cnn, senet.*

These represent the most advanced exploration front, aiming to merge temporal modeling, feature extraction and contextual weighting in a single unified architecture.

## 3.3. Benchmarking on STM32 Edge AI developer Cloud {#3.3.-benchmarking-on-stm32-edge-ai-developer-cloud}

After generating multiple candidate architectures, each model was evaluated using **STM32 Edge AI Developer Cloud**, which served as the primary environment for deployment feasibility analysis. The objective of this stage was not to measure real hardware execution or model generalization performance, but rather to determine whether each model could be executed on STM32 microcontrollers based on **memory footprint, operator support, and estimated inference latency**.

The benchmarking workflow followed this sequence:

1. **Upload ONNX model into ST Edge AI Cloud**
    The exported models were imported individually, allowing the platform to analyse layer structure, compute graph depth, and memory allocation.

2. **Automatic feasibility evaluation**
    The tool generated an immediate compatibility report, flagging models that:

   * exceeded Flash or RAM capacity

   * contained unsupported operators

   * required tensor operations not available in MCU kernels

3. **Memory and latency profiling**
    For each acceptable model, the Cloud produced static estimates of:

   * Flash size required to store weights \+ compiled model

   * RAM required at runtime for intermediate tensors

   * predicted inference time per forward pass

4. **Export of benchmarking report**
    The system generated a summary of deployability, identifying which architectures could run on STM32 without modification and which would require pruning, quantization, or structural redesign.

**Output of this Step**

### This benchmarking step provided: {#this-benchmarking-step-provided:}

| Output                          | Meaning                                                       |
| ------------------------------- | ------------------------------------------------------------- |
| **Raw Flash & RAM usage**       | Determines whether the model fits inside MCU model boundaries |
| **Estimated inference latency** | Gives a pre-deployment indication of real-time viability      |
| **Compatibility status**        | Confirms whether ST kernels support the model’s operations    |
| **Compatibility status**        | Separates deployable vs non-deployable architectures          |

###

Before presenting the results, it is important to note that the benchmarking was conducted on three STM32 development boards — **B-U585I-IOT02A**, **NUCLEO-G474RE**, and **NUCLEO-H743ZI2**. These platforms share architectural similarities with the final deployment target, allowing the benchmarking results to accurately reflect realistic inference performance and memory constraints during embedded execution. The benchmarking was not conducted on the exact destination platform because it is not available as a virtual benchmark (but still available as an export platform).

Importantly, this step did not attempt to improve or modify models — it only assessed them as they exist.

### ***Result Data Extract***

# **4\. Future and final objectives** {#4.-future-and-final-objectives}

## 4.1. Measuring inference time with an oscilloscope {#4.1.-measuring-inference-time-with-an-oscilloscope}

For the future of the project we need a precise measure of the inference time on a STM32. We need to do this for two main reasons, to test our pipeline on a “real” model and to measure with an oscilloscope the differences between the test we made on the benchmark of ST-Edge AI DC and reality with the ST32 we physically possess.
We need to be able to assure the client that our measurements are correct.

## 4.2. Benchmarking all the models types and getting the limits {#4.2.-benchmarking-all-the-models-types-and-getting-the-limits}

To see the limits of the model we need to test how the ST-Edge DC optimises and run it on variable types of ONNX models. This is to tell Hager the maximum size of the model and what type of structure is more optimised for our case.

# **5\. Conclusion** {#5.-conclusion}

In conclusion, we succeeded in converting a pytorch model in a C code or a .elf file using ST-EdgeAI DC.
This allowed us to test with the client’s given model and even with quantization and optimization, but it was still too needy in terms of RAM and Flash.
To explore the limits of the given architecture, we made an extensive set of tests using the benchmark tool of ST-EdgeAI DC, which gave us strong hints for the model maximum size and architecture. In parallel, we tried to export the model on the real ST32 to measure the differences between the benchmark and reality.
This will be our primary focus until we can validate the precision of the benchmark’s results.