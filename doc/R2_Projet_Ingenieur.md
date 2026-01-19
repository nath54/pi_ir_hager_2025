
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

[1.2. Project objectives	3](#1.2.-project-objectives)

[1.2.1. Study of Code Generation Platforms and Tools	3](#1.2.1.-study-of-code-generation-platforms-and-tools)

[1.2.2. Transformation of Python Models into Embedded Code	3](#1.2.2.-transformation-of-python-models-into-embedded-code)

[1.2.3. Implementation and Optimization of Models	3](#1.2.3.-implementation-and-optimization-of-models)

[1.2.4. Simulation of Embedded Models	3](#1.2.4.-simulation-of-embedded-models)

[1.2.5. Evaluation and Deployment	3](#1.2.5.-evaluation-and-deployment)

[1.2.6. Detailed Reporting and Recommendations	3](#1.2.6.-detailed-reporting-and-recommendations)

[**2\. Optimizations	4**](#2.-optimizations)

[2.1. Optimizations on Base Weights	4](#2.1.-optimizations-on-base-weights)

[2.2. Optimization on data input	4](#2.2.-optimization-on-data-input)

[2.3. Optimizations on Model Architecture	4](#2.3.-optimizations-on-model-architecture)

[2.4. Optimization on Code Logic	5](#2.4.-optimization-on-code-logic)

[2.5. Optimization on Hardware	5](#2.5.-optimization-on-hardware)

[**3\. Project architecture	6**](#3.-project-architecture)

[3.1. Project architecture	6](#3.1.-project-architecture)

[3.1.1.Parsing and Conceptualisation	6](#3.1.1.parsing-and-conceptualisation)

[3.1.2. Optimizations	7](#3.1.2.-optimizations)

[3.1.3. Generating target language code	7](#3.1.3.-generating-target-language-code)

[3.2. Neural Blocks architecture	7](#3.2.-neural-blocks-architecture)

[**4\. Work organisation	8**](#4.-work-organisation)

[4.1. Communication Graph	8](#4.1.-communication-graph)

[**5\. Target language changes	9**](#5.-target-language-changes)

[5.1. Change to C++	9](#5.1.-change-to-c++)

[5.2. Tensor calculus library	9](#5.2.-tensor-calculus-library)


# **1\. Introduction** {#1.-introduction}

## 1.1. Summary {#1.1.-summary}

This project explores and implements various types and architectures of neural networks on microcontrollers, with a particular focus on optimization and the generation of embedded code. The goal is to transform Python models developed on frameworks such as TensorFlow or PyTorch into efficient code for microcontrollers, while leveraging hardware accelerators provided by manufacturers. The project will also address simulation to test models before hardware deployment, as well as the evaluation of performance, cost, and energy consumption of the deployed solutions.

## 1.2. Project objectives {#1.2.-project-objectives}

### 1.2.1. Study of Code Generation Platforms and Tools {#1.2.1.-study-of-code-generation-platforms-and-tools}

- Exploration of conversion tools (such as TensorFlow Lite, PyTorch Mobile, etc.).
- Understanding of manufacturer-specific deployment libraries.

### 1.2.2. Transformation of Python Models into Embedded Code {#1.2.2.-transformation-of-python-models-into-embedded-code}

- Conversion of machine learning models into optimized code for microcontrollers.
- Management of specific constraints (quantization, limited memory).

### 1.2.3. Implementation and Optimization of Models {#1.2.3.-implementation-and-optimization-of-models}

- Implementation of models such as CNN, RNN, MLP, and Transformers.
- Compression and optimization for performance and energy constraints.

### 1.2.4. Simulation of Embedded Models {#1.2.4.-simulation-of-embedded-models}

- Use of hardware simulators to predict performance before deployment.
- Functional validation of models on virtual test benches.

### 1.2.5. Evaluation and Deployment {#1.2.5.-evaluation-and-deployment}

- Analysis of performance (accuracy, latency, energy consumption).
- Comparison of results across different hardware architectures.

### 1.2.6. Detailed Reporting and Recommendations {#1.2.6.-detailed-reporting-and-recommendations}

- Documentation of steps, tools used, results obtained, and limitations.
- Recommendations for future improvements.

# **2\. Optimizations** {#2.-optimizations}

The fact that they're listed here doesn't mean we're going to implement them, but it does allow us to check that our project architecture is compatible with the optimizations.

## 2.1. Optimizations on Base Weights {#2.1.-optimizations-on-base-weights}

- **Quantization**: Quantization reduces the precision of neural network weights from floating-point to lower-bit representations, such as 8-bit integers, to decrease model size and accelerate inference. This technique maintains acceptable accuracy while enabling efficient deployment on resource-constrained devices like mobile systems.  [\[1\].](#[1].-nagel,-m.,-fournarakis,-m.,-amjad,-r.-a.,-bondarenko,-y.,-mart,-v.-b.,-&-blankevoort,-t.-\(2021,-june-15\).-a-white-paper-on-neural-network-quantization.-arxiv.org.-https://arxiv.org/abs/2106.08295)

## 2.2. Optimization on data input {#2.2.-optimization-on-data-input}

- **Data pre-processing**: Data pre-processing transforms raw input data into a structured format through techniques like normalization, cleaning, and subsampling, enhancing model training efficiency. For instance, subsampling reduces dataset size by selecting representative samples, tailored to specific model architectures, thereby lowering computational demands. [\[2\].](#[2].-nagori,-m.,-jain,-a.,-&-jain,-r.-\(2022\).-a-review:-data-pre-processing-and-data-augmentation-techniques.-in-artificial-intelligence-in-the-age-of-neural-networks-and-brain-computing-\(pp.-1-24\).-elsevier.-https://www.sciencedirect.com/science/article/pii/s2666285x22000565)

## 2.3. Optimizations on Model Architecture {#2.3.-optimizations-on-model-architecture}

The model will need to be trained again if using any of these methods, because they introduce a change in the model architecture and the pretrained weights.

- **Distillation**: Knowledge distillation transfers expertise from a large, complex teacher model to a smaller, simpler student model, improving efficiency without significant accuracy loss. This process involves retraining the student to mimic the teacher’s outputs, optimizing for deployment on less powerful hardware. [\[3\].](#[3].-hinton,-g.,-vinyals,-o.,-&-dean,-j.-\(2015\).-distilling-the-knowledge-in-a-neural-network.-arxiv-preprint-arxiv:1503.02531.-https://arxiv.org/abs/1503.02531)
- **Attention Head Pruning**: Attention head pruning removes redundant or less impactful attention heads in transformer models, reducing computational complexity while preserving performance. Retraining adjusts the model to maintain effectiveness, making it suitable for resource-efficient applications. [\[4\].](#[4].-su,-j.,-chen,-y.,-wang,-w.,-hu,-y.,-li,-y.,-wang,-x.,-wang,-t.,-luo,-z.,-&-xu,-b.-\(2021\).-layer-wise-pruning-of-transformer-attention-heads-for-efficient-language-modeling.-arxiv-preprint-arxiv:2110.03252.-https://arxiv.org/abs/2110.03252)
- **Dimension Reduction**: Dimension reduction decreases the number of parameters or features in a model, such as through techniques like PCA, to simplify architecture and speed up training. This requires retraining to adapt the reduced model to the original task, balancing efficiency and accuracy. [\[5\].](#[5].-rajesh,-k.,-kumar,-p.-s.,-&-nagori,-m.-\(2020\).-a-review-of-dimensionality-reduction-techniques-for-efficient-computation.-procedia-computer-science,-171,-106-113.-https://www.sciencedirect.com/science/article/pii/s1877050920300879)
- **Use of better model architectures**: Adopting advanced architectures, such as transformers or efficient convolutional networks, enhances performance by leveraging optimized design principles. Retraining integrates these structural improvements, maximizing accuracy and computational efficiency for specific tasks. [\[6\].](#[6].-abdellatif,-a.-a.,-elngar,-a.-a.,-ghoneim,-a.-m.,-soliman,-a.-m.,-tolba,-a.-y.,-elsayed,-a.-a.,-...-&-al-barakati,-a.-\(2024\).-a-comprehensive-review-of-deep-learning:-architectures,-recent-advances,-and-applications.-machines,-12\(1\),-75.-https://www.mdpi.com/2078-2489/15/12/755)

#
## 2.4. Optimization on Code Logic {#2.4.-optimization-on-code-logic}

- **Flow Control Optimization**: Flow control optimization enhances program efficiency by restructuring execution paths, such as optimizing loops or conditionals, to minimize runtime overhead in machine learning code. This technique improves algorithm performance without altering model architecture, focusing on software-level gains.
- **Tensor Libraries Optimization**: Tensor libraries optimization refines frameworks like TensorFlow or PyTorch to execute tensor operations more efficiently, reducing latency and resource use. It leverages advanced scheduling and memory management to accelerate computations critical to neural network training and inference.

## 2.5. Optimization on Hardware {#2.5.-optimization-on-hardware}

- **Use of Hardware Accelerators**: Hardware accelerators, such as GPUs or TPUs, boost computation speed by parallelizing matrix operations inherent in neural networks. This optimization targets hardware-level performance, significantly reducing training and inference times.
- **Cache / RAM \- data control**: Cache and RAM data control optimizes memory access patterns to minimize latency and maximize throughput during model execution. Efficient data staging and retrieval enhance hardware utilization, critical for large-scale machine learning workloads.


# **3\. Project architecture** {#3.-project-architecture}

## 3.1. Project architecture {#3.1.-project-architecture}

![][image1]

There are three main Steps that are conceived in our project, and they all work in tandem, each step is dependent on another's output.
Our three main steps are :

### 3.1.1.Parsing and Conceptualisation {#3.1.1.parsing-and-conceptualisation}

**Parsing**

As previously mentioned in Project architecture, this step consists of parsing all information from the trained models, which is the structure of the pytorch model and the weights, and converting that information to usable classes in our program.

**Conceptualisation**

This part of our first step is crucial for the transfer of information between the different steps. This part consists of creating classes used to embed the different functions of pytorch
(such as softmax, linear, relu, Conv2D ...) and to standardize their usage, as well as classes that handle Tensors, operations on tensors with precision to dimensionality, and all the tools needed to successfully translate the model to our target language.

### 3.1.2. Optimizations {#3.1.2.-optimizations}

This step works with the neural network representation that has been extracted. Different types of optimizations can be applied on it, depending on the observed model architecture. For instance, if the model has MultiAttentionHeads, it is possible to prune some of the heads, or to quantize the model.

### 3.1.3. Generating target language code {#3.1.3.-generating-target-language-code}

This final step involves translating the structured representation of the model into the desired target language. The key task is to generate code from the conceptualisation we obtain after the optimisation steps, preserving the computational logic. At the end of this phase, we obtain a fully functional implementation of the trained model in our target language, ready for execution and further testing.

## 3.2. Neural Blocks architecture {#3.2.-neural-blocks-architecture}

To represent the model in memory, an architecture very close to the base Python code has been used. The Model Blocks represent the Python classes that inherit the Pytorch nn.Module class. They contain functions and layers, and the functions contain a list of flow control operations, that represent operations, variable manipulation, functions call, basic loop and basic conditions.

# **4\. Work organisation** {#4.-work-organisation}

## **4.1. Communication Graph** {#4.1.-communication-graph}

#

# **5\. Target language changes** {#5.-target-language-changes}

## 5.1. Change to C++ {#5.1.-change-to-c++}

In the first project review, we stated that the target language will be assembly or C. We later changed it to C++ for two main reason :

- We need to keep a language that can be compiled, and therefore we need to choose a language for which we have a compiler (C++ is one of those languages).
- We realised that having objects will make the code much clearer and the project easier, without any impact on performance as mentioned above.
- Using C++ makes it easier to use libraries like mentioned in the next part.

## 5.2. Tensor calculus library {#5.2.-tensor-calculus-library}

To understand tensor calculus, we used various articles, such as Introduction to Tensor Calculus [\[7\]](#[7].-taha-sochi-\(may-25,-2016\).-introduction-to-tensor-calculus.-department-of-physics-&-astronomy,-university-college-london,-gower-street,-london,-wc1e-6bt.https://arxiv.org/pdf/1603.01660).
We find some C++ libraries dedicated to tensor implementations, such as tensorlite [\[8\]](#heading=h.kkbqwdtcjrez) which have some functions but not every function we need, or ggml [\[9\]](#[9].-sales@ggml.ai.-ggml.ggml-org/ggml:-tensor-library-for-machine-learning) which is more complete.
We chose the Fastor [\[10\]](#[10].-poya,-roman-and-gil,-antonio-j.-and-ortigosa,-rogelio-\(2017\).-a-high-performance-data-parallel-tensor-contraction-framework:-application-to-coupled-electro-mechanics.-computer-physics-communications.-http://www.sciencedirect.com/science/article/pii/s0010465517300681.-romeric/fastor:-a-lightweight-high-performance-tensor-algebra-framework-for-modern-c++) library over many alternatives because of its licence, optimisations toward microcontrollers and ease of use.
Fastor [\[10\]](#[10].-poya,-roman-and-gil,-antonio-j.-and-ortigosa,-rogelio-\(2017\).-a-high-performance-data-parallel-tensor-contraction-framework:-application-to-coupled-electro-mechanics.-computer-physics-communications.-http://www.sciencedirect.com/science/article/pii/s0010465517300681.-romeric/fastor:-a-lightweight-high-performance-tensor-algebra-framework-for-modern-c++) is provided under the MIT license, and is therefore free of use as long as we mention the author and the original license under which the library is provided.
The library also doesn't have any dependencies, thus facilitating its installation and use.
It is a lightweight library suitable for microcontrollers, as stated on its main page. Its extensive wiki also makes it easily usable.

Some functions necessary to neural networks are not directly provided by the library, thus we will implement them using the basic functions from the library, and will maintain a time and memory efficiency.

###
- #### \[1\]. Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., Mart, V. B., & Blankevoort, T. (2021, June 15). *A white paper on Neural Network Quantization*. arXiv.org. [https://arxiv.org/abs/2106.08295](https://arxiv.org/abs/2106.08295) {#[1].-nagel,-m.,-fournarakis,-m.,-amjad,-r.-a.,-bondarenko,-y.,-mart,-v.-b.,-&-blankevoort,-t.-(2021,-june-15).-a-white-paper-on-neural-network-quantization.-arxiv.org.-https://arxiv.org/abs/2106.08295}

- #### \[2\]. Nagori, M., Jain, A., & Jain, R. (2022). A review: Data pre-processing and data augmentation techniques. In *Artificial Intelligence in the Age of Neural Networks and Brain Computing* (pp. 1-24). Elsevier. [https://www.sciencedirect.com/science/article/pii/S2666285X22000565](https://www.sciencedirect.com/science/article/pii/S2666285X22000565) {#[2].-nagori,-m.,-jain,-a.,-&-jain,-r.-(2022).-a-review:-data-pre-processing-and-data-augmentation-techniques.-in-artificial-intelligence-in-the-age-of-neural-networks-and-brain-computing-(pp.-1-24).-elsevier.-https://www.sciencedirect.com/science/article/pii/s2666285x22000565}

- #### \[3\]. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531. [https://arXiv.org/abs/1503.02531](https://arXiv.org/abs/1503.02531) {#[3].-hinton,-g.,-vinyals,-o.,-&-dean,-j.-(2015).-distilling-the-knowledge-in-a-neural-network.-arxiv-preprint-arxiv:1503.02531.-https://arxiv.org/abs/1503.02531}

- #### \[4\]. Su, J., Chen, Y., Wang, W., Hu, Y., Li, Y., Wang, X., Wang, T., Luo, Z., & Xu, B. (2021). Layer-wise Pruning of Transformer Attention Heads for Efficient Language Modeling. arXiv preprint arXiv:2110.03252. [https://arXiv.org/abs/2110.03252](https://arXiv.org/abs/2110.03252) {#[4].-su,-j.,-chen,-y.,-wang,-w.,-hu,-y.,-li,-y.,-wang,-x.,-wang,-t.,-luo,-z.,-&-xu,-b.-(2021).-layer-wise-pruning-of-transformer-attention-heads-for-efficient-language-modeling.-arxiv-preprint-arxiv:2110.03252.-https://arxiv.org/abs/2110.03252}

- #### \[5\]. Rajesh, K., Kumar, P. S., & Nagori, M. (2020). A review of dimensionality reduction techniques for efficient computation. *Procedia Computer Science*, 171, 106-113. [https://www.sciencedirect.com/science/article/pii/S1877050920300879](https://www.sciencedirect.com/science/article/pii/S1877050920300879) {#[5].-rajesh,-k.,-kumar,-p.-s.,-&-nagori,-m.-(2020).-a-review-of-dimensionality-reduction-techniques-for-efficient-computation.-procedia-computer-science,-171,-106-113.-https://www.sciencedirect.com/science/article/pii/s1877050920300879}

- #### \[6\]. Abdellatif, A. A., Elngar, A. A., Ghoneim, A. M., Soliman, A. M., Tolba, A. Y., Elsayed, A. A., ... & Al-Barakati, A. (2024). A comprehensive review of deep learning: architectures, recent advances, and applications. *Machines*, 12(1), 75\. [https://www.mdpi.com/2078-2489/15/12/755](https://www.mdpi.com/2078-2489/15/12/755) {#[6].-abdellatif,-a.-a.,-elngar,-a.-a.,-ghoneim,-a.-m.,-soliman,-a.-m.,-tolba,-a.-y.,-elsayed,-a.-a.,-...-&-al-barakati,-a.-(2024).-a-comprehensive-review-of-deep-learning:-architectures,-recent-advances,-and-applications.-machines,-12(1),-75.-https://www.mdpi.com/2078-2489/15/12/755}

- #### \[7\]. Taha Sochi (May 25, 2016). Introduction to Tensor Calculus. Department of Physics & Astronomy, University College London, Gower Street, London, WC1E 6BT.[https://arxiv.org/pdf/1603.01660](https://arxiv.org/pdf/1603.01660) {#[7].-taha-sochi-(may-25,-2016).-introduction-to-tensor-calculus.-department-of-physics-&-astronomy,-university-college-london,-gower-street,-london,-wc1e-6bt.https://arxiv.org/pdf/1603.01660}

- #### \[8\]. [@gautamsharma](https://www.github.com/gautam-sharma1). Tensorlite. [ggsharma/tensorlite: A lightweight C++ library for tensors](https://github.com/ggsharma/tensorlite)

- #### \[9\]. [sales@ggml.ai](mailto:sales@ggml.ai). GGML.[ggml-org/ggml: Tensor library for machine learning](https://github.com/ggml-org/ggml) {#[9].-sales@ggml.ai.-ggml.ggml-org/ggml:-tensor-library-for-machine-learning}

- #### \[10\]. Poya, Roman and Gil, Antonio J. and Ortigosa, Rogelio (2017). A high performance data parallel tensor contraction framework: Application to coupled electro-mechanics. Computer Physics Communications. [http://www.sciencedirect.com/science/article/pii/S0010465517300681](http://www.sciencedirect.com/science/article/pii/S0010465517300681). [romeric/Fastor: A lightweight high performance tensor algebra framework for modern C++](https://github.com/romeric/Fastor?tab=readme-ov-file) {#[10].-poya,-roman-and-gil,-antonio-j.-and-ortigosa,-rogelio-(2017).-a-high-performance-data-parallel-tensor-contraction-framework:-application-to-coupled-electro-mechanics.-computer-physics-communications.-http://www.sciencedirect.com/science/article/pii/s0010465517300681.-romeric/fastor:-a-lightweight-high-performance-tensor-algebra-framework-for-modern-c++}

[image1]: <data:image/png;base64,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>