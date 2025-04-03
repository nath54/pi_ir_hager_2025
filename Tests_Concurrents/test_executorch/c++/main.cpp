#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h> // For executorch_init()
#include <executorch/runtime/platform/log.h> // For ET_LOG
#include <executorch/runtime/core/exec_aten/exec_aten.h> // For Tensor
#include <executorch/runtime/core/exec_aten/util/tensor_util.h> // For tensor creation/access utils
#include <executorch/extension/data_loader/buffer_data_loader.h> // For loading from memory buffer
#include <executorch/runtime/core/memory_allocator.h> // For MemoryAllocator, MemoryManager
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h> // For MallocMemoryAllocator

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono> // For timing

// Helper function to load file content into a buffer
std::vector<char> loadFileBuffer(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        ET_LOG(Error, "Failed to open file: %s", path);
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
         ET_LOG(Error, "Failed to read file: %s", path);
        return {};
    }
    ET_LOG(Info, "Successfully loaded %s (%zu bytes)", path, buffer.size());
    return buffer;
}

// Helper to print tensor details (optional)
void printTensorDetails(const torch::executor::Tensor& t, const std::string& name) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < t.dim(); ++i) {
        std::cout << t.size(i) << (i == t.dim() - 1 ? "" : ", ");
    }
    std::cout << "], dtype: " << t.scalar_type();
    std::cout << ", numel: " << t.numel() << std::endl;
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.pte>" << std::endl;
        return 1;
    }
    const char* model_path = argv[1];

    // 1. Initialize ExecuTorch Runtime
    // This must be called before using most ExecuTorch APIs.
    torch::executor::runtime_init();

    // 2. Load the .pte File into a Memory Buffer
    auto pte_buffer = loadFileBuffer(model_path);
    if (pte_buffer.empty()) {
        return 1;
    }

    // 3. Create a DataLoader for the Program
    // BufferDataLoader loads from an existing memory buffer.
    auto data_loader = torch::executor::util::BufferDataLoader(pte_buffer.data(), pte_buffer.size());
    if (!data_loader.ok()) {
         ET_LOG(Error, "Failed to create BufferDataLoader: 0x%x", data_loader.error());
         return 1;
    }

    // 4. Load the ExecuTorch Program
    torch::executor::Result<torch::executor::Program> program = torch::executor::Program::load(&data_loader);
     if (!program.ok()) {
        ET_LOG(Error, "Failed to load program: 0x%x", program.error());
        return 1;
    }
    ET_LOG(Info, "Program loaded successfully.");

    // 5. Get the Inference Method (usually named "forward")
    const char* method_name = "forward"; // Default name, verify if different
    torch::executor::Result<torch::executor::Method> method = program->load_method(method_name);
    if (!method.ok()) {
        ET_LOG(Error, "Failed to load method '%s': 0x%x", method_name, method.error());
        // Optional: Log available methods if loading fails
        ET_LOG(Info,"Available methods:");
        for (size_t i=0; i < program->num_methods(); ++i) {
             ET_LOG(Info,"  %s", program->get_method_name(i).get());
        }
        return 1;
    }
    ET_LOG(Info, "Method '%s' loaded successfully.", method_name);

    // 6. Prepare Input Tensor(s)
    // THIS IS HIGHLY MODEL-SPECIFIC! Match shape and dtype from your Python code.
    // Your Python code used input_data1 with shape [1, 1030, 38, 5]

    // Create a memory allocator and manager for inputs/outputs/intermediate tensors
    torch::executor::extension::MallocMemoryAllocator malloc_allocator; // Simple allocator
    // PlannedMemory needs a buffer for non-const tensors. Let's allocate a scratch space.
    size_t planned_buffer_size = 10 * 1024 * 1024; // 10 MB - Adjust as needed! Might need profiling.
    std::vector<uint8_t> planned_buffer(planned_buffer_size);
    torch::executor::MemoryAllocator* E = &malloc_allocator; // Alias for MemoryAllocator Interface
    torch::executor::MemoryManager memory_manager(E, nullptr); // Pass nullptr for planned buffer initially

    // Describe the input tensor shape and type
    torch::executor::SizesType input_sizes[] = {1, 1030, 38, 5}; // Matches input_data1 shape
    torch::executor::DimOrderType input_dim_order[] = {0, 1, 2, 3}; // Standard dim order
    torch::executor::ScalarType input_dtype = torch::executor::ScalarType::Float;
    size_t input_numel = 1 * 1030 * 38 * 5; // Calculate number of elements

    // Allocate memory for the input tensor data
    std::vector<float> input_data(input_numel);
    // *** IMPORTANT: Populate `input_data` with your actual data here! ***
    // This could involve reading from a file, sensor, etc., and preprocessing.
    // Example: Fill with dummy data like in Python
    for(size_t i = 0; i < input_numel; ++i) {
        input_data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Prepare the inputs for the method using EValue format
    // We need to manually create the TensorImpl and wrap it.
    // This is a lower-level detail often handled by helper functions in full examples.
    torch::executor::Result<std::vector<torch::executor::EValue>> inputs = method->inputs();
     if (!inputs.ok()) {
         ET_LOG(Error, "Failed to get method inputs: 0x%x", inputs.error());
         return 1;
     }
     if (inputs->size() != 1) { // Check if the method expects exactly one input
         ET_LOG(Error, "Model expects %zu inputs, but we prepared 1.", inputs->size());
         return 1;
     }

     // Create the input tensor directly
     // Note: In more complex scenarios, you'd use MemoryManager to allocate this
     torch::executor::TensorImpl input_tensor_impl(
         input_dtype,
         /*dim=*/4,
         input_sizes,
         input_data.data(), // Pointer to your input data
         input_dim_order);

     // Wrap the TensorImpl in a Tensor object
     torch::executor::Tensor input_tensor(&input_tensor_impl);

     // Assign it to the method's input EValue list
     // This step replaces the placeholder input tensor inside the method's
     // internal state with our actual input tensor.
     Error set_input_status = method->set_input(input_tensor, 0); // Set input at index 0
      if (set_input_status != Error::Ok) {
          ET_LOG(Error, "Failed to set method input: 0x%x", set_input_status);
          return 1;
      }

    ET_LOG(Info, "Input tensor prepared successfully.");
    printTensorDetails(input_tensor, "Input");


    // 7. Execute the Inference
    ET_LOG(Info, "Running inference...");
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::executor::Error status = method->execute(); // Run the actual computation
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (status != torch::executor::Error::Ok) {
        ET_LOG(Error, "Method execution failed: 0x%x", status);
        return 1;
    }
    ET_LOG(Info, "Inference completed successfully in %lld ms.", duration.count());

    // 8. Retrieve Output Tensor(s)
    // Your Python code showed one output tensor with shape [1, 6]
    if (method->outputs_size() < 1) {
         ET_LOG(Error, "Model returned no outputs!");
         return 1;
    }

    torch::executor::EValue output_evalue = method->get_output(0); // Get the first output
    if (!output_evalue.isTensor()) {
        ET_LOG(Error, "Expected output 0 to be a Tensor, but it's not.");
        return 1;
    }

    const torch::executor::Tensor& output_tensor = output_evalue.toTensor();
    printTensorDetails(output_tensor, "Output");

    // 9. Process Output Data
    // Ensure the output tensor has the expected type and dimensions
    if (output_tensor.scalar_type() == torch::executor::ScalarType::Float && output_tensor.dim() == 2 && output_tensor.size(0) == 1) {
        const float* output_ptr = output_tensor.const_data_ptr<float>();
        size_t num_outputs = output_tensor.numel(); // Should be 6 based on Python output

        std::cout << "Output values: ";
        for(size_t i = 0; i < num_outputs; ++i) {
            std::cout << output_ptr[i] << " ";
        }
        std::cout << std::endl;

        // Add your logic here: find max value, apply threshold, etc.

    } else {
        ET_LOG(Error, "Output tensor has unexpected shape or dtype.");
    }

    // 10. Cleanup (RAII helps, memory manager scope, etc.)
    // TensorImpl memory for inputs needs manual care if not using MemoryManager allocation.
    // Program, Method, etc. use RAII or need explicit cleanup depending on how they are managed.

    ET_LOG(Info, "Execution finished.");
    return 0;
}