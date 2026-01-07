
import os
import shutil
import subprocess
import sys
import glob

# Constants
MODELS_DIR = "models"
CURRENT_MODEL_FILE = os.path.join(MODELS_DIR, "current_loaded_model.txt")
BUILD_DIR = "build"
ROOT_DIR = "."

# Files to manage (copy from model dir to root)
FILES_TO_COPY = [
    "network.c",
    "network.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c"
]
DIRS_TO_COPY = [
    "Lib",
    "Inc"
]

def list_models():
    """List available models in the models directory."""
    if not os.path.exists(MODELS_DIR):
        print(f"Error: '{MODELS_DIR}' directory not found.")
        return []

    models = [d for d in os.listdir(MODELS_DIR)
              if os.path.isdir(os.path.join(MODELS_DIR, d)) and not d.startswith(".")]
    models.sort()
    return models

def get_current_model():
    """Read the currently loaded model from file."""
    if os.path.exists(CURRENT_MODEL_FILE):
        with open(CURRENT_MODEL_FILE, 'r') as f:
            return f.read().strip()
    return None

def set_current_model(model_name):
    """Save the currently loaded model to file."""
    with open(CURRENT_MODEL_FILE, 'w') as f:
        f.write(model_name)

def clean_root_files():
    """Remove existing model files from the root directory."""
    print("Cleaning old model files...")
    for file in FILES_TO_COPY:
        if os.path.exists(file):
            os.remove(file)
            # print(f"Removed {file}")

    for directory in DIRS_TO_COPY:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            # print(f"Removed {directory}/")

def create_dummy_network_data_params():
    """Create a dummy network_data_params.c file if it's missing."""
    print("Creating dummy network_data_params.c...")
    content = """
/**
  * @file    network_data_params.c
  * @brief   Dummy file created by manage_models.py to satisfy Makefile compilation.
  *          The actual model likely doesn't use this file or encodes params differently.
  */
#include "network_data_params.h"
// No content needed for some models
"""
    # Look for header to see if we need to mock anything else?
    # Usually if the file is missing, the symbols are likely in network_data.c or unused.
    # However, 'network_data_params.h' might not exist either if params.c is missing.
    # Let's check headers.

    # If network_data_params.h doesn't exist, we might need to create a dummy header too?
    # But usually headers are copied. Let's write the dummy file.

    # UPDATE: We should check if network_data_params.h exists. If not, the include might fail.
    # But let's assume if the model doesn't have the .c file, it might not have the .h file either.
    # If so, we shouldn't include it.
    # But the Makefile expects the .c file to be compiled.

    with open("network_data_params.c", "w") as f:
        f.write("/** Dummy network_data_params.c */\n")
        # Empty file is valid C object

def load_model(model_name):
    """Load a specific model by copying files to root."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_name}' not found.")
        return False

    clean_root_files()

    print(f"Loading files from {model_name}...")

    # Copy files
    for file in FILES_TO_COPY:
        src = os.path.join(model_path, file)
        dst = os.path.join(ROOT_DIR, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {file}")
        elif file == "network_data_params.c":
            # Special handling for this optional file
            create_dummy_network_data_params()
        else:
            print(f"Warning: {file} not found in model directory.")

    # Copy directories
    for directory in DIRS_TO_COPY:
        src = os.path.join(model_path, directory)
        dst = os.path.join(ROOT_DIR, directory)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            print(f"Copied {directory}/")
        else:
            print(f"Warning: {directory}/ not found in model directory.")

    set_current_model(model_name)
    print(f"Model '{model_name}' loaded successfully.")
    return True

def is_int8_model(model_name):
    """Check if the model is INT8 quantized based on name."""
    lower_name = model_name.lower()
    return "int8" in lower_name or "quant" in lower_name

def build_model():
    """Build the currently loaded model."""
    model_name = get_current_model()
    if not model_name:
        print("Error: No model loaded. Please load a model first.")
        return

    print(f"Building model: {model_name}")

    # Clean build
    subprocess.run(["make", "clean"], check=True)

    # Build command
    cmd = ["make"]
    if is_int8_model(model_name):
        print("Detected INT8 model. Enabling INT_QUANTIZATION flag.")
        cmd.append("INT_QUANTIZATION=1")
    else:
        print("Detected Float32 model. Building standard version.")

    try:
        subprocess.run(cmd, check=True)
        print("Build successful!")
    except subprocess.CalledProcessError:
        print("Build failed.")

def flash_model():
    """Flash the built binary."""
    print("Flashing model...")
    try:
        subprocess.run(["./flash.sh"], check=True)
    except subprocess.CalledProcessError:
        print("Flash failed.")
    except FileNotFoundError:
        # Fallback to make flash if script is missing
        try:
             subprocess.run(["make", "flash"], check=True)
        except subprocess.CalledProcessError:
             print("Flash failed.")

def debug_model():
    """Build, Flash and Debug (not fully automated, just runs build/flash for now)."""
    # This matches user request "build and flash and run in debug mode"
    # Usually this implies running openocd or st-util, but standard practice via script
    # might just be flashing.
    # We will implement Build + Flash.
    build_model()
    flash_model()
    print("For debug output, please connect via UART or use: screen /dev/ttyACM0 115200")

def main():
    while True:
        current_model = get_current_model()
        models = list_models()

        print("\n--- STM32 Model Manager ---")
        print(f"Current Model: {current_model if current_model else 'None'}")
        print("1. List and Load Model")
        print("2. Build Current Model")
        print("3. Flash Current Model")
        print("4. Build, Flash & Run (Debug)")
        print("5. Quit")

        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            print("\nAvailable Models:")
            for i, m in enumerate(models):
                print(f"{i+1}. {m}")

            try:
                idx = int(input("Select model number: ")) - 1
                if 0 <= idx < len(models):
                    load_model(models[idx])
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")

        elif choice == '2':
            build_model()

        elif choice == '3':
            flash_model()

        elif choice == '4':
            debug_model()

        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
