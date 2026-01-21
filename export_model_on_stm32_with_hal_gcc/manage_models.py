#!/usr/bin/env python3
"""
STM32 HAL AI Model Manager

Manages AI models for the STM32U545RE-Q HAL project.
Simpler version focused on single device with HAL library.

Features:
- Load models from ../stm32_ai_models/
- Auto-detect INT8/F32 quantization
- Update project_config.yaml
- Build with CMake
- Flash to board
"""

import os
import shutil
import subprocess
import sys

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Install with: pip install pyyaml")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = "../stm32_ai_models"
CONFIG_FILE = "project_config.yaml"
AI_DIR = "AI"
BUILD_DIR = "build"

# Files to copy from model directory
FILES_TO_COPY = [
    "network.c",
    "network.h",
    "network_data.c",
    "network_data.h",
    "network_details.h",
]

# Directories to copy
DIRS_TO_COPY = [
    "Lib",
    "Inc"
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_config():
    """Load project configuration from YAML file."""
    if not YAML_AVAILABLE:
        return get_default_config()
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    return get_default_config()


def save_config(config):
    """Save project configuration to YAML file."""
    if not YAML_AVAILABLE:
        print("Warning: Cannot save config - PyYAML not installed")
        return
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config():
    """Return default configuration."""
    return {
        'device': {
            'name': 'NUCLEO-U545RE-Q',
            'cpu': 'cortex-m33',
            'fpu': 'fpv5-sp-d16',
            'float_abi': 'hard'
        },
        'build': {
            'main_source': 'main_ai.c',
            'quantization': 'F32',
            'build_type': 'Release'
        },
        'paths': {
            'models_dir': '../stm32_ai_models',
            'ai_dir': 'AI'
        },
        'current_model': None
    }


def list_models():
    """List available models in the models directory."""
    if not os.path.exists(MODELS_DIR):
        print(f"Error: '{MODELS_DIR}' directory not found.")
        return []
    
    # Only list models compatible with U545RE-Q
    models = []
    for d in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, d)
        if os.path.isdir(path) and not d.startswith("."):
            # Filter for U545RE compatible models
            if "U545" in d.upper() or "NUCLEO-U545RE" in d.upper():
                models.append(d)
    
    models.sort()
    return models


def is_int8_model(model_name):
    """Check if the model is INT8 quantized based on name."""
    lower_name = model_name.lower()
    return "int8" in lower_name or "quant" in lower_name


def clean_ai_dir():
    """Remove existing AI files."""
    if os.path.exists(AI_DIR):
        shutil.rmtree(AI_DIR)
    os.makedirs(AI_DIR, exist_ok=True)


def load_model(model_name):
    """Load a specific model by copying files to AI directory."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_name}' not found.")
        return False
    
    clean_ai_dir()
    
    print(f"\nLoading model: {model_name}")
    print("-" * 50)
    
    # Copy files
    for file in FILES_TO_COPY:
        src = os.path.join(model_path, file)
        dst = os.path.join(AI_DIR, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ⚠ Warning: {file} not found")
    
    # Copy directories
    lib_found = False
    for directory in DIRS_TO_COPY:
        src = os.path.join(model_path, directory)
        dst = os.path.join(AI_DIR, directory)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            print(f"  ✓ Copied {directory}/")
            if directory == "Lib":
                lib_found = True
        else:
            if directory == "Lib":
                print(f"  ✗ ERROR: {directory}/ not found - build will fail!")
            else:
                print(f"  ⚠ Warning: {directory}/ not found")
    
    # Update config
    config = load_config()
    config['current_model'] = model_name
    config['build']['quantization'] = 'INT8' if is_int8_model(model_name) else 'F32'
    save_config(config)
    
    print("-" * 50)
    print(f"Model loaded: {model_name}")
    print(f"Quantization: {'INT8' if is_int8_model(model_name) else 'F32'}")
    
    if not lib_found:
        print("\n⚠ WARNING: Lib/ directory missing. Copy from another model.")
    
    return True


# =============================================================================
# BUILD FUNCTIONS
# =============================================================================

def clean_all_builds():
    """Remove the entire build directory."""
    if os.path.exists(BUILD_DIR):
        print(f"Cleaning entire build directory: {BUILD_DIR}...")
        shutil.rmtree(BUILD_DIR)
        print("✓ Done. All build artifacts removed.")
    else:
        print("Build directory already clean.")


def build_project(main_source=None, quantization=None, build_type=None, clean=False):
    """Build the project using CMake."""
    config = load_config()
    
    # Use provided values or defaults from config
    if main_source is None:
        main_source = config['build']['main_source']
    if quantization is None:
        quantization = config['build']['quantization']
    if build_type is None:
        build_type = config['build']['build_type']
    
    # Update config with current choices so Flash picks it up
    config['build']['main_source'] = main_source
    config['build']['quantization'] = quantization
    config['build']['build_type'] = build_type
    save_config(config)

    build_path = os.path.join(BUILD_DIR, build_type)
    
    print(f"\n{'='*50}")
    print(f"Building Project")
    print(f"{'='*50}")
    print(f"  Main source:  {main_source}")
    print(f"  Quantization: {quantization}")
    print(f"  Build type:   {build_type}")
    print(f"  Build path:   {build_path}")
    print(f"{'='*50}\n")
    
    # Clean if requested (but preferably use clean_all_builds from menu)
    if clean and os.path.exists(build_path):
        print("Cleaning build directory...")
        shutil.rmtree(build_path)
    
    # Create build directory
    os.makedirs(build_path, exist_ok=True)
    
    # CMake configure
    cmake_cmd = [
        "cmake",
        "-S", ".",
        "-B", build_path,
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DMAIN_SOURCE={main_source}",
        f"-DQUANTIZATION={quantization}",
    ]
    
    print("Running CMake configure...")
    result = subprocess.run(cmake_cmd, capture_output=False)
    if result.returncode != 0:
        print("CMake configure failed!")
        return False
    
    # CMake build
    print("\nRunning CMake build...")
    build_cmd = ["cmake", "--build", build_path, "--parallel"]
    result = subprocess.run(build_cmd, capture_output=False)
    if result.returncode != 0:
        print("Build failed!")
        return False
    
    # Verify binary exists, or create it if needed
    bin_file = os.path.join(build_path, "stm32_hal_ai.bin")
    elf_file = os.path.join(build_path, "stm32_hal_ai.elf")
    
    if not os.path.exists(bin_file) and os.path.exists(elf_file):
        print("\nInternal bin file missing, creating from ELF...")
        # Try to make bin
        try:
             subprocess.run(["arm-none-eabi-objcopy", "-O", "binary", elf_file, bin_file], check=True)
             print("Created binary file.")
        except Exception as e:
             print(f"Warning: Could not create binary file: {e}")

    print("\n✓ Build successful!")
    return True


def flash_project(build_type=None):
    """Flash the built binary to the board with multiple fallbacks."""
    config = load_config()
    if build_type is None:
        build_type = config['build']['build_type']
    
    build_path = os.path.join(BUILD_DIR, build_type)
    bin_file = os.path.join(build_path, "stm32_hal_ai.bin")
    elf_file = os.path.join(build_path, "stm32_hal_ai.elf")
    
    if not os.path.exists(bin_file):
        print(f"Error: Binary not found: {bin_file}")
        if os.path.exists(elf_file):
            print("ELF exists but BIN missing. Re-build or check objcopy.")
        else:
            print("Run build first!")
        return False
    
    print(f"\nFlashing: {bin_file}")
    
    # 1. Try st-flash (with reset)
    print("Attempting: st-flash --connect-under-reset")
    cmd = ["st-flash", "--connect-under-reset", "write", bin_file, "0x08000000"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Flash successful (st-flash w/ reset)!")
        return True
    else:
        print(f"  Failed: {result.stderr.strip()}")

    # 2. Try st-flash (standard)
    print("Attempting: st-flash (standard)")
    cmd = ["st-flash", "write", bin_file, "0x08000000"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Flash successful (st-flash)!")
        return True
    else:
        print(f"  Failed: {result.stderr.strip()}")
    
    # 3. Try OpenOCD (since we found it on the system and flash.sh uses it)
    print("Attempting: OpenOCD")
    openocd_cmd = [
        "openocd",
        "-f", "interface/stlink.cfg",
        "-f", "target/stm32u5x.cfg",
        "-c", "init",
        "-c", "reset halt",
        "-c", f"flash write_image erase {elf_file}",  # OpenOCD likes ELF usually
        "-c", "verify_image {elf_file}",
        "-c", "reset run",
        "-c", "exit"
    ]
    # If elf file is missing, try bin with offset
    if not os.path.exists(elf_file):
         openocd_cmd[8] = f"flash write_image erase {bin_file} 0x08000000"
         openocd_cmd[10] = f"verify_image {bin_file} 0x08000000"

    try:
        result = subprocess.run(openocd_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Flash successful (OpenOCD)!")
            return True
        else:
            print(f"  Failed. Output snippet: {result.stderr[-200:] if result.stderr else 'No output'}")
    except FileNotFoundError:
        print("  OpenOCD not found.")

    # 4. Try STM32_Programmer_CLI as last resort
    print("Attempting: STM32_Programmer_CLI")
    try:
        result = subprocess.run(
            ["STM32_Programmer_CLI", "-c", "port=SWD", "-w", bin_file, "0x08000000", "-v", "-rst"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✓ Flash successful (STM32_Programmer_CLI)!")
            return True
        else:
            print(f"  Failed: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  STM32_Programmer_CLI not found in PATH.")
    
    print("\n❌ All flash methods failed.")
    print("Suggestions:")
    print("  - Check USB connection")
    print("  - Try holding the RESET button while starting flash")
    print("  - Disconnect and reconnect the board")
    return False


# =============================================================================
# MENU FUNCTIONS
# =============================================================================

def select_model():
    """Interactive model selection."""
    models = list_models()
    if not models:
        print("\nNo compatible models found for U545RE-Q!")
        print(f"Looking in: {os.path.abspath(MODELS_DIR)}")
        return
    
    print("\nAvailable Models (U545RE-Q compatible):")
    print("-" * 60)
    for i, m in enumerate(models):
        model_type = "INT8" if is_int8_model(m) else "F32 "
        print(f"  {i+1:2}. [{model_type}] {m}")
    print("-" * 60)
    
    try:
        idx = int(input("\nSelect model (0 to cancel): ")) - 1
        if idx == -1:
            return
        if 0 <= idx < len(models):
            load_model(models[idx])
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


def select_main_source():
    """Select main source file."""
    print("\nMain Source Options:")
    print("  1. main_ai.c   - AI inference (requires loaded model)")
    print("  2. main_demo.c - Simple demo (LED blink, no AI)")
    
    choice = input("\nSelect (1-2): ").strip()
    
    config = load_config()
    if choice == "1":
        config['build']['main_source'] = "main_ai.c"
        print("Selected: main_ai.c")
    elif choice == "2":
        config['build']['main_source'] = "main_demo.c"
        print("Selected: main_demo.c")
    else:
        print("Invalid choice.")
        return
    
    save_config(config)


def show_status():
    """Show current project status."""
    config = load_config()
    
    print("\n" + "=" * 50)
    print("         STM32 HAL AI Project Status")
    print("=" * 50)
    print(f"  Device:       {config['device']['name']}")
    print(f"  Current Model: {config.get('current_model', 'None')}")
    print(f"  Main Source:  {config['build']['main_source']}")
    print(f"  Quantization: {config['build']['quantization']}")
    print(f"  Build Type:   {config['build']['build_type']}")
    print("-" * 50)
    
    # Check if AI files exist
    if os.path.exists(os.path.join(AI_DIR, "network.c")):
        print("  AI files:     ✓ Present")
    else:
        print("  AI files:     ✗ Not loaded (run 'Load Model')")
    
    print("=" * 50)


def main_menu():
    """Main interactive menu."""
    while True:
        config = load_config()
        current_model = config.get('current_model', None)
        main_source = config['build']['main_source']
        
        print("\n" + "=" * 50)
        print("        STM32 HAL AI Model Manager")
        print("=" * 50)
        print(f"  Model:  {current_model if current_model else 'None'}")
        print(f"  Main:   {main_source}")
        print(f"  Type:   {'INT8' if current_model and is_int8_model(current_model) else 'F32'}")
        print("-" * 50)
        print("  1. Load Model")
        print("  2. Select Main Source (AI/Demo)")
        print("  3. Build (Release)")
        print("  4. Build (Debug)")
        print("  5. Build + Flash")
        print("  6. Flash Only")
        print("  7. Clean All Builds")
        print("  8. Show Status")
        print("  0. Quit")
        print("-" * 50)
        
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            select_model()
        
        elif choice == '2':
            select_main_source()
        
        elif choice == '3':
            build_project(build_type="Release")
        
        elif choice == '4':
            build_project(build_type="Debug")
        
        elif choice == '5':
            if build_project():
                flash_project()
        
        elif choice == '6':
            flash_project()
        
        elif choice == '7':
            clean_all_builds()
        
        elif choice == '8':
            show_status()
        
        elif choice == '0':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main_menu()
