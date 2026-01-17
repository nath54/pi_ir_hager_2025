#!/usr/bin/env python3
"""
STM32 Model Manager

Manages AI models exported from ST Edge AI Developer Cloud for STM32 microcontrollers.
Supports multiple devices and library backends.

Features:
- Load models from the models/ directory
- Automatic INT8/F32 detection
- Device selection (H723ZG, U545RE)
- Library selection (libopencm3, HAL - future)
- Build, flash, and debug workflows
"""

import os
import shutil
import subprocess
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = "models"
CURRENT_MODEL_FILE = os.path.join(MODELS_DIR, "current_loaded_model.txt")
CURRENT_DEVICE_FILE = os.path.join(MODELS_DIR, "current_device.txt")
BUILD_DIR = "build"
ROOT_DIR = "."

# Files to copy from model directory to root
FILES_TO_COPY = [
    "network.c",
    "network.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c"
]

# Directories to copy
DIRS_TO_COPY = [
    "Lib",
    "Inc"
]

# Supported devices
DEVICES = {
    "H723ZG": {
        "name": "NUCLEO-H723ZG",
        "core": "Cortex-M7",
        "max_mhz": 550,
        "clock_options": [64, 120, 240, 480, 550],
        "default_clock": 240,
    },
    "U545RE": {
        "name": "NUCLEO-U545RE-Q",
        "core": "Cortex-M33",
        "max_mhz": 160,
        "clock_options": [16, 80, 160],
        "default_clock": 160,
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def get_current_device():
    """Read the currently selected device from file."""
    if os.path.exists(CURRENT_DEVICE_FILE):
        with open(CURRENT_DEVICE_FILE, 'r') as f:
            return f.read().strip()
    return "H723ZG"  # Default


def set_current_device(device):
    """Save the currently selected device to file."""
    with open(CURRENT_DEVICE_FILE, 'w') as f:
        f.write(device)


def is_int8_model(model_name):
    """Check if the model is INT8 quantized based on name."""
    lower_name = model_name.lower()
    return "int8" in lower_name or "quant" in lower_name


def get_model_device(model_name):
    """Extract target device from model name if present."""
    upper_name = model_name.upper()
    for device in DEVICES.keys():
        if device in upper_name or DEVICES[device]["name"].upper().replace("-", "") in upper_name.replace("-", ""):
            return device
    return None


def clean_root_files():
    """Remove existing model files from the root directory."""
    print("Cleaning old model files...")
    for file in FILES_TO_COPY:
        if os.path.exists(file):
            os.remove(file)

    for directory in DIRS_TO_COPY:
        if os.path.exists(directory):
            shutil.rmtree(directory)


def create_dummy_network_data_params():
    """Create a dummy network_data_params.c file if it's missing."""
    print("Creating dummy network_data_params.c...")
    with open("network_data_params.c", "w") as f:
        f.write("/** Dummy network_data_params.c */\n")


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
            print(f"  Copied {file}")
        elif file == "network_data_params.c":
            create_dummy_network_data_params()
        else:
            print(f"  Warning: {file} not found in model directory.")

    # Copy directories
    for directory in DIRS_TO_COPY:
        src = os.path.join(model_path, directory)
        dst = os.path.join(ROOT_DIR, directory)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            print(f"  Copied {directory}/")
        else:
            print(f"  Warning: {directory}/ not found in model directory.")

    set_current_model(model_name)

    # Auto-detect device from model name
    detected_device = get_model_device(model_name)
    if detected_device:
        set_current_device(detected_device)
        print(f"  Auto-detected device: {DEVICES[detected_device]['name']}")

    print(f"\nModel '{model_name}' loaded successfully.")
    return True


# =============================================================================
# BUILD AND FLASH FUNCTIONS
# =============================================================================

def build_model(device=None, fast_mode=False, debug=False, sysclk=None, library="opencm3", safe_mode=False):
    """Build the currently loaded model."""
    model_name = get_current_model()
    if not model_name:
        print("Error: No model loaded. Please load a model first.")
        return False

    if device is None:
        device = get_current_device()

    device_info = DEVICES.get(device)
    if not device_info:
        print(f"Error: Unknown device '{device}'")
        return False

    if sysclk is None:
        sysclk = device_info["default_clock"]

    if sysclk not in device_info["clock_options"]:
        print(f"Warning: {sysclk}MHz not supported for {device}. Using {device_info['default_clock']}MHz.")
        sysclk = device_info["default_clock"]

    print(f"\n{'='*50}")
    print(f"Building: {model_name}")
    print(f"Device:   {device_info['name']} ({device_info['core']})")
    print(f"Clock:    {sysclk} MHz")
    print(f"Type:     {'INT8' if is_int8_model(model_name) else 'F32'}")
    print(f"Mode:     {'Debug' if debug else 'Release'}{' (FAST)' if fast_mode else ''}{' (SAFE)' if safe_mode else ''}")
    print(f"Library:  {'ST HAL' if library == 'hal' else 'libopencm3'}")
    print(f"{'='*50}\n")

    # Clean build
    subprocess.run(["make", "clean"], check=False)

    # Build command
    cmd = ["make", f"DEVICE={device}", f"SYSCLK={sysclk}"]

    if fast_mode:
        cmd.append("FAST_MODE=1")

    if is_int8_model(model_name):
        cmd.append("INT_QUANTIZATION=1")

    if debug:
        cmd.append("DEBUG=1")

    if library == "hal":
        cmd.append("LIB=hal")

    if safe_mode:
        cmd.append("SAFE_MODE=1")

    try:
        subprocess.run(cmd, check=True)
        print("\nBuild successful!")
        return True
    except subprocess.CalledProcessError:
        print("\nBuild failed.")
        return False


def flash_model():
    """Flash the built binary."""
    print("\nFlashing model...")
    try:
        if os.path.exists("./flash.sh"):
            result = subprocess.run(["./flash.sh"], capture_output=True, text=True)
        else:
            result = subprocess.run(["make", "flash"], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            if any(x in result.stderr for x in ["libusb", "ACCESS", "Permission denied"]):
                print("\n\033[91mError: Permission denied accessing USB device.\033[0m")
                print("Try running with sudo, or verify your udev rules for ST-Link.")
            else:
                print("Flash failed.")
            return False
        return True
    except Exception as e:
        print(f"An error occurred during flashing: {e}")
        return False


def run_debug():
    """Run with OpenOCD semihosting for debug output."""
    if os.path.exists("./run.sh"):
        subprocess.run(["./run.sh"])
    else:
        print("run.sh not found. Please run OpenOCD manually.")


# =============================================================================
# MENU FUNCTIONS
# =============================================================================

def select_model():
    """Interactive model selection."""
    models = list_models()
    if not models:
        print("No models found in models/ directory.")
        return

    current_device = get_current_device()

    print("\nAvailable Models:")
    print("-" * 60)
    for i, m in enumerate(models):
        model_type = "INT8" if is_int8_model(m) else "F32 "
        detected = get_model_device(m)
        device_str = f"[{detected}]" if detected else "[?]"
        print(f"  {i+1:2}. {device_str} {model_type} {m}")

    print("-" * 60)
    print(f"Current device: {DEVICES[current_device]['name']}")

    try:
        idx = int(input("\nSelect model number (0 to cancel): ")) - 1
        if idx == -1:
            return
        if 0 <= idx < len(models):
            load_model(models[idx])
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


def select_device():
    """Interactive device selection."""
    print("\nAvailable Devices:")
    print("-" * 40)
    for i, (key, info) in enumerate(DEVICES.items()):
        print(f"  {i+1}. {info['name']} ({info['core']}, max {info['max_mhz']}MHz)")

    current = get_current_device()
    print("-" * 40)
    print(f"Current: {DEVICES[current]['name']}")

    try:
        idx = int(input("\nSelect device (0 to cancel): ")) - 1
        if idx == -1:
            return
        device_keys = list(DEVICES.keys())
        if 0 <= idx < len(device_keys):
            set_current_device(device_keys[idx])
            print(f"Device set to {DEVICES[device_keys[idx]]['name']}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


def select_clock():
    """Interactive clock selection."""
    device = get_current_device()
    info = DEVICES[device]

    print(f"\nClock options for {info['name']}:")
    for mhz in info["clock_options"]:
        suffix = " (max)" if mhz == info["max_mhz"] else ""
        print(f"  - {mhz} MHz{suffix}")

    return input(f"\nEnter clock speed in MHz (default: {info['default_clock']}): ").strip()


def custom_build():
    """Interactive custom build with all options configurable."""
    model_name = get_current_model()
    if not model_name:
        print("Error: No model loaded. Please load a model first.")
        return

    device = get_current_device()
    device_info = DEVICES[device]

    print("\n" + "=" * 55)
    print("           CUSTOM BUILD CONFIGURATION")
    print("=" * 55)
    print(f"Model: {model_name}")
    print(f"Type:  {'INT8 (Quantized)' if is_int8_model(model_name) else 'F32 (Float)'}")
    print("=" * 55)

    # Option 1: Device
    print(f"\n[1] Device: {device_info['name']}")
    change = input("    Change device? (y/N): ").strip().lower()
    if change == 'y':
        select_device()
        device = get_current_device()
        device_info = DEVICES[device]

    # Option 2: Clock Speed
    print(f"\n[2] Clock Speed Options: {device_info['clock_options']} MHz")
    clock_input = input(f"    Enter clock (default={device_info['default_clock']}): ").strip()
    sysclk = int(clock_input) if clock_input else device_info['default_clock']
    if sysclk not in device_info['clock_options']:
        print(f"    Warning: {sysclk} not valid, using {device_info['default_clock']}")
        sysclk = device_info['default_clock']

    # Option 3: Debug Mode
    print("\n[3] Debug Mode: Enables UART output via semihosting")
    debug = input("    Enable debug? (y/N): ").strip().lower() == 'y'

    # Option 4: Fast Mode
    print("\n[4] Fast Mode: Removes all delays between inferences")
    fast_mode = input("    Enable fast mode? (y/N): ").strip().lower() == 'y'

    # Option 5: Safe Mode
    print("\n[5] Safe Mode: Skips PLL/Cache/MPU init (use if program won't run)")
    safe_mode = input("    Enable safe mode? (y/N): ").strip().lower() == 'y'

    # Option 6: Library Selection
    print("\n[6] Library: Which low-level library to use")
    print("    1. libopencm3 (open-source, default)")
    print("    2. ST HAL (proprietary, not yet fully implemented)")
    lib_choice = input("    Select (1 or 2, default=1): ").strip()
    library = "hal" if lib_choice == "2" else "opencm3"

    # Summary
    print("\n" + "-" * 55)
    print("BUILD CONFIGURATION SUMMARY:")
    print("-" * 55)
    print(f"  Device:       {device_info['name']}")
    print(f"  Clock:        {sysclk} MHz")
    print(f"  Model Type:   {'INT8' if is_int8_model(model_name) else 'F32'}")
    print(f"  Debug Mode:   {'YES' if debug else 'NO'}")
    print(f"  Fast Mode:    {'YES (no delays)' if fast_mode else 'NO'}")
    print(f"  Safe Mode:    {'YES (skip init)' if safe_mode else 'NO'}")
    print(f"  Library:      {'ST HAL' if library == 'hal' else 'libopencm3'}")
    print("-" * 55)

    confirm = input("\nProceed with build? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Build cancelled.")
        return

    # Build!
    build_model(device=device, fast_mode=fast_mode, debug=debug,
                sysclk=sysclk, safe_mode=safe_mode, library=library)


def main_menu():
    """Main interactive menu."""
    while True:
        current_model = get_current_model()
        current_device = get_current_device()
        device_info = DEVICES.get(current_device, DEVICES["H723ZG"])

        print("\n" + "=" * 50)
        print("        STM32 AI Model Manager")
        print("=" * 50)
        print(f"  Model:  {current_model if current_model else 'None'}")
        print(f"  Device: {device_info['name']}")
        print(f"  Type:   {'INT8' if current_model and is_int8_model(current_model) else 'F32'}")
        print("-" * 50)
        print("  1. Load Model")
        print("  2. Select Device")
        print("  3. Build (Release)")
        print("  4. Build (Debug + Semihosting)")
        print("  5. Build (FAST MODE)")
        print("  6. Build (SAFE MODE - Skip PLL/Cache/MPU)")
        print("  c. Build (CUSTOM - All Options)")
        print("  7. Flash")
        print("  8. Build + Flash + Run Debug")
        print("  9. Show Build Info")
        print("  0. Quit")
        print("-" * 50)

        choice = input("Enter choice: ").strip()

        if choice == '1':
            select_model()

        elif choice == '2':
            select_device()

        elif choice == '3':
            build_model()

        elif choice == '4':
            build_model(debug=True)

        elif choice == '5':
            # Fast mode: max clock, no sleep
            clock_input = select_clock()
            sysclk = int(clock_input) if clock_input else None
            build_model(fast_mode=True, sysclk=sysclk)

        elif choice == '6':
            # Safe mode: skip PLL/cache/MPU init
            print("\n[SAFE MODE] Skipping PLL, cache, and MPU initialization.")
            print("Runs at default HSI clock (~64MHz on H723ZG, ~16MHz on U545RE).")
            build_model(safe_mode=True)

        elif choice.lower() == 'c':
            custom_build()

        elif choice == '7':
            flash_model()

        elif choice == '8':
            if build_model(debug=True):
                if flash_model():
                    print("\nStarting debug session...")
                    run_debug()

        elif choice == '9':
            # Show current build info based on loaded model
            model_name = current_model
            int8_flag = "INT_QUANTIZATION=1" if model_name and is_int8_model(model_name) else ""
            cmd = ["make", "info", f"DEVICE={current_device}"]
            if int8_flag:
                cmd.append(int8_flag)
            subprocess.run(cmd)

        elif choice == '0':
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main_menu()
