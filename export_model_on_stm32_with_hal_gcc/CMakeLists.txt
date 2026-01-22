cmake_minimum_required(VERSION 3.22)

#
# STM32 HAL AI Project CMakeLists.txt
# Supports dynamic main source selection and AI model integration.
#

# Setup ARM toolchain BEFORE project()
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gcc-arm-none-eabi.cmake)

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Configurable main source (passed from manage_models.py)
if(NOT DEFINED MAIN_SOURCE)
    set(MAIN_SOURCE "main_ai.c")
endif()

# Configurable quantization (for compile definitions)
if(NOT DEFINED QUANTIZATION)
    set(QUANTIZATION "F32")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME stm32_hal_ai)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME} C ASM)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Main source: ${MAIN_SOURCE}")
message(STATUS "Quantization: ${QUANTIZATION}")

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# ============================================================================
# AI Model Sources
# ============================================================================

# Check if AI directory exists with network files
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/AI/network.c")
    message(STATUS "AI model found - including AI sources")

    # AI Network sources
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/AI/network.c
        ${CMAKE_CURRENT_SOURCE_DIR}/AI/network_data.c
    )

    # AI includes
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/AI
        ${CMAKE_CURRENT_SOURCE_DIR}/AI/Inc
    )

    # AI runtime library (search in Lib directory)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/AI/Lib")
        # Find the .a library file (may vary by name)
        file(GLOB AI_LIB_FILES "${CMAKE_CURRENT_SOURCE_DIR}/AI/Lib/*.a")
        if(AI_LIB_FILES)
            # Link using the full path directly
            # foreach(LIB_PATH ${AI_LIB_FILES})
            #     message(STATUS "AI library found: ${LIB_PATH}")
            #     target_link_libraries(${CMAKE_PROJECT_NAME} ${LIB_PATH})
            # endforeach()
            set(AI_LIBRARIES "")
            foreach(LIB_PATH ${AI_LIB_FILES})
                message(STATUS "AI library found: ${LIB_PATH}")
                list(APPEND AI_LIBRARIES ${LIB_PATH})
            endforeach()
        else()
            message(WARNING "AI Lib/ directory exists but no .a files found!")
        endif()
    else()
        message(WARNING "AI/Lib directory not found - build may fail!")
    endif()

    # Quantization define
    if(QUANTIZATION STREQUAL "INT8")
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE AI_QUANTIZATION_INT8)
    else()
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE AI_QUANTIZATION_F32)
    endif()
else()
    message(STATUS "No AI model loaded - AI features disabled")
endif()

# ============================================================================
# Additional Application Sources
# ============================================================================

# Add custom application sources
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/ai_led.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/ai_network.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/ai_signal.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/system_init.c
)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
# target_link_libraries(${CMAKE_PROJECT_NAME}
#     stm32cubemx
#     # Add user defined libraries
# )
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx
    ${AI_LIBRARIES}
    m
)

# ============================================================================
# Post-Build: Generate .bin and .hex files
# ============================================================================

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${CMAKE_PROJECT_NAME}> ${CMAKE_PROJECT_NAME}.hex
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${CMAKE_PROJECT_NAME}> ${CMAKE_PROJECT_NAME}.bin
    COMMENT "Generating ${CMAKE_PROJECT_NAME}.hex and ${CMAKE_PROJECT_NAME}.bin"
)
