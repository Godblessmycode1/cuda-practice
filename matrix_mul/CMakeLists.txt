cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(matrix_mul_optimized LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Set default matrix size if not specified
if(NOT DEFINED MATRIX_SIZE)
    set(MATRIX_SIZE 1024)
endif()

# Configure matrix size
add_compile_definitions(MATRIX_SIZE=${MATRIX_SIZE})

# Set CUDA architecture (adjust based on your GPU)
# Option 1: Auto-detect (recommended for single GPU systems)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Option 2: If auto-detect fails, use common modern architectures
# Uncomment the line below and comment the auto-detect line above if needed
# set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")

# Common architectures for reference:
# - sm_60: Pascal (GTX 10 series)
# - sm_70: Volta (V100)
# - sm_75: Turing (RTX 20 series)
# - sm_80: Ampere (A100, RTX 30 series)
# - sm_86: Ampere (RTX 30 series mobile)
# - sm_89: Ada Lovelace (RTX 40 series)

# Set optimization flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Create the executable
add_executable(matrix_mul_optimized matrix_mul_optimized.cu)

# Set target properties
set_target_properties(matrix_mul_optimized PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(matrix_mul_optimized 
    CUDA::cudart
    CUDA::cublas
)

# Compiler-specific options
target_compile_options(matrix_mul_optimized PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v
        $<$<CONFIG:Debug>:-lineinfo>
    >
)

# Print configuration information
message(STATUS "CUDA Matrix Multiplication Configuration:")
message(STATUS "  Matrix Size: ${MATRIX_SIZE}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Toolkit Version: ${CUDAToolkit_VERSION}")

# Custom targets for different matrix sizes
add_custom_target(test-small
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=512 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMENT "Testing with small matrix (512x512)"
    VERBATIM
)

add_custom_target(test-medium
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=1024 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMENT "Testing with medium matrix (1024x1024)"
    VERBATIM
)

add_custom_target(test-large
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=2048 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMENT "Testing with large matrix (2048x2048)"
    VERBATIM
)

add_custom_target(test-xlarge
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=4096 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMENT "Testing with extra large matrix (4096x4096)"
    VERBATIM
)

# Benchmark target that runs all sizes
add_custom_target(benchmark
    COMMAND echo "Running comprehensive benchmark..."
    COMMAND echo "==================================="
    COMMAND echo ""
    COMMAND echo "Small Matrix (512x512):"
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=512 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMAND echo ""
    COMMAND echo "Medium Matrix (1024x1024):"
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=1024 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMAND echo ""
    COMMAND echo "Large Matrix (2048x2048):"
    COMMAND ${CMAKE_COMMAND} -DMATRIX_SIZE=2048 ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND $<TARGET_FILE:matrix_mul_optimized>
    COMMENT "Running benchmark with multiple matrix sizes"
    VERBATIM
)

# Profile target for NVIDIA profiling tools
add_custom_target(profile
    COMMAND nvprof --print-gpu-trace $<TARGET_FILE:matrix_mul_optimized>
    DEPENDS matrix_mul_optimized
    COMMENT "Profiling with nvprof"
    VERBATIM
)

# Advanced profiling with Nsight Compute (if available)
add_custom_target(profile-compute
    COMMAND ncu --set full $<TARGET_FILE:matrix_mul_optimized>
    DEPENDS matrix_mul_optimized
    COMMENT "Profiling with Nsight Compute"
    VERBATIM
)

# Help target
add_custom_target(help-targets
    COMMAND echo "Available targets:"
    COMMAND echo "  matrix_mul_optimized - Build the main executable"
    COMMAND echo "  test-small          - Test with 512x512 matrices"
    COMMAND echo "  test-medium         - Test with 1024x1024 matrices"
    COMMAND echo "  test-large          - Test with 2048x2048 matrices"
    COMMAND echo "  test-xlarge         - Test with 4096x4096 matrices"
    COMMAND echo "  benchmark           - Run comprehensive benchmark"
    COMMAND echo "  profile             - Profile with nvprof"
    COMMAND echo "  profile-compute     - Profile with Nsight Compute"
    COMMAND echo ""
    COMMAND echo "Build types:"
    COMMAND echo "  Release (default)   - Optimized build"
    COMMAND echo "  Debug               - Debug build with symbols"
    COMMAND echo ""
    COMMAND echo "Custom matrix size:"
    COMMAND echo "  cmake -DMATRIX_SIZE=<size> .."
    COMMENT "Display help information"
    VERBATIM
)
