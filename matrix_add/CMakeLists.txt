cmake_minimum_required(VERSION 3.18)
project(MatrixAdd LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Enable CUDA language support
enable_language(CUDA)

# Set CUDA architectures (adjust based on your GPU)
# RTX A6000 supports compute capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add executable for matrix addition with hierarchy comparison
add_executable(matrix_add matrix_add.cu)

# Set properties for CUDA executable
set_target_properties(matrix_add PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Link CUDA runtime
target_link_libraries(matrix_add ${CUDA_LIBRARIES})

# Optional: Add custom target to run the program
add_custom_target(run_matrix_add
    COMMAND ./matrix_add
    DEPENDS matrix_add
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running matrix addition hierarchy comparison"
)

# Print build information
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
