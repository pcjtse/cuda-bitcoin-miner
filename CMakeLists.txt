cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_bitcoin_miner CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find CURL package for HTTP requests
find_package(CURL REQUIRED)

# Add executable
add_executable(miner
    src/main.cpp
    src/cuda_mining.cu
    src/sha256.cpp
)

# Set CUDA architectures
set_target_properties(miner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)

# Include directories
target_include_directories(miner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(miner PRIVATE
    ${CURL_LIBRARIES}
) 