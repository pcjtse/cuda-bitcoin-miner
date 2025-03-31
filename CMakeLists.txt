cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_bitcoin_miner CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows-specific settings
if(WIN32)
    # Disable CRT secure warnings on Windows
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    
    # Enable parallel build on MSVC
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    endif()
endif()

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

# Windows-specific post-build steps
if(WIN32)
    # Copy DLL dependencies to output directory
    add_custom_command(TARGET miner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:miner>
        $<TARGET_FILE_DIR:miner>
        COMMAND_EXPAND_LISTS
    )
endif() 