# CUDA Bitcoin Miner

A simple Bitcoin mining implementation using CUDA for educational purposes. This miner demonstrates the basic concepts of Bitcoin mining and GPU acceleration using NVIDIA's CUDA platform.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 8.0 or higher)
- CMake (version 3.8 or higher)
- C++ compiler with C++14 support
- libcurl development package

### Installing Prerequisites on Windows

1. Install Visual Studio 2019 or later with C++ development tools
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Install [CMake](https://cmake.org/download/)
4. Install [vcpkg](https://github.com/Microsoft/vcpkg) package manager:
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install curl:x64-windows
```

### Installing Prerequisites on Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libcurl4-openssl-dev
```

### Installing Prerequisites on macOS

```bash
brew install cmake curl
```

## Building the Project

### On Windows

1. Create a build directory and navigate to it:
```bash
mkdir build
cd build
```

2. Generate the build files with CMake:
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake -G "Visual Studio 16 2019" -A x64
```

3. Build the project:
```bash
cmake --build . --config Release
```

The executable will be located in `build\Release\miner.exe`

### On Linux/macOS

1. Create a build directory and navigate to it:
```bash
mkdir build
cd build
```

2. Generate the build files with CMake:
```bash
cmake ..
```

3. Build the project:
```bash
make
```

## Running the Miner

After building, you can run the miner with:

**Windows:**
```bash
.\Release\miner.exe
```

**Linux/macOS:**
```bash
./miner
```

The miner will start searching for a valid block hash that meets the target difficulty. The program will display:
- Current hash rate in MH/s (Mega Hashes per second)
- When a valid block is found:
  - The winning nonce
  - Time taken
  - Number of attempts
  - The resulting block hash

## Implementation Details

This implementation includes:
- CUDA-accelerated SHA256 hashing
- Double SHA256 implementation as used in Bitcoin
- Basic block header structure
- Simplified difficulty targeting
- Multi-threaded GPU mining with configurable parameters

The miner is configured with:
- 256 threads per block
- 4096 blocks
- 1,000,000 iterations per kernel launch

## Note

This is an educational implementation and is not intended for actual Bitcoin mining. Real Bitcoin mining requires:
- Connection to the Bitcoin network
- Up-to-date block templates
- Much more optimized hashing implementations
- Proper difficulty adjustment
- Pool support

## Troubleshooting

### Windows-specific Issues

1. If you get CUDA-related errors, make sure:
   - Your NVIDIA drivers are up to date
   - CUDA Toolkit is properly installed
   - You're using the x64 build configuration

2. If you get curl-related errors:
   - Verify vcpkg is properly installed and integrated
   - Try rebuilding with the correct vcpkg toolchain file

### Common Issues

- If the build fails with missing CUDA architectures, update the `CUDA_ARCHITECTURES` in CMakeLists.txt to match your GPU
- For "compiler not found" errors, ensure you have the proper C++ development tools installed

## License

This project is released under the MIT License. See the LICENSE file for details. 