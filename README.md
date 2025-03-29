# CUDA Bitcoin Miner

A simple Bitcoin mining implementation using CUDA for educational purposes. This miner demonstrates the basic concepts of Bitcoin mining and GPU acceleration using NVIDIA's CUDA platform.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 8.0 or higher)
- CMake (version 3.8 or higher)
- C++ compiler with C++14 support
- libcurl development package

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

## License

This project is released under the MIT License. See the LICENSE file for details. 