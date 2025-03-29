#pragma once

#include <cstdint>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Structure to hold mining work data
struct MiningWork {
    uint32_t version;
    uint8_t prev_block[32];
    uint8_t merkle_root[32];
    uint32_t timestamp;
    uint32_t bits;
    uint32_t nonce;
};

// CUDA kernel function declaration
__global__ void bitcoin_mining_kernel(
    const MiningWork* work,
    uint32_t* nonce_result,
    bool* found,
    uint32_t start_nonce,
    uint32_t iterations
);

// Host function declarations
void initialize_cuda_mining();
void cleanup_cuda_mining();
bool mine_block(
    const MiningWork* work,
    uint32_t* nonce_result,
    uint32_t start_nonce,
    uint32_t iterations_per_kernel
); 