#pragma once

#include <cstdint>

// SHA256 block size and hash size
#define SHA256_BLOCK_SIZE 64
#define SHA256_HASH_SIZE 32

// SHA256 context structure
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[SHA256_BLOCK_SIZE];
} SHA256_CTX;

// SHA256 function declarations
void sha256_init(SHA256_CTX* ctx);
void sha256_update(SHA256_CTX* ctx, const uint8_t* data, size_t len);
void sha256_final(SHA256_CTX* ctx, uint8_t* hash);
void sha256_transform(uint32_t* state, const uint8_t* block);

// Double SHA256 helper function
void double_sha256(const uint8_t* input, size_t length, uint8_t* output);

// CUDA device functions for SHA256
__device__ void cuda_sha256_transform(uint32_t* state, const uint8_t* block);
__device__ void cuda_double_sha256(const uint8_t* input, size_t length, uint8_t* output); 