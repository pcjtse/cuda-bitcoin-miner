#include <cuda_runtime.h>
#include "../include/cuda_mining.h"
#include "../include/sha256.h"

// Constants for the mining kernel
const int THREADS_PER_BLOCK = 256;
const int NUM_BLOCKS = 4096;

// Device pointers
static MiningWork* d_work = nullptr;
static uint32_t* d_nonce_result = nullptr;
static bool* d_found = nullptr;

__device__ __constant__ uint32_t dev_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ void cuda_sha256_transform(uint32_t* state, const uint8_t* block) {
    uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
    int i, j;

    // Prepare message schedule
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (block[j] << 24) | (block[j + 1] << 16) | (block[j + 2] << 8) | (block[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + dev_k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void cuda_double_sha256(const uint8_t* input, size_t length, uint8_t* output) {
    uint32_t state[8];
    uint8_t temp_hash[32];
    
    // First SHA256
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;

    cuda_sha256_transform(state, input);
    
    for (int i = 0; i < 8; i++) {
        temp_hash[i * 4] = (state[i] >> 24) & 0xff;
        temp_hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
        temp_hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
        temp_hash[i * 4 + 3] = state[i] & 0xff;
    }
    
    // Second SHA256
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;

    cuda_sha256_transform(state, temp_hash);
    
    for (int i = 0; i < 8; i++) {
        output[i * 4] = (state[i] >> 24) & 0xff;
        output[i * 4 + 1] = (state[i] >> 16) & 0xff;
        output[i * 4 + 2] = (state[i] >> 8) & 0xff;
        output[i * 4 + 3] = state[i] & 0xff;
    }
}

__global__ void bitcoin_mining_kernel(
    const MiningWork* work,
    uint32_t* nonce_result,
    bool* found,
    uint32_t start_nonce,
    uint32_t iterations
) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    uint32_t nonce = start_nonce + thread_id;
    
    uint8_t header[80];
    uint8_t hash[32];
    
    // Copy the header data
    *(uint32_t*)&header[0] = work->version;
    memcpy(&header[4], work->prev_block, 32);
    memcpy(&header[36], work->merkle_root, 32);
    *(uint32_t*)&header[68] = work->timestamp;
    *(uint32_t*)&header[72] = work->bits;
    
    for (uint32_t i = 0; i < iterations; i++) {
        if (*found) return;
        
        *(uint32_t*)&header[76] = nonce;
        
        cuda_double_sha256(header, 80, hash);
        
        // Check if hash is below target
        bool is_below_target = true;
        for (int j = 31; j >= 0; j--) {
            if (hash[j] > 0) {
                is_below_target = false;
                break;
            }
        }
        
        if (is_below_target) {
            *nonce_result = nonce;
            *found = true;
            return;
        }
        
        nonce += stride;
    }
}

void initialize_cuda_mining() {
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(MiningWork)));
    CUDA_CHECK(cudaMalloc(&d_nonce_result, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(bool)));
}

void cleanup_cuda_mining() {
    if (d_work) cudaFree(d_work);
    if (d_nonce_result) cudaFree(d_nonce_result);
    if (d_found) cudaFree(d_found);
    
    d_work = nullptr;
    d_nonce_result = nullptr;
    d_found = nullptr;
}

bool mine_block(
    const MiningWork* work,
    uint32_t* nonce_result,
    uint32_t start_nonce,
    uint32_t iterations_per_kernel
) {
    // Copy work to device
    CUDA_CHECK(cudaMemcpy(d_work, work, sizeof(MiningWork), cudaMemcpyHostToDevice));
    
    // Reset found flag
    bool found = false;
    CUDA_CHECK(cudaMemcpy(d_found, &found, sizeof(bool), cudaMemcpyHostToDevice));
    
    // Launch kernel
    bitcoin_mining_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        d_work,
        d_nonce_result,
        d_found,
        start_nonce,
        iterations_per_kernel
    );
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check if we found a valid nonce
    CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));
    
    if (found) {
        CUDA_CHECK(cudaMemcpy(nonce_result, d_nonce_result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return true;
    }
    
    return false;
} 