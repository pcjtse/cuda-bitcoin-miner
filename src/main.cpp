#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include "../include/cuda_mining.h"
#include "../include/sha256.h"

// Utility function to print hash in hex
void print_hash(const uint8_t* hash) {
    for (int i = 0; i < 32; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    std::cout << std::dec << std::endl;
}

// Example mining work data
MiningWork create_example_work() {
    MiningWork work;
    work.version = 2;
    
    // Example previous block hash (all zeros for demonstration)
    memset(work.prev_block, 0, 32);
    
    // Example merkle root (all zeros for demonstration)
    memset(work.merkle_root, 0, 32);
    
    // Current timestamp
    work.timestamp = static_cast<uint32_t>(std::time(nullptr));
    
    // Example difficulty (make it easier for demonstration)
    work.bits = 0x1f00ffff;
    
    // Initial nonce
    work.nonce = 0;
    
    return work;
}

int main() {
    std::cout << "Initializing CUDA Bitcoin Miner..." << std::endl;
    
    // Initialize CUDA resources
    initialize_cuda_mining();
    
    // Create example mining work
    MiningWork work = create_example_work();
    
    std::cout << "Starting mining process..." << std::endl;
    std::cout << "Target difficulty: " << std::hex << work.bits << std::dec << std::endl;
    
    uint32_t start_nonce = 0;
    uint32_t iterations_per_kernel = 1000000;
    uint32_t nonce_result;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int attempts = 0;
    
    while (true) {
        attempts++;
        
        if (mine_block(&work, &nonce_result, start_nonce, iterations_per_kernel)) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            
            std::cout << "\nBlock found!" << std::endl;
            std::cout << "Nonce: " << nonce_result << std::endl;
            std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
            std::cout << "Attempts: " << attempts << std::endl;
            
            // Verify the result
            work.nonce = nonce_result;
            uint8_t header[80];
            uint8_t hash[32];
            
            // Reconstruct header
            *(uint32_t*)&header[0] = work.version;
            memcpy(&header[4], work.prev_block, 32);
            memcpy(&header[36], work.merkle_root, 32);
            *(uint32_t*)&header[68] = work.timestamp;
            *(uint32_t*)&header[72] = work.bits;
            *(uint32_t*)&header[76] = work.nonce;
            
            // Calculate hash
            double_sha256(header, 80, hash);
            
            std::cout << "Block hash: ";
            print_hash(hash);
            break;
        }
        
        start_nonce += iterations_per_kernel * THREADS_PER_BLOCK * NUM_BLOCKS;
        
        if (attempts % 10 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            double hash_rate = (double)(attempts * iterations_per_kernel * THREADS_PER_BLOCK * NUM_BLOCKS) / duration.count();
            std::cout << "\rHash rate: " << std::fixed << std::setprecision(2) << hash_rate / 1e6 << " MH/s" << std::flush;
        }
    }
    
    // Cleanup CUDA resources
    cleanup_cuda_mining();
    
    return 0;
} 