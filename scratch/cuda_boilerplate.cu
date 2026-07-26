#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: executed on the GPU by multiple threads in parallel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check boundaries to avoid writing outside array limits
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 256;
    size_t size = N * sizeof(float);

    // 1. Host (CPU) memory allocations
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize CPU data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 2. Device (GPU) memory allocations
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. Copy input data from CPU Host -> GPU Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. Configure and Launch the Kernel
    // We launch 1 block of 256 threads.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "[*] Launching vectorAdd kernel on GPU with " 
              << blocksPerGrid << " block(s) and " 
              << threadsPerBlock << " thread(s) per block...\n";
              
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch-time errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[-] CUDA Launch Error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // 5. Synchronize the CPU with the GPU (waits for GPU to complete)
    cudaDeviceSynchronize();

    // Check for runtime execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[-] CUDA Execution Error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // 6. Copy results back from GPU Device -> CPU Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 7. Verify correctness on CPU
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            std::cerr << "[-] Mismatch at index " << i << ": expected " << expected 
                      << ", got " << h_C[i] << "\n";
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "[+] CUDA Boilerplate Vector Addition Success!\n";
        std::cout << "    Sample verification: A[10] + B[10] = " << h_A[10] << " + " << h_B[10] 
                  << " = " << h_C[10] << " (Expected: 30)\n";
    }

    // 8. Clean up GPU VRAM allocations
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Clean up CPU Host RAM allocations
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
