#include <iostream>
#include <cuda_runtime.h>

__global__ void multiply(const float* A, const float* B, float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] * B[idx] * alpha;
    }
}

int main() {

    int N = 9000;
    float h_alpha = 0.f;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size),
          *h_B = (float*)malloc(size),
          *h_C = (float*)malloc(size);

    for(size_t i = 0; i < N; ++i){
        h_A[i] = static_cast<float>(i * i);
        h_B[i] = static_cast<float>(i + i);
    }

    std::cin >> h_alpha;
    
    float* d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);   
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);   

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, h_alpha);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    for(size_t i = 0; i < N; ++i){
        float expected = h_A[i] * h_B[i] * h_alpha;
        if(expected != h_C[i])
            return -1;
    }

    std::cout << "[+] CUDA Boilerplate Vector Addition Success!\n";
    std::cout << "    Sample verification: A[10] + B[10] = " << h_A[10] << " * " << h_B[10] << " * " << h_alpha
                << " = " << h_C[10] << " Expected: " << h_A[10] * h_B[10] * h_alpha;

    return 0;
}