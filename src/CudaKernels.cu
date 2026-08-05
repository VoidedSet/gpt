#include <cuda_runtime.h>
#include "CudaKernels.hpp"

__global__ void embedding_forward(const int* tokens, const float* wte_, const float* wpe_, float* output, int B, int T, int C){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < (B*T*C))
    {
        int b = idx / (T * C),
            t = (idx / C) % T,
            c = idx % C;

        int token_id = tokens[(b * T) + t];
        
        output[idx] = wte_[token_id * C + c] + wpe_[t * C + c];
    }
}

__global__ void embedding_backward(const int* tokens, const float* dY, float* dwte, float* dwpe, int B, int T, int C){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < B * T * C){
        int b = idx / (T * C),
            t = (idx / C) % T,
            c = idx % C;

        int token_id = tokens[(b * T) + t];

        atomicAdd(&dwte[token_id * C + c], dY[idx]);
        atomicAdd(&dwpe[t * C + c], dY[idx]);

    }
}

void launch_embedding_forward(const int* tokens, const float* wte_, const float* wpe_, float* output, int B, int T, int C){
    int threadsPerBlock = 256, N = B * T * C;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
    
embedding_forward<<<blocksPerGrid, threadsPerBlock>>>(tokens, wte_, wpe_, output, B, T, C);
}

void launch_embedding_backward(const int* tokens, const float* dY, float* dwte, float* dwpe, int B, int T, int C){
    int threadsPerBlock = 256, N = B * T * C;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    embedding_backward<<<blocksPerGrid, threadsPerBlock>>>(tokens, dY, dwte, dwpe, B, T, C);
}