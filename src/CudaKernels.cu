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


void launch_embedding_forward(const int* tokens, const float* wte_, const float* wpe_, float* output, int B, int T, int C){
    int threadsPerBlock = 256, N = B * T * C;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
    
embedding_forward<<<blocksPerGrid, threadsPerBlock>>>(tokens, wte_, wpe_, output, B, T, C);
}