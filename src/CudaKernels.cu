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

__global__ void layernorm_forward(float* X, const float* gamma, 
                          const float* beta, float* x_hat, 
                          float* mean, float* var, int B,
                          int T, int C, float eps){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < B * T){
        // int b = idx / (T * C),
        //     t = (idx / C) % T,
        //     // c = idx % C,
        int row_offset = idx * C;

        float sum = 0.00f;
        for(size_t c = 0; c < C; ++c)
            sum += X[row_offset + c];
        
        mean[idx] = sum / C;

        float sum_sq_diff = 0.00f;
        for(size_t c = 0; c < C; ++c){
            float diff = X[row_offset + c] - mean[idx];
            sum_sq_diff += diff * diff;
        }

        var[idx] = sum_sq_diff / C;
        float scale = 1.00f / sqrtf(var[idx] + eps); 

        for(size_t c = 0; c < C; ++c){
            float xh = (X[row_offset + c] - mean[idx]) * scale;
            x_hat[row_offset + c] = xh;
            X[row_offset + c] = xh * gamma[c] + beta[c];
        }
        
    }

}

__global__ void layernorm_backward(const float* dY, const float* x_hat, const float* var, const float* gamma,
                                float* dgamma, float* dbeta, float* dX, int B, int T, int C, float eps){
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < B * T){
        size_t row_offset = idx * C;
        float scale = 1.00f / sqrtf(var[idx] + eps);

        float sum_dy_gamma = 0.00f, sum_dy_gamma_xhat = 0.00f;

        for(size_t c = 0; c < C; ++c){
            float dy = dY[row_offset + c],  
                  xh = x_hat[row_offset + c];
            
            atomicAdd(&dgamma[c], dy * xh);
            atomicAdd(&dbeta[c], dy); 

            sum_dy_gamma += dy * gamma[c];
            sum_dy_gamma_xhat += dy * gamma[c] * xh;
        }

        for(size_t c = 0; c < C; ++c){
            float dy = dY[row_offset + c],
                  xh = x_hat[row_offset + c];
                
            dX[row_offset + c] = (gamma[c] * dy - (sum_dy_gamma + xh * sum_dy_gamma_xhat) / C) * scale;
        }
    }
}

// cpp wrappers for cuda kernels

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

void launch_layernorm_forward(float* X, const float* gamma, 
                          const float* beta, float* x_hat, 
                          float* mean, float* var, int B,
                          int T, int C, float eps){
    int threadsPerBlock = 256, N = B * T;

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    layernorm_forward<<<blocksPerGrid, threadsPerBlock>>>(X, gamma, beta, x_hat, mean, var, B, T, C, eps);
}


void launch_layernorm_backward(const float* dY, const float* x_hat, const float* var, const float* gamma,
                                float* dgamma, float* dbeta, float* dX, int B, int T, int C, float eps){

    int threadsPerBlock = 256, N = B * T;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    layernorm_backward<<<blocksPerGrid, threadsPerBlock>>>(dY, x_hat, var, gamma, dgamma, dbeta, dX, B, T, C, eps);
}