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

__global__ void gelu_forward_kernel(float* X, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = X[idx];
        X[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void gelu_backward_kernel(const float* h1, const float* dh2, float* dh1, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = h1[idx];
        float x3 = x * x * x;
        float a = 0.79788456f * (x + 0.044715f * x3);
        float t = tanhf(a);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.79788456f * (1.0f + 0.134145f * x * x);
        dh1[idx] = dh2[idx] * dgelu;
    }
}

void launch_gelu_forward(float* X, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gelu_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, N);
}

void launch_gelu_backward(const float* h1, const float* dh2, float* dh1, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gelu_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(h1, dh2, dh1, N);
}

__global__ void add_bias_kernel(float* X, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        X[idx] += bias[col];
    }
}

__global__ void accumulate_bias_grad_kernel(const float* dY, float* dbias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; ++r) {
            sum += dY[r * cols + col];
        }
        atomicAdd(&dbias[col], sum);
    }
}

void launch_add_bias(float* X, const float* bias, int rows, int cols) {
    int threadsPerBlock = 256;
    int N = rows * cols;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_bias_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, bias, rows, cols);
}

void launch_accumulate_bias_grad(const float* dY, float* dbias, int rows, int cols) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    accumulate_bias_grad_kernel<<<blocksPerGrid, threadsPerBlock>>>(dY, dbias, rows, cols);
}

__global__ void attention_forward_kernel(const float* qkv, float* att_probs, float* O, 
                                         int B, int T, int C, int num_heads, int head_dim, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = B * num_heads * T;
    if (idx < total_queries) {
        int b = idx / (num_heads * T);
        int head = (idx / T) % num_heads;
        int t_q = idx % T;

        float scores[1024];

        // 1. Compute dot products (Query x Key)
        for (int t_k = 0; t_k <= t_q; ++t_k) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float q = qkv[b * T * 3 * C + t_q * 3 * C + head * head_dim + d];
                float k = qkv[b * T * 3 * C + t_k * 3 * C + C + head * head_dim + d];
                dot += q * k;
            }
            scores[t_k] = dot * scale;
        }

        // 2. Softmax
        float max_val = scores[0];
        for (int t_k = 1; t_k <= t_q; ++t_k) {
            if (scores[t_k] > max_val) {
                max_val = scores[t_k];
            }
        }

        float sum_exp = 0.0f;
        for (int t_k = 0; t_k <= t_q; ++t_k) {
            scores[t_k] = expf(scores[t_k] - max_val);
            sum_exp += scores[t_k];
        }

        int att_offset = b * num_heads * T * T + head * T * T + t_q * T;
        for (int t_k = 0; t_k <= t_q; ++t_k) {
            float p = scores[t_k] / sum_exp;
            scores[t_k] = p;
            att_probs[att_offset + t_k] = p;
        }
        for (int t_k = t_q + 1; t_k < T; ++t_k) {
            att_probs[att_offset + t_k] = 0.0f;
        }

        // 3. Value aggregation
        for (int d = 0; d < head_dim; ++d) {
            float val_sum = 0.0f;
            for (int t_k = 0; t_k <= t_q; ++t_k) {
                float v = qkv[b * T * 3 * C + t_k * 3 * C + 2 * C + head * head_dim + d];
                val_sum += scores[t_k] * v;
            }
            O[b * T * C + t_q * C + head * head_dim + d] = val_sum;
        }
    }
}

__global__ void attention_backward_kernel(const float* dO, const float* qkv, const float* att_probs, float* dqkv,
                                          int B, int T, int C, int num_heads, int head_dim, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = B * num_heads * T;
    if (idx < total_queries) {
        int b = idx / (num_heads * T);
        int head = (idx / T) % num_heads;
        int t_q = idx % T;

        int att_offset = b * num_heads * T * T + head * T * T + t_q * T;
        float dp_vec[1024];
        float sum_dp_p = 0.0f;

        // 1. Compute dot product of dO and V
        for (int t_k = 0; t_k <= t_q; ++t_k) {
            float dp = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float do_val = dO[b * T * C + t_q * C + head * head_dim + d];
                float v = qkv[b * T * 3 * C + t_k * 3 * C + 2 * C + head * head_dim + d];
                dp += do_val * v;
            }
            dp_vec[t_k] = dp;
            float p = att_probs[att_offset + t_k];
            sum_dp_p += dp * p;
        }

        // 2. Accumulate gradients
        for (int t_k = 0; t_k <= t_q; ++t_k) {
            float p = att_probs[att_offset + t_k];
            float dS_scaled = p * (dp_vec[t_k] - sum_dp_p) * scale;

            for (int d = 0; d < head_dim; ++d) {
                // Gradient for V (value) at t_k
                float do_val = dO[b * T * C + t_q * C + head * head_dim + d];
                atomicAdd(&dqkv[b * T * 3 * C + t_k * 3 * C + 2 * C + head * head_dim + d], p * do_val);

                // Gradient for Q (query) at t_q
                float k = qkv[b * T * 3 * C + t_k * 3 * C + C + head * head_dim + d];
                atomicAdd(&dqkv[b * T * 3 * C + t_q * 3 * C + head * head_dim + d], dS_scaled * k);

                // Gradient for K (key) at t_k
                float q = qkv[b * T * 3 * C + t_q * 3 * C + head * head_dim + d];
                atomicAdd(&dqkv[b * T * 3 * C + t_k * 3 * C + C + head * head_dim + d], dS_scaled * q);
            }
        }
    }
}

void launch_attention_forward(const float* qkv, float* att_probs, float* O, 
                              int B, int T, int C, int num_heads, int head_dim, float scale) {
    int N = B * num_heads * T;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    attention_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(qkv, att_probs, O, B, T, C, num_heads, head_dim, scale);
}

void launch_attention_backward(const float* dO, const float* qkv, const float* att_probs, float* dqkv,
                               int B, int T, int C, int num_heads, int head_dim, float scale) {
    int N = B * num_heads * T;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    attention_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(dO, qkv, att_probs, dqkv, B, T, C, num_heads, head_dim, scale);
}