#pragma once


void launch_embedding_forward(const int* tokens, const float* wte_, const float* wpe_, 
                              float* output,
                              int B, int T, int C);

void launch_embedding_backward(const int* tokens, const float* dY, float* dwte, float* dwpe, 
                                int B, int T, int C);

void launch_layernorm_forward(float* X, const float* gamma, 
                          const float* beta, float* x_hat, 
                          float* mean, float* var, int B,
                          int T, int C, float eps);

void launch_layernorm_backward(const float* dY, const float* x_hat, const float* var, const float* gamma,
                                float* dgamma, float* dbeta, float* dX, int B, int T, int C, float eps);

void launch_gelu_forward(float* X, int N);
void launch_gelu_backward(const float* h1, const float* dh2, float* dh1, int N);

void launch_add_bias(float* X, const float* bias, int rows, int cols);
void launch_accumulate_bias_grad(const float* dY, float* dbias, int rows, int cols);

void launch_attention_forward(const float* qkv, float* att_probs, float* O, 
                              int B, int T, int C, int num_heads, int head_dim, float scale);

void launch_attention_backward(const float* dO, const float* qkv, const float* att_probs, float* dqkv,
                               int B, int T, int C, int num_heads, int head_dim, float scale);

void launch_add_tensors(float* dest, const float* src, int N);

void launch_adamw_step(float* w, const float* g, float* m, float* v,
                       size_t N, float lr, float beta1, float beta2,
                       float eps, float weight_decay,
                       float bias_correction1, float bias_correction2);
