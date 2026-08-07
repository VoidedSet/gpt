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
