#pragma once


void launch_embedding_forward(const int* tokens, const float* wte_, const float* wpe_, 
                              float* output,
                              int B, int T, int C);

void launch_embedding_backward(const int* tokens, const float* dY, float* dwte, float* dwpe, 
                                int B, int T, int C);
