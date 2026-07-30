#pragma once


void launch_embedding_forward(const int* tokens, const float* wte_, const float* wpe_, 
                              float* output,
                              int B, int T, int C);