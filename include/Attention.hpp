#pragma once

#include "NastyTensors.hpp"
#include <vector>

class MultiHeadAttention {
public:
    size_t C_dim;
    size_t num_heads;
    size_t head_dim;

    Parameter W_qkv;
    Parameter b_qkv;
    Parameter W_proj;
    Parameter b_proj;

    NastyTensors qkv_act;
    NastyTensors Q;
    NastyTensors K;
    NastyTensors V;
    NastyTensors S;
    NastyTensors O_heads;
    NastyTensors O_flat;
    NastyTensors d_O_flat;
    NastyTensors dO_heads;
    NastyTensors dS;
    NastyTensors dS_raw;
    NastyTensors dQ;
    NastyTensors dK;
    NastyTensors dV;
    NastyTensors d_qkv_act;

    MultiHeadAttention(size_t C_dim, size_t num_heads);
    void forward(const NastyTensors& x, NastyTensors& y, size_t B, size_t T);
    void backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx, size_t B, size_t T);
    std::vector<Parameter*> get_parameters();
};
