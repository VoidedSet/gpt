#pragma once

#include "NastyTensors.hpp"
#include <vector>

class MLP {
public:
    size_t C_dim;

    Parameter W_fc;
    Parameter b_fc;
    Parameter W_proj;
    Parameter b_proj;

    NastyTensors fc_act;
    NastyTensors gelu_act;
    NastyTensors d_fc_act;
    NastyTensors d_gelu_act;

    MLP(size_t C_dim);
    void forward(const NastyTensors& x, NastyTensors& y);
    void backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx);
    std::vector<Parameter*> get_parameters();
};
