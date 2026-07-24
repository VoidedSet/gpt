#pragma once

#include "NastyTensors.hpp"
#include <vector>

class AdamW {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    size_t step;

    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;

    AdamW(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f);
    void update(const std::vector<Parameter*>& params);
    void zero_grad(const std::vector<Parameter*>& params);
};
