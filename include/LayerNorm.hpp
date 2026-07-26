#pragma once

#include "NastyTensors.hpp"

class LayerNorm {
private:
    size_t embedding_dim_;
    float epsilon_;
    NastyTensors gamma_;
    NastyTensors beta_;

public:
    LayerNorm(size_t embedding_dim, float epsilon = 1e-5f);
    ~LayerNorm() = default;

    void forward(NastyTensors& X) const;

    const NastyTensors& gamma() const { return gamma_; }
    const NastyTensors& beta() const { return beta_; }
};
