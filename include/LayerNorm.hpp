#pragma once

#include "NastyTensors.hpp"

class LayerNorm {
private:
    size_t embedding_dim_;
    float epsilon_;
    NastyTensors gamma_;
    NastyTensors beta_;

    mutable NastyTensors x_hat_;
    mutable std::vector<float> mean_;
    mutable std::vector<float> var_;

public:
    LayerNorm(size_t embedding_dim, float epsilon = 1e-5f);
    ~LayerNorm() = default;

    void forward(NastyTensors& X) const;
    void backward(const NastyTensors& dY, NastyTensors& dX);

    const NastyTensors& gamma() const { return gamma_; }
    const NastyTensors& beta() const { return beta_; }
    NastyTensors& gamma() { return gamma_; }
    NastyTensors& beta() { return beta_; }
    const NastyTensors& x_hat() const { return x_hat_; }
};
