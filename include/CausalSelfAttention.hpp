#pragma once

#include "NastyTensors.hpp"
#include <vector>

class CausalSelfAttention {
private:
    size_t embedding_dim_,
           num_heads_,
           head_dim_;

    NastyTensors w_qkv_, 
                 b_qkv_,
                 w_proj_;
    NastyTensors b_proj_;

    mutable NastyTensors x_2d_;
    mutable NastyTensors qkv_;
    mutable NastyTensors o_;
    mutable NastyTensors att_probs_;

    void init_weights();

public:
    CausalSelfAttention(size_t embedding_dim, size_t num_heads);
    ~CausalSelfAttention() = default;

    NastyTensors forward(const NastyTensors& X) const;
    void backward(const NastyTensors& dY, NastyTensors& dX);

    const NastyTensors& w_qkv() const { return w_qkv_; }
    const NastyTensors& b_qkv() const { return b_qkv_; }
    const NastyTensors& w_proj() const { return w_proj_; }
    const NastyTensors& b_proj() const { return b_proj_; }
    NastyTensors& w_qkv() { return w_qkv_; }
    NastyTensors& b_qkv() { return b_qkv_; }
    NastyTensors& w_proj() { return w_proj_; }
    NastyTensors& b_proj() { return b_proj_; }
};
