#pragma once

#include "NastyTensors.hpp"
#include "LayerNorm.hpp"
#include "CausalSelfAttention.hpp"
#include "FeedForward.hpp"

class Block {
private:
    LayerNorm ln1_;
    CausalSelfAttention attn_;
    LayerNorm ln2_;
    FeedForward mlp_;

public:
    Block(size_t embedding_dim, size_t num_heads);
    ~Block() = default;

    NastyTensors forward(const NastyTensors& X) const;
    void backward(const NastyTensors& dY, NastyTensors& dX);

    const LayerNorm& ln1() const { return ln1_; }
    const CausalSelfAttention& attn() const { return attn_; }
    const LayerNorm& ln2() const { return ln2_; }
    const FeedForward& mlp() const { return mlp_; }

    LayerNorm& ln1() { return ln1_; }
    CausalSelfAttention& attn() { return attn_; }
    LayerNorm& ln2() { return ln2_; }
    FeedForward& mlp() { return mlp_; }
};
