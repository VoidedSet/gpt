#pragma once

#include "NastyTensors.hpp"

class FeedForward {
private:
    size_t embedding_dim_;
    NastyTensors w_fc_;
    NastyTensors b_fc_;
    NastyTensors w_proj_;
    NastyTensors b_proj_;

    mutable NastyTensors x_2d_;
    mutable NastyTensors h1_2d_;
    mutable NastyTensors h2_2d_;

    void init_weights();

public:
    FeedForward(size_t embedding_dim);
    ~FeedForward() = default;

    NastyTensors forward(const NastyTensors& X) const;
    void backward(const NastyTensors& dY, NastyTensors& dX);

    const NastyTensors& w_fc() const { return w_fc_; }
    const NastyTensors& b_fc() const { return b_fc_; }
    const NastyTensors& w_proj() const { return w_proj_; }
    const NastyTensors& b_proj() const { return b_proj_; }
    NastyTensors& w_fc() { return w_fc_; }
    NastyTensors& b_fc() { return b_fc_; }
    NastyTensors& w_proj() { return w_proj_; }
    NastyTensors& b_proj() { return b_proj_; }
};
