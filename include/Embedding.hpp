#pragma once

#include "NastyTensors.hpp"
#include <vector>

class Embedding {
private:
    size_t vocab_size_;
    size_t max_seq_len_;
    size_t embedding_dim_;
    NastyTensors wte_;
    NastyTensors wpe_;
    mutable std::vector<int> input_tokens_;

    void init_weights();

public:
    Embedding(size_t vocab_size, size_t max_seq_len, size_t embedding_dim);
    ~Embedding() = default;

    NastyTensors forward(const std::vector<int>& input_tokens, size_t B, size_t T) const;
    void backward(const NastyTensors& dY);

    const NastyTensors& wte() const { return wte_; }
    const NastyTensors& wpe() const { return wpe_; }
    NastyTensors& wte() { return wte_; }
    NastyTensors& wpe() { return wpe_; }
};