#pragma once

#include "NastyTensors.hpp"
#include "Embedding.hpp"
#include "Block.hpp"
#include "LayerNorm.hpp"
#include <vector>

class GPT {
private:
    size_t vocab_size_;
    size_t max_seq_len_;
    size_t embedding_dim_;
    size_t num_heads_;
    size_t num_layers_;

    Embedding embedding_;
    std::vector<Block> blocks_;
    LayerNorm ln_f_;

    mutable NastyTensors logits_;
    mutable size_t B_ = 0;
    mutable size_t T_ = 0;

public:
    GPT(size_t vocab_size, size_t max_seq_len, size_t embedding_dim, size_t num_heads, size_t num_layers);
    ~GPT() = default;

    NastyTensors forward(const std::vector<int>& input_tokens, size_t B, size_t T) const;
    float backward(const std::vector<int>& targets);

    std::vector<NastyTensors*> get_parameters();
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens);
    bool save_binary(const std::string& filepath, const std::vector<char>& id_to_char) const;
};
