#include "Embedding.hpp"
#include <random>
#include <cassert>

Embedding::Embedding(size_t vocab_size, size_t max_seq_len, size_t embedding_dim)
    : vocab_size_(vocab_size), max_seq_len_(max_seq_len), embedding_dim_(embedding_dim),
      wte_({vocab_size, embedding_dim}), wpe_({max_seq_len, embedding_dim}) {
    init_weights();
}

void Embedding::init_weights() {
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    float* wte_ptr = wte_.data();
    size_t wte_size = wte_.size();
    for (size_t i = 0; i < wte_size; ++i) {
        wte_ptr[i] = dist(gen);
    }

    float* wpe_ptr = wpe_.data();
    size_t wpe_size = wpe_.size();
    for (size_t i = 0; i < wpe_size; ++i) {
        wpe_ptr[i] = dist(gen);
    }
}

NastyTensors Embedding::forward(const std::vector<int>& input_tokens, size_t B, size_t T) const {
    assert(input_tokens.size() == B * T && "Invalid BxT Size.");
    assert(T <= max_seq_len_ && "Token Size Exceeded");

    NastyTensors output({B, T, embedding_dim_});

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            int token_id = input_tokens[b * T + t];
            assert(token_id >= 0 && static_cast<size_t>(token_id) < vocab_size_ && "Invalid Token ID.");

            for (size_t c = 0; c < embedding_dim_; ++c) 
                output(b, t, c) = wte_(token_id, c) + wpe_(t, c);
        }
    }

    return output;
}