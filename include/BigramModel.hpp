#ifndef BIGRAMMODEL_HPP
#define BIGRAMMODEL_HPP

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

class BigramModel {
private:
    int vocab_size;
    // Flat 1D lookup table of shape [vocab_size, vocab_size]
    std::vector<float> W; 
    std::mt19937 rng;

public:
    BigramModel(int vocab_size) : vocab_size(vocab_size), rng(1337) {
        // Initialize weights with small random floats
        W.resize(vocab_size * vocab_size);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (auto& w : W) {
            w = dist(rng);
        }
    }

    const float* get_logits(int token_x) const { return &W[token_x * vocab_size]; }

    std::vector<float> softmax(const float* logits) const;
    float compute_loss(const std::vector<int>& X, const std::vector<int>& Y) const;
    void train_step(const std::vector<int>& X, const std::vector<int>& Y, float learning_rate);
    std::vector<int> generate(int start_token, int max_new_tokens);

    int get_vocab_size() const { return vocab_size; }
};

#endif // BIGRAMMODEL_HPP