#include "GPT.hpp"
#include <cmath>
#include <cassert>
#include <cstdlib>

GPT::GPT(size_t vocab_size, size_t max_seq_len, size_t embedding_dim, size_t num_heads, size_t num_layers)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embedding_dim_(embedding_dim),
      num_heads_(num_heads),
      num_layers_(num_layers),
      embedding_(vocab_size, max_seq_len, embedding_dim),
      ln_f_(embedding_dim) {
    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(embedding_dim, num_heads);
    }
}

NastyTensors GPT::forward(const std::vector<int>& input_tokens, size_t B, size_t T) const {
    B_ = B;
    T_ = T;

    NastyTensors x = embedding_.forward(input_tokens, B, T);

    for (size_t l = 0; l < num_layers_; ++l) {
        x = blocks_[l].forward(x);
    }

    ln_f_.forward(x);

    NastyTensors x_2d = x.reshape({B * T, embedding_dim_});

    logits_ = x_2d.matmul_transposed_b(embedding_.wte());
    return logits_;
}

float GPT::backward(const std::vector<int>& targets) {
    assert(targets.size() == B_ * T_ && "Invalid target size.");

    size_t B = B_;
    size_t T = T_;
    size_t V = vocab_size_;
    size_t C = embedding_dim_;

    NastyTensors dZ({B * T, V}, 0.0f);
    float total_loss = 0.0f;
    float* dz_ptr = dZ.data();
    const float* logits_ptr = logits_.data();

    for (size_t i = 0; i < B * T; ++i) {
        int target = targets[i];
        assert(target >= 0 && static_cast<size_t>(target) < V);

        float max_val = logits_ptr[i * V];
        for (size_t j = 1; j < V; ++j) {
            if (logits_ptr[i * V + j] > max_val) {
                max_val = logits_ptr[i * V + j];
            }
        }

        std::vector<float> exps(V);
        float sum_exp = 0.0f;
        for (size_t j = 0; j < V; ++j) {
            exps[j] = std::exp(logits_ptr[i * V + j] - max_val);
            sum_exp += exps[j];
        }

        float prob_target = exps[target] / sum_exp;
        total_loss -= std::log(prob_target);

        float factor = 1.0f / (B * T);
        for (size_t j = 0; j < V; ++j) {
            float prob = exps[j] / sum_exp;
            dz_ptr[i * V + j] = (prob - (j == static_cast<size_t>(target) ? 1.0f : 0.0f)) * factor;
        }
    }

    total_loss /= (B * T);

    NastyTensors dX_norm_2d = dZ.matmul(embedding_.wte());

    float* dwte = embedding_.wte().grad();
    NastyTensors x_norm_2d = ln_f_.x_hat().reshape({B * T, C});
    for (size_t i = 0; i < B * T; ++i) {
        for (size_t v = 0; v < V; ++v) {
            float dz_val = dZ(i, v);
            if (dz_val != 0.0f) {
                for (size_t c = 0; c < C; ++c) {
                    dwte[v * C + c] += dz_val * x_norm_2d(i, c);
                }
            }
        }
    }

    NastyTensors dX_norm = dX_norm_2d.reshape({B, T, C});
    NastyTensors dX_blocks({B, T, C});
    ln_f_.backward(dX_norm, dX_blocks);

    NastyTensors curr_grad = dX_blocks;
    for (int l = static_cast<int>(num_layers_) - 1; l >= 0; --l) {
        NastyTensors next_grad({B, T, C});
        blocks_[l].backward(curr_grad, next_grad);
        curr_grad = next_grad;
    }

    embedding_.backward(curr_grad);

    return total_loss;
}

std::vector<NastyTensors*> GPT::get_parameters() {
    std::vector<NastyTensors*> params;
    params.push_back(&embedding_.wte());
    params.push_back(&embedding_.wpe());
    for (size_t i = 0; i < num_layers_; ++i) {
        params.push_back(&blocks_[i].ln1().gamma());
        params.push_back(&blocks_[i].ln1().beta());
        params.push_back(&blocks_[i].attn().w_qkv());
        params.push_back(&blocks_[i].attn().b_qkv());
        params.push_back(&blocks_[i].attn().w_proj());
        params.push_back(&blocks_[i].attn().b_proj());
        
        params.push_back(&blocks_[i].ln2().gamma());
        params.push_back(&blocks_[i].ln2().beta());
        params.push_back(&blocks_[i].mlp().w_fc());
        params.push_back(&blocks_[i].mlp().b_fc());
        params.push_back(&blocks_[i].mlp().w_proj());
        params.push_back(&blocks_[i].mlp().b_proj());
    }
    params.push_back(&ln_f_.gamma());
    params.push_back(&ln_f_.beta());
    return params;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, size_t max_new_tokens) {
    std::vector<int> generated = prompt;
    for (size_t i = 0; i < max_new_tokens; ++i) {
        size_t start_idx = 0;
        size_t len = generated.size();
        if (len > max_seq_len_) {
            start_idx = len - max_seq_len_;
            len = max_seq_len_;
        }
        std::vector<int> input(generated.begin() + start_idx, generated.end());
        
        NastyTensors logits = forward(input, 1, len);
        
        float* logits_ptr = logits.data();
        size_t last_row_offset = (len - 1) * vocab_size_;
        
        float max_logit = logits_ptr[last_row_offset];
        for (size_t j = 1; j < vocab_size_; ++j) {
            if (logits_ptr[last_row_offset + j] > max_logit) {
                max_logit = logits_ptr[last_row_offset + j];
            }
        }
        
        std::vector<float> probs(vocab_size_);
        float sum_prob = 0.0f;
        for (size_t j = 0; j < vocab_size_; ++j) {
            probs[j] = std::exp(logits_ptr[last_row_offset + j] - max_logit);
            sum_prob += probs[j];
        }
        
        float r = (static_cast<float>(rand()) / RAND_MAX) * sum_prob;
        float running_sum = 0.0f;
        int next_token = 0;
        for (size_t j = 0; j < vocab_size_; ++j) {
            running_sum += probs[j];
            if (r <= running_sum) {
                next_token = j;
                break;
            }
        }
        
        generated.push_back(next_token);
    }
    return generated;
}
