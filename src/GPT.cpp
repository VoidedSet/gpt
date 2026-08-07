#include "GPT.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstring>

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

    if (logits_.device_data() != nullptr) {
        logits_.to_cpu(DATA_);
    }

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

    if (logits_.device_data() != nullptr) {
        dZ.to_gpu(DATA_);
        embedding_.wte().to_gpu(GRAD_);

        NastyTensors dX_norm_2d = dZ.matmul(embedding_.wte());



        NastyTensors x_norm_2d = ln_f_.x_hat().reshape({B * T, C});
        
        NastyTensors::gemm(true, false, 
                           V, C, B * T, 
                           1.0f, 
                           dZ.device_data(), V, 
                           x_norm_2d.device_data(), C, 
                           1.0f, 
                           embedding_.wte().device_grad(), C);



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
        logits.to_cpu(DATA_);
        
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

bool GPT::save_binary(const std::string& filepath, const Tokenizer& tokenizer, int quantization_level) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[-] Error: Failed to open file for saving model weights: " << filepath << "\n";
        return false;
    }

    int magic = 0x47505432; // 'GPT2' in ASCII hex
    int version = 2; // Bump version to 2 for BPE/Quantization support
    int vocab_size = static_cast<int>(vocab_size_);
    int max_seq_len = static_cast<int>(max_seq_len_);
    int embedding_dim = static_cast<int>(embedding_dim_);
    int num_heads = static_cast<int>(num_heads_);
    int num_layers = static_cast<int>(num_layers_);
    int tok_type = static_cast<int>(tokenizer.get_type()); // 0 = CHAR, 1 = BPE
    int quant_level = quantization_level; // 0 = FP32, 1 = BF16, 2 = INT8

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    out.write(reinterpret_cast<const char*>(&max_seq_len), sizeof(max_seq_len));
    out.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));
    out.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    out.write(reinterpret_cast<const char*>(&tok_type), sizeof(tok_type));
    out.write(reinterpret_cast<const char*>(&quant_level), sizeof(quant_level));

    // Serialize Vocabulary
    if (tok_type == 0) {
        // CHAR type tokenizer: save as a flat sequence of characters
        const std::vector<char>& id_to_char = tokenizer.get_id_to_char();
        assert(id_to_char.size() == vocab_size_);
        out.write(id_to_char.data(), vocab_size * sizeof(char));
        
        // Pad to 4-byte boundary
        size_t vocab_bytes = vocab_size * sizeof(char);
        size_t padding = (4 - (vocab_bytes % 4)) % 4;
        if (padding > 0) {
            char pad[3] = {0, 0, 0};
            out.write(pad, padding);
        }
    } else {
        // BPE type tokenizer:
        // We write the number of merges, followed by the merges themselves as pair of ints: (left, right).
        const std::vector<std::pair<int, int>>& merges = tokenizer.get_merges();
        int num_merges = static_cast<int>(merges.size());
        out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));
        for (const auto& merge : merges) {
            out.write(reinterpret_cast<const char*>(&merge.first), sizeof(merge.first));
            out.write(reinterpret_cast<const char*>(&merge.second), sizeof(merge.second));
        }
    }

    // Cast away constness to call get_parameters()
    GPT* non_const_this = const_cast<GPT*>(this);
    std::vector<NastyTensors*> params = non_const_this->get_parameters();

    for (auto* param : params) {
        param->to_cpu(DATA_);
        const float* p_data = param->data();
        size_t num_elements = param->size();

        if (quant_level == 0) {
            // FP32: Standard 32-bit floats
            out.write(reinterpret_cast<const char*>(p_data), num_elements * sizeof(float));
        } else if (quant_level == 1) {
            // BF16: Write top 16 bits of 32-bit floats
            std::vector<uint16_t> bf16_data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &p_data[i], sizeof(float));
                bf16_data[i] = static_cast<uint16_t>(bits >> 16);
            }
            out.write(reinterpret_cast<const char*>(bf16_data.data()), num_elements * sizeof(uint16_t));
            
            // Align to 4-byte boundary if needed
            if ((num_elements * sizeof(uint16_t)) % 4 != 0) {
                uint16_t pad = 0;
                out.write(reinterpret_cast<const char*>(&pad), sizeof(uint16_t));
            }
        } else if (quant_level == 2) {
            // INT8 Quantization: Symmetric tensor/channel scale
            float max_val = 0.0f;
            for (size_t i = 0; i < num_elements; ++i) {
                float abs_v = std::abs(p_data[i]);
                if (abs_v > max_val) max_val = abs_v;
            }
            
            float scale = max_val / 127.0f;
            if (scale == 0.0f) scale = 1.0f;
            
            // Write scale (4 bytes float)
            out.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
            
            // Quantize and write bytes
            std::vector<int8_t> q_data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                float val = p_data[i] / scale;
                int q = static_cast<int>(std::round(val));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                q_data[i] = static_cast<int8_t>(q);
            }
            out.write(reinterpret_cast<const char*>(q_data.data()), num_elements * sizeof(int8_t));
            
            // Align to 4-byte boundary
            size_t bytes_written = sizeof(scale) + num_elements * sizeof(int8_t);
            size_t padding = (4 - (bytes_written % 4)) % 4;
            if (padding > 0) {
                char pad[3] = {0, 0, 0};
                out.write(pad, padding);
            }
        }
    }

    std::cout << "[+] Saved model binary to " << filepath << "\n";
    return true;
}
