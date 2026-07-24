#pragma once

#include "NastyTensors.hpp"
#include "Attention.hpp"
#include "MLP.hpp"
#include <vector>
#include <random>

class Block {
public:
    Parameter ln1_gamma;
    Parameter ln1_beta;
    MultiHeadAttention attn;
    Parameter ln2_gamma;
    Parameter ln2_beta;
    MLP mlp;

    NastyTensors ln1_out;
    NastyTensors ln1_mean;
    NastyTensors ln1_rstd;
    NastyTensors attn_out;
    NastyTensors x_after_attn;

    NastyTensors ln2_out;
    NastyTensors ln2_mean;
    NastyTensors ln2_rstd;
    NastyTensors mlp_out;

    NastyTensors d_attn_out;
    NastyTensors d_ln1_out;
    NastyTensors d_mlp_out;
    NastyTensors d_ln2_out;

    Block(size_t C_dim, size_t num_heads);
    void forward(const NastyTensors& x, NastyTensors& y, size_t B, size_t T);
    void backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx, size_t B, size_t T);
    std::vector<Parameter*> get_parameters();
};

class GPT {
public:
    size_t vocab_size;
    size_t max_seq_len;
    size_t C_dim;
    size_t num_heads;
    size_t num_layers;

    Parameter W_te;
    Parameter W_pe;
    
    std::vector<Block> blocks;
    
    Parameter lnf_gamma;
    Parameter lnf_beta;
    Parameter W_lm;

    std::vector<NastyTensors> block_acts;
    NastyTensors x_emb;
    NastyTensors lnf_out;
    NastyTensors lnf_mean;
    NastyTensors lnf_rstd;
    NastyTensors logits;
    NastyTensors probs;

    NastyTensors d_logits;
    NastyTensors d_lnf_out;
    std::vector<NastyTensors> block_grads;
    NastyTensors d_x_emb;

    GPT(size_t vocab_size, size_t max_seq_len = 256, size_t C_dim = 128, size_t num_heads = 4, size_t num_layers = 6);
    
    float forward(const std::vector<int>& X, const std::vector<int>& Y, size_t B, size_t T);
    void backward(const std::vector<int>& X, const std::vector<int>& Y, size_t B, size_t T);
    
    std::vector<Parameter*> get_parameters();
    std::vector<int> generate(int start_token, int max_new_tokens, std::mt19937& rng);
};
