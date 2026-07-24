#include "GPT.hpp"
#include "Ops.hpp"
#include <algorithm>
#include <cmath>

Block::Block(size_t C_dim, size_t num_heads)
    : attn(C_dim, num_heads), mlp(C_dim) {
    ln1_gamma = Parameter({C_dim});
    ln1_beta = Parameter({C_dim});
    ln2_gamma = Parameter({C_dim});
    ln2_beta = Parameter({C_dim});

    float* ln1_g = ln1_gamma.value.data();
    float* ln2_g = ln2_gamma.value.data();
    for (size_t i = 0; i < C_dim; ++i) {
        ln1_g[i] = 1.0f;
        ln2_g[i] = 1.0f;
    }
}

void Block::forward(const NastyTensors& x, NastyTensors& y, size_t B, size_t T) {
    size_t M = B * T;
    size_t C_dim = ln1_gamma.value.size();

    if (ln1_out.shape() != std::vector<size_t>{M, C_dim}) {
        ln1_out = NastyTensors({M, C_dim});
        ln1_mean = NastyTensors({M});
        ln1_rstd = NastyTensors({M});
        attn_out = NastyTensors({M, C_dim});
        x_after_attn = NastyTensors({M, C_dim});
        ln2_out = NastyTensors({M, C_dim});
        ln2_mean = NastyTensors({M});
        ln2_rstd = NastyTensors({M});
        mlp_out = NastyTensors({M, C_dim});
    }

    ops::layernorm_forward(x, ln1_gamma.value, ln1_beta.value, ln1_out, ln1_mean, ln1_rstd);
    attn.forward(ln1_out, attn_out, B, T);
    ops::add_forward(x, attn_out, x_after_attn);
    ops::layernorm_forward(x_after_attn, ln2_gamma.value, ln2_beta.value, ln2_out, ln2_mean, ln2_rstd);
    mlp.forward(ln2_out, mlp_out);
    ops::add_forward(x_after_attn, mlp_out, y);
}

void Block::backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx, size_t B, size_t T) {
    size_t M = B * T;
    size_t C_dim = ln1_gamma.value.size();

    if (d_mlp_out.shape() != std::vector<size_t>{M, C_dim}) {
        d_mlp_out = NastyTensors({M, C_dim});
        d_ln2_out = NastyTensors({M, C_dim}, 0.0f);
        d_attn_out = NastyTensors({M, C_dim});
        d_ln1_out = NastyTensors({M, C_dim}, 0.0f);
    } else {
        float* p1 = d_ln2_out.data(); for (size_t i = 0; i < d_ln2_out.size(); ++i) p1[i] = 0.0f;
        float* p2 = d_ln1_out.data(); for (size_t i = 0; i < d_ln1_out.size(); ++i) p2[i] = 0.0f;
    }

    NastyTensors dx_after_attn = dy.clone();
    float* d_mlp_ptr = d_mlp_out.data();
    const float* dy_ptr = dy.data();
    for (size_t i = 0; i < dy.size(); ++i) {
        d_mlp_ptr[i] = dy_ptr[i];
    }

    mlp.backward(d_mlp_out, ln2_out, d_ln2_out);
    ops::layernorm_backward(d_ln2_out, x_after_attn, ln2_gamma.value, ln2_mean, ln2_rstd, dx_after_attn, ln2_gamma.grad, ln2_beta.grad);

    float* dx_after_ptr = dx_after_attn.data();
    float* dx_ptr = dx.data();
    float* d_attn_ptr = d_attn_out.data();
    for (size_t i = 0; i < dx_after_attn.size(); ++i) {
        dx_ptr[i] = dx_after_ptr[i];
        d_attn_ptr[i] = dx_after_ptr[i];
    }

    attn.backward(d_attn_out, ln1_out, d_ln1_out, B, T);
    ops::layernorm_backward(d_ln1_out, x, ln1_gamma.value, ln1_mean, ln1_rstd, dx, ln1_gamma.grad, ln1_beta.grad);
}

std::vector<Parameter*> Block::get_parameters() {
    std::vector<Parameter*> params = attn.get_parameters();
    std::vector<Parameter*> mlp_params = mlp.get_parameters();
    params.push_back(&ln1_gamma);
    params.push_back(&ln1_beta);
    params.push_back(&ln2_gamma);
    params.push_back(&ln2_beta);
    params.insert(params.end(), mlp_params.begin(), mlp_params.end());
    return params;
}

GPT::GPT(size_t vocab_size, size_t max_seq_len, size_t C_dim, size_t num_heads, size_t num_layers)
    : vocab_size(vocab_size), max_seq_len(max_seq_len), C_dim(C_dim), num_heads(num_heads), num_layers(num_layers) {
    
    W_te = Parameter({vocab_size, C_dim});
    W_pe = Parameter({max_seq_len, C_dim});
    lnf_gamma = Parameter({C_dim});
    lnf_beta = Parameter({C_dim});
    W_lm = Parameter({C_dim, vocab_size});

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    float* w_te_ptr = W_te.value.data();
    for (size_t i = 0; i < W_te.value.size(); ++i) w_te_ptr[i] = dist(rng);

    float* w_pe_ptr = W_pe.value.data();
    for (size_t i = 0; i < W_pe.value.size(); ++i) w_pe_ptr[i] = dist(rng);

    float* w_lm_ptr = W_lm.value.data();
    for (size_t i = 0; i < W_lm.value.size(); ++i) w_lm_ptr[i] = dist(rng);

    float* lnf_g = lnf_gamma.value.data();
    for (size_t i = 0; i < C_dim; ++i) lnf_g[i] = 1.0f;

    for (size_t i = 0; i < num_layers; ++i) {
        blocks.emplace_back(C_dim, num_heads);
    }
}

float GPT::forward(const std::vector<int>& X, const std::vector<int>& Y, size_t B, size_t T) {
    size_t M = B * T;

    if (x_emb.shape() != std::vector<size_t>{M, C_dim}) {
        x_emb = NastyTensors({M, C_dim});
        block_acts.clear();
        for (size_t i = 0; i <= num_layers; ++i) {
            block_acts.emplace_back(std::vector<size_t>{M, C_dim});
        }
        lnf_out = NastyTensors({M, C_dim});
        lnf_mean = NastyTensors({M});
        lnf_rstd = NastyTensors({M});
        logits = NastyTensors({M, vocab_size});
        probs = NastyTensors({M, vocab_size});
    }

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t idx = b * T + t;
            int tok = X[idx];
            size_t pos = t;
            for (size_t c = 0; c < C_dim; ++c) {
                x_emb(idx, c) = W_te.value(tok, c) + W_pe.value(pos, c);
            }
        }
    }

    float* b0 = block_acts[0].data();
    const float* xe = x_emb.data();
    for (size_t i = 0; i < x_emb.size(); ++i) b0[i] = xe[i];

    for (size_t l = 0; l < num_layers; ++l) {
        blocks[l].forward(block_acts[l], block_acts[l+1], B, T);
    }

    ops::layernorm_forward(block_acts[num_layers], lnf_gamma.value, lnf_beta.value, lnf_out, lnf_mean, lnf_rstd);
    ops::matmul(lnf_out, W_lm.value, logits);

    return ops::cross_entropy_forward(logits, Y, probs);
}

void GPT::backward(const std::vector<int>& X, const std::vector<int>& Y, size_t B, size_t T) {
    size_t M = B * T;

    if (d_logits.shape() != std::vector<size_t>{M, vocab_size}) {
        d_logits = NastyTensors({M, vocab_size}, 0.0f);
        d_lnf_out = NastyTensors({M, C_dim}, 0.0f);
        d_x_emb = NastyTensors({M, C_dim}, 0.0f);
        block_grads.clear();
        for (size_t i = 0; i <= num_layers; ++i) {
            block_grads.emplace_back(std::vector<size_t>{M, C_dim}, 0.0f);
        }
    } else {
        float* p1 = d_logits.data();  for (size_t i = 0; i < d_logits.size(); ++i) p1[i] = 0.0f;
        float* p2 = d_lnf_out.data(); for (size_t i = 0; i < d_lnf_out.size(); ++i) p2[i] = 0.0f;
        float* p3 = d_x_emb.data();   for (size_t i = 0; i < d_x_emb.size(); ++i) p3[i] = 0.0f;
        for (size_t l = 0; l <= num_layers; ++l) {
            float* p = block_grads[l].data();
            for (size_t i = 0; i < block_grads[l].size(); ++i) p[i] = 0.0f;
        }
    }

    ops::cross_entropy_backward(probs, Y, d_logits);

    for (size_t i = 0; i < M; ++i) {
        for (size_t c = 0; c < C_dim; ++c) {
            float sum = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v) {
                sum += d_logits(i, v) * W_lm.value(c, v);
            }
            d_lnf_out(i, c) = sum;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t c = 0; c < C_dim; ++c) {
            float val = lnf_out(i, c);
            for (size_t v = 0; v < vocab_size; ++v) {
                W_lm.grad(c, v) += val * d_logits(i, v);
            }
        }
    }

    ops::layernorm_backward(d_lnf_out, block_acts[num_layers], lnf_gamma.value, lnf_mean, lnf_rstd, block_grads[num_layers], lnf_gamma.grad, lnf_beta.grad);

    for (int l = static_cast<int>(num_layers) - 1; l >= 0; --l) {
        blocks[l].backward(block_grads[l+1], block_acts[l], block_grads[l], B, T);
    }

    float* d_x_emb_ptr = d_x_emb.data();
    float* b0_grad_ptr = block_grads[0].data();
    for (size_t i = 0; i < d_x_emb.size(); ++i) d_x_emb_ptr[i] = b0_grad_ptr[i];

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t idx = b * T + t;
            int tok = X[idx];
            size_t pos = t;
            for (size_t c = 0; c < C_dim; ++c) {
                float grad_val = d_x_emb(idx, c);
                W_te.grad(tok, c) += grad_val;
                W_pe.grad(pos, c) += grad_val;
            }
        }
    }
}

std::vector<Parameter*> GPT::get_parameters() {
    std::vector<Parameter*> params;
    params.push_back(&W_te);
    params.push_back(&W_pe);
    for (auto& b : blocks) {
        std::vector<Parameter*> bp = b.get_parameters();
        params.insert(params.end(), bp.begin(), bp.end());
    }
    params.push_back(&lnf_gamma);
    params.push_back(&lnf_beta);
    params.push_back(&W_lm);
    return params;
}

std::vector<int> GPT::generate(int start_token, int max_new_tokens, std::mt19937& rng) {
    std::vector<int> generated;
    generated.push_back(start_token);

    for (int step = 0; step < max_new_tokens - 1; ++step) {
        std::vector<int> context = generated;
        if (context.size() > max_seq_len) {
            context.erase(context.begin(), context.begin() + (context.size() - max_seq_len));
        }

        size_t B = 1;
        size_t T = context.size();
        std::vector<int> dummy_Y(T, 0);
        
        this->forward(context, dummy_Y, B, T);

        size_t last_token_row = T - 1;
        std::vector<float> last_token_logits(vocab_size);
        for (size_t v = 0; v < vocab_size; ++v) {
            last_token_logits[v] = logits(last_token_row, v);
        }

        float max_logit = *std::max_element(last_token_logits.begin(), last_token_logits.end());
        std::vector<float> probs_local(vocab_size);
        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; ++v) {
            probs_local[v] = std::exp(last_token_logits[v] - max_logit);
            sum_exp += probs_local[v];
        }
        for (size_t v = 0; v < vocab_size; ++v) {
            probs_local[v] /= sum_exp;
        }

        std::discrete_distribution<int> dist(probs_local.begin(), probs_local.end());
        int next_token = dist(rng);
        generated.push_back(next_token);
    }
    return generated;
}
