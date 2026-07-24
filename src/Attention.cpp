#include "Attention.hpp"
#include "Ops.hpp"
#include <random>

MultiHeadAttention::MultiHeadAttention(size_t C_dim, size_t num_heads)
    : C_dim(C_dim), num_heads(num_heads), head_dim(C_dim / num_heads) {
    W_qkv = Parameter({C_dim, 3 * C_dim});
    b_qkv = Parameter({3 * C_dim});
    W_proj = Parameter({C_dim, C_dim});
    b_proj = Parameter({C_dim});

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    float* w_qkv_ptr = W_qkv.value.data();
    for (size_t i = 0; i < W_qkv.value.size(); ++i) {
        w_qkv_ptr[i] = dist(rng);
    }
    float* w_proj_ptr = W_proj.value.data();
    for (size_t i = 0; i < W_proj.value.size(); ++i) {
        w_proj_ptr[i] = dist(rng);
    }
}

void MultiHeadAttention::forward(const NastyTensors& x, NastyTensors& y, size_t B, size_t T) {
    size_t M = B * T;
    
    if (qkv_act.shape() != std::vector<size_t>{M, 3 * C_dim}) {
        qkv_act = NastyTensors({M, 3 * C_dim});
        Q = NastyTensors({B, num_heads, T, head_dim});
        K = NastyTensors({B, num_heads, T, head_dim});
        V = NastyTensors({B, num_heads, T, head_dim});
        S = NastyTensors({B, num_heads, T, T});
        O_heads = NastyTensors({B, num_heads, T, head_dim});
        O_flat = NastyTensors({M, C_dim});
    }

    ops::linear_forward(x, W_qkv.value, b_qkv.value, qkv_act);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t m = b * T + t;
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t c = 0; c < head_dim; ++c) {
                    Q(b, h, t, c) = qkv_act(m, h * head_dim + c);
                    K(b, h, t, c) = qkv_act(m, C_dim + h * head_dim + c);
                    V(b, h, t, c) = qkv_act(m, 2 * C_dim + h * head_dim + c);
                }
            }
        }
    }

    ops::matmul_batched_Q_KT(Q, K, S);
    ops::causal_softmax_forward(S, S);
    ops::matmul_batched_S_V(S, V, O_heads);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t m = b * T + t;
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t c = 0; c < head_dim; ++c) {
                    O_flat(m, h * head_dim + c) = O_heads(b, h, t, c);
                }
            }
        }
    }

    ops::linear_forward(O_flat, W_proj.value, b_proj.value, y);
}

void MultiHeadAttention::backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx, size_t B, size_t T) {
    size_t M = B * T;

    if (d_O_flat.shape() != std::vector<size_t>{M, C_dim}) {
        d_O_flat = NastyTensors({M, C_dim}, 0.0f);
        dO_heads = NastyTensors({B, num_heads, T, head_dim}, 0.0f);
        dS = NastyTensors({B, num_heads, T, T}, 0.0f);
        dS_raw = NastyTensors({B, num_heads, T, T}, 0.0f);
        dQ = NastyTensors({B, num_heads, T, head_dim}, 0.0f);
        dK = NastyTensors({B, num_heads, T, head_dim}, 0.0f);
        dV = NastyTensors({B, num_heads, T, head_dim}, 0.0f);
        d_qkv_act = NastyTensors({M, 3 * C_dim}, 0.0f);
    } else {
        float* p1 = d_O_flat.data(); for (size_t i = 0; i < d_O_flat.size(); ++i) p1[i] = 0.0f;
        float* p2 = dO_heads.data(); for (size_t i = 0; i < dO_heads.size(); ++i) p2[i] = 0.0f;
        float* p3 = dS.data();       for (size_t i = 0; i < dS.size(); ++i) p3[i] = 0.0f;
        float* p4 = dS_raw.data();   for (size_t i = 0; i < dS_raw.size(); ++i) p4[i] = 0.0f;
        float* p5 = dQ.data();       for (size_t i = 0; i < dQ.size(); ++i) p5[i] = 0.0f;
        float* p6 = dK.data();       for (size_t i = 0; i < dK.size(); ++i) p6[i] = 0.0f;
        float* p7 = dV.data();       for (size_t i = 0; i < dV.size(); ++i) p7[i] = 0.0f;
        float* p8 = d_qkv_act.data(); for (size_t i = 0; i < d_qkv_act.size(); ++i) p8[i] = 0.0f;
    }

    ops::linear_backward(dy, O_flat, W_proj.value, d_O_flat, W_proj.grad, b_proj.grad);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t m = b * T + t;
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t c = 0; c < head_dim; ++c) {
                    dO_heads(b, h, t, c) = d_O_flat(m, h * head_dim + c);
                }
            }
        }
    }

    ops::matmul_batched_S_V_backward(dO_heads, S, V, dS, dV);
    ops::causal_softmax_backward(dS, S, dS_raw);
    ops::matmul_batched_Q_KT_backward(dS_raw, Q, K, dQ, dK);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t m = b * T + t;
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t c = 0; c < head_dim; ++c) {
                    d_qkv_act(m, h * head_dim + c) = dQ(b, h, t, c);
                    d_qkv_act(m, C_dim + h * head_dim + c) = dK(b, h, t, c);
                    d_qkv_act(m, 2 * C_dim + h * head_dim + c) = dV(b, h, t, c);
                }
            }
        }
    }

    ops::linear_backward(d_qkv_act, x, W_qkv.value, dx, W_qkv.grad, b_qkv.grad);
}

std::vector<Parameter*> MultiHeadAttention::get_parameters() {
    return { &W_qkv, &b_qkv, &W_proj, &b_proj };
}
