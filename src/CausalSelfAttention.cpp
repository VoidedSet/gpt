#include "CausalSelfAttention.hpp"
#include "CudaKernels.hpp"
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <cassert>

CausalSelfAttention::CausalSelfAttention(size_t embedding_dim, size_t num_heads)
    : embedding_dim_(embedding_dim),
      num_heads_(num_heads),
      head_dim_(embedding_dim / num_heads),
      w_qkv_({embedding_dim, 3 * embedding_dim}),
      b_qkv_({3 * embedding_dim}, 0.0f),
      w_proj_({embedding_dim, embedding_dim}),
      b_proj_({embedding_dim}, 0.0f) {
    w_qkv_.init_grad();
    b_qkv_.init_grad();
    w_proj_.init_grad();
    b_proj_.init_grad();
    init_weights();
}

void CausalSelfAttention::init_weights() {
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    float* w_qkv_ptr = w_qkv_.data();
    size_t w_qkv_size = w_qkv_.size();
    for (size_t i = 0; i < w_qkv_size; ++i) {
        w_qkv_ptr[i] = dist(gen);
    }

    float* w_proj_ptr = w_proj_.data();
    size_t w_proj_size = w_proj_.size();
    for (size_t i = 0; i < w_proj_size; ++i) {
        w_proj_ptr[i] = dist(gen);
    }
}

NastyTensors CausalSelfAttention::forward(const NastyTensors& X) const {
    assert(X.ndim() == 3);
    assert(X.shape()[2] == embedding_dim_);

    size_t B = X.shape()[0];
    size_t T = X.shape()[1];
    size_t C = X.shape()[2];
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    x_2d_ = X.reshape({B * T, C}).clone();

    if (w_qkv_.device_data() != nullptr) {
        x_2d_.to_gpu(DATA_);

        NastyTensors qkv_2d = x_2d_.matmul(w_qkv_);
        launch_add_bias(qkv_2d.device_data(), b_qkv_.device_data(), B * T, 3 * C);

        qkv_ = qkv_2d.reshape({B, T, 3 * C}).clone();
        NastyTensors O({B, T, C}, 0.0f);
        O.to_gpu(DATA_);

        att_probs_ = NastyTensors({B * num_heads_ * T * T}, 0.0f);
        att_probs_.to_gpu(DATA_);

        launch_attention_forward(
            qkv_.device_data(), 
            att_probs_.device_data(), 
            O.device_data(), 
            B, T, C, num_heads_, head_dim_, scale
        );

        o_ = O.clone();

        NastyTensors o_2d = O.reshape({B * T, C});
        NastyTensors proj_2d = o_2d.matmul(w_proj_);
        launch_add_bias(proj_2d.device_data(), b_proj_.device_data(), B * T, C);

        return proj_2d.reshape({B, T, C});
    }

    NastyTensors qkv_2d = x_2d_.matmul(w_qkv_);

    float* qkv_ptr = qkv_2d.data();
    const float* b_qkv_ptr = b_qkv_.data();
    size_t num_rows = B * T;
    size_t num_cols = 3 * C;
    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t col = 0; col < num_cols; ++col) {
            qkv_ptr[r * num_cols + col] += b_qkv_ptr[col];
        }
    }

    qkv_ = qkv_2d.reshape({B, T, 3 * C}).clone();
    NastyTensors O({B, T, C}, 0.0f);

    att_probs_ = NastyTensors({B * num_heads_ * T * T}, 0.0f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t head = 0; head < num_heads_; ++head) {
            for (size_t t_q = 0; t_q < T; ++t_q) {
                std::vector<float> scores(t_q + 1);

                for (size_t t_k = 0; t_k <= t_q; ++t_k) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        float q = qkv_(b, t_q, head * head_dim_ + d);
                        float k = qkv_(b, t_k, C + head * head_dim_ + d);
                        dot += q * k;
                    }
                    scores[t_k] = dot * scale;
                }

                float max_val = scores[0];
                for (size_t t_k = 1; t_k <= t_q; ++t_k) {
                    if (scores[t_k] > max_val) {
                        max_val = scores[t_k];
                    }
                }

                float sum_exp = 0.0f;
                for (size_t t_k = 0; t_k <= t_q; ++t_k) {
                    scores[t_k] = std::exp(scores[t_k] - max_val);
                    sum_exp += scores[t_k];
                }

                for (size_t t_k = 0; t_k <= t_q; ++t_k) {
                    scores[t_k] /= sum_exp;
                    att_probs_(b * num_heads_ * T * T + head * T * T + t_q * T + t_k) = scores[t_k];
                }

                for (size_t d = 0; d < head_dim_; ++d) {
                    float val_sum = 0.0f;
                    for (size_t t_k = 0; t_k <= t_q; ++t_k) {
                        float v = qkv_(b, t_k, 2 * C + head * head_dim_ + d);
                        val_sum += scores[t_k] * v;
                    }
                    O(b, t_q, head * head_dim_ + d) = val_sum;
                }
            }
        }
    }

    o_ = O.clone();

    NastyTensors o_2d = O.reshape({B * T, C});
    NastyTensors proj_2d = o_2d.matmul(w_proj_);

    float* proj_ptr = proj_2d.data();
    const float* b_proj_ptr = b_proj_.data();
    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t col = 0; col < C; ++col) {
            proj_ptr[r * C + col] += b_proj_ptr[col];
        }
    }

    return proj_2d.reshape({B, T, C});
}

void CausalSelfAttention::backward(const NastyTensors& dY, NastyTensors& dX) {
    assert(dY.ndim() == 3);
    size_t B = dY.shape()[0];
    size_t T = dY.shape()[1];
    size_t C = dY.shape()[2];
    assert(C == embedding_dim_);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    if (w_qkv_.device_data() != nullptr) {
        w_proj_.to_gpu(GRAD_);
        b_proj_.to_gpu(GRAD_);
        w_qkv_.to_gpu(GRAD_);
        b_qkv_.to_gpu(GRAD_);
        dX.to_gpu(DATA_);

        NastyTensors dY_2d = dY.reshape({B * T, C});
        NastyTensors dO_2d = dY_2d.matmul_transposed_b(w_proj_);
        NastyTensors dO = dO_2d.reshape({B, T, C});

        NastyTensors::gemm(true, false, 
                           C, C, B * T, 
                           1.0f, 
                           o_.device_data(), C, 
                           dY_2d.device_data(), C, 
                           1.0f, 
                           w_proj_.device_grad(), C);

        launch_accumulate_bias_grad(dY_2d.device_data(), b_proj_.device_grad(), B * T, C);

        NastyTensors dqkv({B, T, 3 * C}, 0.0f);
        dqkv.to_gpu(DATA_);

        launch_attention_backward(
            dO.device_data(), 
            qkv_.device_data(), 
            att_probs_.device_data(), 
            dqkv.device_data(), 
            B, T, C, num_heads_, head_dim_, scale
        );

        NastyTensors dqkv_2d = dqkv.reshape({B * T, 3 * C});
        NastyTensors dX_2d = dqkv_2d.matmul_transposed_b(w_qkv_);

        NastyTensors::gemm(true, false, 
                           C, 3 * C, B * T, 
                           1.0f, 
                           x_2d_.device_data(), C, 
                           dqkv_2d.device_data(), 3 * C, 
                           1.0f, 
                           w_qkv_.device_grad(), 3 * C);

        launch_accumulate_bias_grad(dqkv_2d.device_data(), b_qkv_.device_grad(), B * T, 3 * C);

        NastyTensors dX_reshaped = dX_2d.reshape({B, T, C});
        cudaMemcpy(dX.device_data(), dX_reshaped.device_data(), B * T * C * sizeof(float), cudaMemcpyDeviceToDevice);

        return;
    }

    NastyTensors dY_2d = dY.reshape({B * T, C});
    NastyTensors dO_2d = dY_2d.matmul_transposed_b(w_proj_);
    NastyTensors dO = dO_2d.reshape({B, T, C});

    float* dw_proj = w_proj_.grad();
    float* db_proj = b_proj_.grad();
    NastyTensors o_2d = o_.reshape({B * T, C});

    for (size_t i = 0; i < B * T; ++i) {
        for (size_t r = 0; r < C; ++r) {
            float o_val = o_2d(i, r);
            for (size_t c = 0; c < C; ++c) {
                dw_proj[r * C + c] += o_val * dY_2d(i, c);
            }
        }
        for (size_t c = 0; c < C; ++c) {
            db_proj[c] += dY_2d(i, c);
        }
    }

    NastyTensors dqkv({B, T, 3 * C}, 0.0f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t head = 0; head < num_heads_; ++head) {
            for (size_t t_q = 0; t_q < T; ++t_q) {
                size_t att_offset = b * num_heads_ * T * T + head * T * T + t_q * T;

                std::vector<float> dp_vec(t_q + 1, 0.0f);
                float sum_dp_p = 0.0f;
                for (size_t t_k = 0; t_k <= t_q; ++t_k) {
                    float dp = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        dp += dO(b, t_q, head * head_dim_ + d) * qkv_(b, t_k, 2 * C + head * head_dim_ + d);
                    }
                    dp_vec[t_k] = dp;
                    float p = att_probs_(att_offset + t_k);
                    sum_dp_p += dp * p;
                }

                for (size_t t_k = 0; t_k <= t_q; ++t_k) {
                    float p = att_probs_(att_offset + t_k);
                    float dS = p * (dp_vec[t_k] - sum_dp_p);
                    float dS_scaled = dS * scale;

                    for (size_t d = 0; d < head_dim_; ++d) {
                        dqkv(b, t_k, 2 * C + head * head_dim_ + d) += p * dO(b, t_q, head * head_dim_ + d);
                        dqkv(b, t_q, head * head_dim_ + d) += dS_scaled * qkv_(b, t_k, C + head * head_dim_ + d);
                        dqkv(b, t_k, C + head * head_dim_ + d) += dS_scaled * qkv_(b, t_q, head * head_dim_ + d);
                    }
                }
            }
        }
    }

    NastyTensors dqkv_2d = dqkv.reshape({B * T, 3 * C});
    NastyTensors dX_2d = dqkv_2d.matmul_transposed_b(w_qkv_);

    float* dw_qkv = w_qkv_.grad();
    float* db_qkv = b_qkv_.grad();
    for (size_t i = 0; i < B * T; ++i) {
        for (size_t r = 0; r < C; ++r) {
            float x_val = x_2d_(i, r);
            for (size_t c = 0; c < 3 * C; ++c) {
                dw_qkv[r * (3 * C) + c] += x_val * dqkv_2d(i, c);
            }
        }
        for (size_t c = 0; c < 3 * C; ++c) {
            db_qkv[c] += dqkv_2d(i, c);
        }
    }

    NastyTensors dX_reshaped = dX_2d.reshape({B, T, C});
    float* dest = dX.data();
    const float* src = dX_reshaped.data();
    size_t total_elements = dX.size();
    if (dest && src) {
        for (size_t i = 0; i < total_elements; ++i) {
            dest[i] = src[i];
        }
    }
}
