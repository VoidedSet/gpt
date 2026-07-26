#include "FeedForward.hpp"
#include <random>
#include <cmath>
#include <cassert>

FeedForward::FeedForward(size_t embedding_dim)
    : embedding_dim_(embedding_dim),
      w_fc_({embedding_dim, 4 * embedding_dim}),
      b_fc_({4 * embedding_dim}, 0.0f),
      w_proj_({4 * embedding_dim, embedding_dim}),
      b_proj_({embedding_dim}, 0.0f) {
    w_fc_.init_grad();
    b_fc_.init_grad();
    w_proj_.init_grad();
    b_proj_.init_grad();
    init_weights();
}

void FeedForward::init_weights() {
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    float* w_fc_ptr = w_fc_.data();
    size_t w_fc_size = w_fc_.size();
    for (size_t i = 0; i < w_fc_size; ++i) {
        w_fc_ptr[i] = dist(gen);
    }

    float* w_proj_ptr = w_proj_.data();
    size_t w_proj_size = w_proj_.size();
    for (size_t i = 0; i < w_proj_size; ++i) {
        w_proj_ptr[i] = dist(gen);
    }
}

NastyTensors FeedForward::forward(const NastyTensors& X) const {
    assert(X.ndim() == 3);
    assert(X.shape()[2] == embedding_dim_);

    size_t B = X.shape()[0];
    size_t T = X.shape()[1];
    size_t C = X.shape()[2];

    x_2d_ = X.reshape({B * T, C}).clone();

    h1_2d_ = x_2d_.matmul(w_fc_);
    float* h1_ptr = h1_2d_.data();
    const float* b_fc_ptr = b_fc_.data();
    size_t rows = B * T;
    size_t cols_4c = 4 * C;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t col = 0; col < cols_4c; ++col) {
            h1_ptr[r * cols_4c + col] += b_fc_ptr[col];
        }
    }

    h2_2d_ = h1_2d_.clone();
    h2_2d_.gelu();

    NastyTensors y_2d = h2_2d_.matmul(w_proj_);
    float* y_ptr = y_2d.data();
    const float* b_proj_ptr = b_proj_.data();
    for (size_t r = 0; r < rows; ++r) {
        for (size_t col = 0; col < C; ++col) {
            y_ptr[r * C + col] += b_proj_ptr[col];
        }
    }

    return y_2d.reshape({B, T, C});
}

void FeedForward::backward(const NastyTensors& dY, NastyTensors& dX) {
    assert(dY.ndim() == 3);
    size_t B = dY.shape()[0];
    size_t T = dY.shape()[1];
    size_t C = dY.shape()[2];
    assert(C == embedding_dim_);

    NastyTensors dY_2d = dY.reshape({B * T, C});

    NastyTensors dh2_2d = dY_2d.matmul_transposed_b(w_proj_);

    float* dw_proj = w_proj_.grad();
    float* db_proj = b_proj_.grad();
    for (size_t i = 0; i < B * T; ++i) {
        for (size_t r = 0; r < 4 * C; ++r) {
            float h2_val = h2_2d_(i, r);
            for (size_t c = 0; c < C; ++c) {
                dw_proj[r * C + c] += h2_val * dY_2d(i, c);
            }
        }
        for (size_t c = 0; c < C; ++c) {
            db_proj[c] += dY_2d(i, c);
        }
    }

    NastyTensors dh1_2d({B * T, 4 * C});
    float* dh1_ptr = dh1_2d.data();
    const float* dh2_ptr = dh2_2d.data();
    const float* h1_ptr = h1_2d_.data();
    size_t size_4c = B * T * 4 * C;

    for (size_t i = 0; i < size_4c; ++i) {
        float x = h1_ptr[i];
        float x3 = x * x * x;
        float a = 0.79788456f * (x + 0.044715f * x3);
        float t = std::tanh(a);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.79788456f * (1.0f + 0.134145f * x * x);
        dh1_ptr[i] = dh2_ptr[i] * dgelu;
    }

    NastyTensors dX_2d = dh1_2d.matmul_transposed_b(w_fc_);

    float* dw_fc = w_fc_.grad();
    float* db_fc = b_fc_.grad();
    for (size_t i = 0; i < B * T; ++i) {
        for (size_t r = 0; r < C; ++r) {
            float x_val = x_2d_(i, r);
            for (size_t c = 0; c < 4 * C; ++c) {
                dw_fc[r * (4 * C) + c] += x_val * dh1_2d(i, c);
            }
        }
        for (size_t c = 0; c < 4 * C; ++c) {
            db_fc[c] += dh1_2d(i, c);
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
