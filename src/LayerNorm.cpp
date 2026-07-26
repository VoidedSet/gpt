#include "LayerNorm.hpp"
#include <cmath>
#include <cassert>

LayerNorm::LayerNorm(size_t embedding_dim, float epsilon)
    : embedding_dim_(embedding_dim), epsilon_(epsilon),
      gamma_({embedding_dim}, 1.0f), beta_({embedding_dim}, 0.0f) {}

void LayerNorm::forward(NastyTensors& X) const {
    assert(X.ndim() == 3 && "Input Tensor must be [BxTxC]");
    assert(X.shape()[2] == embedding_dim_ && "Channel count mis-match.");

    size_t B = X.shape()[0];
    size_t T = X.shape()[1];
    size_t C = X.shape()[2];

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            float sum = 0.0f;
            for (size_t c = 0; c < C; ++c) {
                sum += X(b, t, c);
            }
            float mean = sum / C;

            float sum_sq_diff = 0.0f;
            for (size_t c = 0; c < C; ++c) {
                float diff = X(b, t, c) - mean;
                sum_sq_diff += diff * diff;
            }
            float variance = sum_sq_diff / C;

            float scale = 1.0f / std::sqrt(variance + epsilon_);
            for (size_t c = 0; c < C; ++c) {
                X(b, t, c) = ((X(b, t, c) - mean) * scale) * gamma_(c) + beta_(c);
            }
        }
    }
}
