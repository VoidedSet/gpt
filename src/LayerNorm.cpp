#include "LayerNorm.hpp"
#include "CudaKernels.hpp"

#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

LayerNorm::LayerNorm(size_t embedding_dim, float epsilon)
    : embedding_dim_(embedding_dim), epsilon_(epsilon),
      gamma_({embedding_dim}, 1.0f), beta_({embedding_dim}, 0.0f) {
    gamma_.init_grad();
    beta_.init_grad();
}

void LayerNorm::forward(NastyTensors& X) const {
    assert(X.ndim() == 3);
    assert(X.shape()[2] == embedding_dim_);

    size_t B = X.shape()[0];
    size_t T = X.shape()[1];
    size_t C = X.shape()[2];

    x_hat_ = NastyTensors({B, T, C});
    mean_ = NastyTensors({B, T});
    var_ = NastyTensors({B, T});
    
    if (gamma_.device_data() != nullptr) {

        x_hat_.to_gpu(DATA_);
        mean_.to_gpu(DATA_);
        var_.to_gpu(DATA_);
        X.to_gpu(DATA_);

        launch_layernorm_forward(
            X.device_data(), 
            gamma_.device_data(), 
            beta_.device_data(), 
            x_hat_.device_data(), 
            mean_.device_data(), 
            var_.device_data(), 
            B, T, C, epsilon_
        );
        return;
    }

    // x_hat_ = NastyTensors({B, T, C});
    // mean_ = NastyTensors({B, T});
    // var_ = NastyTensors({B, T});

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
            mean_(b, t) = mean;
            var_(b, t) = variance;

            for (size_t c = 0; c < C; ++c) {
                float xh = (X(b, t, c) - mean) * scale;
                x_hat_(b, t, c) = xh;
                X(b, t, c) = xh * gamma_(c) + beta_(c);
            }
        }
    }
}

void LayerNorm::backward(const NastyTensors& dY, NastyTensors& dX) {
    assert(dY.ndim() == 3);
    size_t B = dY.shape()[0];
    size_t T = dY.shape()[1];
    size_t C = dY.shape()[2];
    assert(C == embedding_dim_);

    if (gamma_.device_data() != nullptr) {
        dX.to_gpu(DATA_);
        gamma_.to_gpu(GRAD_);
        beta_.to_gpu(GRAD_);

        launch_layernorm_backward(
            dY.device_data(),
            x_hat_.device_data(),
            var_.device_data(),
            gamma_.device_data(),
            gamma_.device_grad(),
            beta_.device_grad(),
            dX.device_data(),
            B, T, C, epsilon_
        );



        return;
    }

    float* dgamma = gamma_.grad();
    float* dbeta = beta_.grad();

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            float variance = var_(b, t);
            float scale = 1.0f / std::sqrt(variance + epsilon_);

            float sum_dy_gamma = 0.0f;
            float sum_dy_gamma_xhat = 0.0f;

            for (size_t c = 0; c < C; ++c) {
                float dy = dY(b, t, c);
                float xh = x_hat_(b, t, c);

                dgamma[c] += dy * xh;
                dbeta[c] += dy;

                sum_dy_gamma += dy * gamma_(c);
                sum_dy_gamma_xhat += dy * gamma_(c) * xh;
            }

            for (size_t c = 0; c < C; ++c) {
                float dy = dY(b, t, c);
                float xh = x_hat_(b, t, c);

                float dx = (gamma_(c) * dy - (sum_dy_gamma + xh * sum_dy_gamma_xhat) / C) * scale;
                dX(b, t, c) = dx;
            }
        }
    }


}
