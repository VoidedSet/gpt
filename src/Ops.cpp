#include "Ops.hpp"
#include <cmath>
#include <stdexcept>
#include <immintrin.h>

namespace ops {
    void matmul(const NastyTensors& A, const NastyTensors& B, NastyTensors& C) {
        if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
            throw std::runtime_error("matmul: Tensors must be 2D.");
        }
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t K2 = B.shape()[0];
        size_t N = B.shape()[1];
        if (K != K2) {
            throw std::runtime_error("matmul: Inner dimensions do not match.");
        }
        if (C.shape()[0] != M || C.shape()[1] != N) {
            throw std::runtime_error("matmul: Output tensor shape does not match M x N.");
        }
        float* C_data = C.data();
        size_t C_size = C.size();
        for (size_t i = 0; i < C_size; ++i) {
            C_data[i] = 0.0f;
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float val = A(i, k);
                __m256 val_vec = _mm256_set1_ps(val);
                size_t j = 0;
                for (; j + 7 < N; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&B(k, j));
                    __m256 c_vec = _mm256_loadu_ps(&C(i, j));
                    __m256 res = _mm256_fmadd_ps(val_vec, b_vec, c_vec);
                    _mm256_storeu_ps(&C(i, j), res);
                }
                for (; j < N; ++j) {
                    C(i, j) += val * B(k, j);
                }
            }
        }
    }

    void layernorm_forward(const NastyTensors& x, const NastyTensors& gamma, const NastyTensors& beta, 
                           NastyTensors& y, NastyTensors& mean, NastyTensors& rstd, float eps) {
        size_t M = x.shape()[0];
        size_t C = x.shape()[1];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                sum += x(i, j);
            }
            float m = sum / C;
            mean(i) = m;

            float var_sum = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                float diff = x(i, j) - m;
                var_sum += diff * diff;
            }
            float r = 1.0f / std::sqrt(var_sum / C + eps);
            rstd(i) = r;

            for (size_t j = 0; j < C; ++j) {
                float x_hat = (x(i, j) - m) * r;
                y(i, j) = x_hat * gamma(j) + beta(j);
            }
        }
    }

    void layernorm_backward(const NastyTensors& dy, const NastyTensors& x, const NastyTensors& gamma,
                            const NastyTensors& mean, const NastyTensors& rstd, 
                            NastyTensors& dx, NastyTensors& dgamma, NastyTensors& dbeta) {
        size_t M = x.shape()[0];
        size_t C = x.shape()[1];
        for (size_t i = 0; i < M; ++i) {
            float m = mean(i);
            float r = rstd(i);
            float sum_dy_gamma = 0.0f;
            float sum_dy_gamma_xhat = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                float x_hat = (x(i, j) - m) * r;
                float dy_j = dy(i, j);
                dgamma(j) += dy_j * x_hat;
                dbeta(j) += dy_j;
                float dy_gamma = dy_j * gamma(j);
                sum_dy_gamma += dy_gamma;
                sum_dy_gamma_xhat += dy_gamma * x_hat;
            }
            for (size_t j = 0; j < C; ++j) {
                float x_hat = (x(i, j) - m) * r;
                float dy_gamma = dy(i, j) * gamma(j);
                dx(i, j) += (r / C) * (C * dy_gamma - sum_dy_gamma - x_hat * sum_dy_gamma_xhat);
            }
        }
    }

    void gelu_forward(const NastyTensors& x, NastyTensors& y) {
        size_t size = x.size();
        const float* x_data = x.data();
        float* y_data = y.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; ++i) {
            float val = x_data[i];
            float fit = 0.7978845608f * (val + 0.044715f * val * val * val);
            float g = std::tanh(fit);
            y_data[i] = 0.5f * val * (1.0f + g);
        }
    }

    void gelu_backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx) {
        size_t size = x.size();
        const float* x_data = x.data();
        const float* dy_data = dy.data();
        float* dx_data = dx.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; ++i) {
            float val = x_data[i];
            float fit = 0.7978845608f * (val + 0.044715f * val * val * val);
            float g = std::tanh(fit);
            float sech2 = 1.0f - g * g;
            float deriv = 0.7978845608f * (1.0f + 3.0f * 0.044715f * val * val);
            dx_data[i] += dy_data[i] * (0.5f * (1.0f + g) + 0.5f * val * sech2 * deriv);
        }
    }

    void causal_softmax_forward(const NastyTensors& x, NastyTensors& y) {
        size_t B = x.shape()[0];
        size_t h = x.shape()[1];
        size_t T = x.shape()[2];

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < B; ++b) {
            for (size_t head = 0; head < h; ++head) {
                for (size_t t1 = 0; t1 < T; ++t1) {
                    float max_val = x(b, head, t1, 0);
                    for (size_t t2 = 1; t2 <= t1; ++t2) {
                        if (x(b, head, t1, t2) > max_val) {
                            max_val = x(b, head, t1, t2);
                        }
                    }
                    float sum_exp = 0.0f;
                    for (size_t t2 = 0; t2 <= t1; ++t2) {
                        float e = std::exp(x(b, head, t1, t2) - max_val);
                        y(b, head, t1, t2) = e;
                        sum_exp += e;
                    }
                    for (size_t t2 = 0; t2 <= t1; ++t2) {
                        y(b, head, t1, t2) /= sum_exp;
                    }
                    for (size_t t2 = t1 + 1; t2 < T; ++t2) {
                        y(b, head, t1, t2) = 0.0f;
                    }
                }
            }
        }
    }

    void causal_softmax_backward(const NastyTensors& dy, const NastyTensors& y, NastyTensors& dx) {
        size_t B = y.shape()[0];
        size_t h = y.shape()[1];
        size_t T = y.shape()[2];

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < B; ++b) {
            for (size_t head = 0; head < h; ++head) {
                for (size_t t1 = 0; t1 < T; ++t1) {
                    float sum_dy_y = 0.0f;
                    for (size_t t2 = 0; t2 <= t1; ++t2) {
                        sum_dy_y += dy(b, head, t1, t2) * y(b, head, t1, t2);
                    }
                    for (size_t t2 = 0; t2 <= t1; ++t2) {
                        float prob = y(b, head, t1, t2);
                        dx(b, head, t1, t2) += prob * (dy(b, head, t1, t2) - sum_dy_y);
                    }
                    for (size_t t2 = t1 + 1; t2 < T; ++t2) {
                        dx(b, head, t1, t2) = 0.0f;
                    }
                }
            }
        }
    }

    void linear_forward(const NastyTensors& X, const NastyTensors& W, const NastyTensors& b, NastyTensors& Y) {
        matmul(X, W, Y);
        size_t M = Y.shape()[0];
        size_t N = Y.shape()[1];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                Y(i, j) += b(j);
            }
        }
    }

    void linear_backward(const NastyTensors& dY, const NastyTensors& X, const NastyTensors& W,
                         NastyTensors& dX, NastyTensors& dW, NastyTensors& db) {
        size_t M = dY.shape()[0];
        size_t N = dY.shape()[1];
        size_t K = W.shape()[0];

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < M; ++i) {
                sum += dY(i, j);
            }
            db(j) += sum;
        }

        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < K; ++k) {
            for (size_t i = 0; i < M; ++i) {
                float x_val = X(i, k);
                __m256 xv_vec = _mm256_set1_ps(x_val);
                size_t j = 0;
                for (; j + 7 < N; j += 8) {
                    __m256 dy_vec = _mm256_loadu_ps(&dY(i, j));
                    __m256 dw_vec = _mm256_loadu_ps(&dW(k, j));
                    __m256 res = _mm256_fmadd_ps(xv_vec, dy_vec, dw_vec);
                    _mm256_storeu_ps(&dW(k, j), res);
                }
                for (; j < N; ++j) {
                    dW(k, j) += x_val * dY(i, j);
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float sum = 0.0f;
                __m256 sum_vec = _mm256_setzero_ps();
                size_t j = 0;
                for (; j + 7 < N; j += 8) {
                    __m256 dy_vec = _mm256_loadu_ps(&dY(i, j));
                    __m256 w_vec = _mm256_loadu_ps(&W(k, j));
                    sum_vec = _mm256_fmadd_ps(dy_vec, w_vec, sum_vec);
                }
                float temp[8];
                _mm256_storeu_ps(temp, sum_vec);
                for (int idx = 0; idx < 8; ++idx) sum += temp[idx];
                for (; j < N; ++j) {
                    sum += dY(i, j) * W(k, j);
                }
                dX(i, k) += sum;
            }
        }
    }

    void matmul_batched_Q_KT(const NastyTensors& Q, const NastyTensors& K, NastyTensors& S) {
        size_t B = Q.shape()[0];
        size_t h = Q.shape()[1];
        size_t T = Q.shape()[2];
        size_t d = Q.shape()[3];
        float scale = 1.0f / std::sqrt(static_cast<float>(d));

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < B; ++b) {
            for (size_t head = 0; head < h; ++head) {
                for (size_t t1 = 0; t1 < T; ++t1) {
                    for (size_t t2 = 0; t2 < T; ++t2) {
                        float sum = 0.0f;
                        __m256 sum_vec = _mm256_setzero_ps();
                        size_t c = 0;
                        for (; c + 7 < d; c += 8) {
                            __m256 q_vec = _mm256_loadu_ps(&Q(b, head, t1, c));
                            __m256 k_vec = _mm256_loadu_ps(&K(b, head, t2, c));
                            sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                        }
                        float temp[8];
                        _mm256_storeu_ps(temp, sum_vec);
                        for (int idx = 0; idx < 8; ++idx) sum += temp[idx];
                        for (; c < d; ++c) {
                            sum += Q(b, head, t1, c) * K(b, head, t2, c);
                        }
                        S(b, head, t1, t2) = sum * scale;
                    }
                }
            }
        }
    }

    void matmul_batched_Q_KT_backward(const NastyTensors& dS, const NastyTensors& Q, const NastyTensors& K,
                                      NastyTensors& dQ, NastyTensors& dK) {
        size_t B = Q.shape()[0];
        size_t h = Q.shape()[1];
        size_t T = Q.shape()[2];
        size_t d = Q.shape()[3];
        float scale = 1.0f / std::sqrt(static_cast<float>(d));

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < B; ++b) {
            for (size_t head = 0; head < h; ++head) {
                for (size_t t1 = 0; t1 < T; ++t1) {
                    for (size_t t2 = 0; t2 < T; ++t2) {
                        float ds_val = dS(b, head, t1, t2) * scale;
                        __m256 ds_vec = _mm256_set1_ps(ds_val);
                        size_t c = 0;
                        for (; c + 7 < d; c += 8) {
                            __m256 k_vec = _mm256_loadu_ps(&K(b, head, t2, c));
                            __m256 dq_vec = _mm256_loadu_ps(&dQ(b, head, t1, c));
                            __m256 res_q = _mm256_fmadd_ps(ds_vec, k_vec, dq_vec);
                            _mm256_storeu_ps(&dQ(b, head, t1, c), res_q);

                            __m256 q_vec = _mm256_loadu_ps(&Q(b, head, t1, c));
                            __m256 dk_vec = _mm256_loadu_ps(&dK(b, head, t2, c));
                            __m256 res_k = _mm256_fmadd_ps(ds_vec, q_vec, dk_vec);
                            _mm256_storeu_ps(&dK(b, head, t2, c), res_k);
                        }
                        for (; c < d; ++c) {
                            dQ(b, head, t1, c) += ds_val * K(b, head, t2, c);
                            dK(b, head, t2, c) += ds_val * Q(b, head, t1, c);
                        }
                    }
                }
            }
        }
    }

    void matmul_batched_S_V(const NastyTensors& S, const NastyTensors& V, NastyTensors& O) {
        size_t B = S.shape()[0];
        size_t h = S.shape()[1];
        size_t T = S.shape()[2];
        size_t d = V.shape()[3];

        float* O_data = O.data();
        size_t O_size = O.size();
        for (size_t i = 0; i < O_size; ++i) {
            O_data[i] = 0.0f;
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < B; ++b) {
            for (size_t head = 0; head < h; ++head) {
                for (size_t t1 = 0; t1 < T; ++t1) {
                    for (size_t t2 = 0; t2 < T; ++t2) {
                        float s_val = S(b, head, t1, t2);
                        __m256 s_vec = _mm256_set1_ps(s_val);
                        size_t c = 0;
                        for (; c + 7 < d; c += 8) {
                            __m256 v_vec = _mm256_loadu_ps(&V(b, head, t2, c));
                            __m256 o_vec = _mm256_loadu_ps(&O(b, head, t1, c));
                            __m256 res = _mm256_fmadd_ps(s_vec, v_vec, o_vec);
                            _mm256_storeu_ps(&O(b, head, t1, c), res);
                        }
                        for (; c < d; ++c) {
                            O(b, head, t1, c) += s_val * V(b, head, t2, c);
                        }
                    }
                }
            }
        }
    }

    void matmul_batched_S_V_backward(const NastyTensors& dO, const NastyTensors& S, const NastyTensors& V,
                                     NastyTensors& dS, NastyTensors& dV) {
        size_t B = S.shape()[0];
        size_t h = S.shape()[1];
        size_t T = S.shape()[2];
        size_t d = V.shape()[3];

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < B; ++b) {
            for (size_t head = 0; head < h; ++head) {
                for (size_t t1 = 0; t1 < T; ++t1) {
                    for (size_t t2 = 0; t2 < T; ++t2) {
                        float s_val = S(b, head, t1, t2);
                        __m256 s_vec = _mm256_set1_ps(s_val);
                        float sum = 0.0f;
                        __m256 sum_vec = _mm256_setzero_ps();
                        size_t c = 0;
                        for (; c + 7 < d; c += 8) {
                            __m256 do_vec = _mm256_loadu_ps(&dO(b, head, t1, c));
                            __m256 v_vec = _mm256_loadu_ps(&V(b, head, t2, c));
                            sum_vec = _mm256_fmadd_ps(do_vec, v_vec, sum_vec);

                            __m256 dv_vec = _mm256_loadu_ps(&dV(b, head, t2, c));
                            __m256 res_v = _mm256_fmadd_ps(do_vec, s_vec, dv_vec);
                            _mm256_storeu_ps(&dV(b, head, t2, c), res_v);
                        }
                        float temp[8];
                        _mm256_storeu_ps(temp, sum_vec);
                        for (int idx = 0; idx < 8; ++idx) sum += temp[idx];
                        for (; c < d; ++c) {
                            sum += dO(b, head, t1, c) * V(b, head, t2, c);
                            dV(b, head, t2, c) += dO(b, head, t1, c) * s_val;
                        }
                        dS(b, head, t1, t2) += sum;
                    }
                }
            }
        }
    }

    void add_forward(const NastyTensors& x1, const NastyTensors& x2, NastyTensors& y) {
        size_t size = x1.size();
        const float* x1_data = x1.data();
        const float* x2_data = x2.data();
        float* y_data = y.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; ++i) {
            y_data[i] = x1_data[i] + x2_data[i];
        }
    }

    void add_backward(const NastyTensors& dy, NastyTensors& dx1, NastyTensors& dx2) {
        size_t size = dy.size();
        const float* dy_data = dy.data();
        float* dx1_data = dx1.data();
        float* dx2_data = dx2.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; ++i) {
            dx1_data[i] += dy_data[i];
            dx2_data[i] += dy_data[i];
        }
    }

    float cross_entropy_forward(const NastyTensors& logits, const std::vector<int>& targets, NastyTensors& probs) {
        size_t N = logits.shape()[0];
        size_t V = logits.shape()[1];
        float total_loss = 0.0f;

        #pragma omp parallel for reduction(+:total_loss) schedule(static)
        for (size_t i = 0; i < N; ++i) {
            float max_val = logits(i, 0);
            for (size_t j = 1; j < V; ++j) {
                if (logits(i, j) > max_val) max_val = logits(i, j);
            }
            float sum_exp = 0.0f;
            for (size_t j = 0; j < V; ++j) {
                float e = std::exp(logits(i, j) - max_val);
                probs(i, j) = e;
                sum_exp += e;
            }
            for (size_t j = 0; j < V; ++j) {
                probs(i, j) /= sum_exp;
            }
            int target = targets[i];
            total_loss += -std::log(probs(i, target) + 1e-9f);
        }
        return total_loss / N;
    }

    void cross_entropy_backward(const NastyTensors& probs, const std::vector<int>& targets, NastyTensors& dlogits) {
        size_t N = probs.shape()[0];
        size_t V = probs.shape()[1];

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            int target = targets[i];
            for (size_t j = 0; j < V; ++j) {
                dlogits(i, j) += (probs(i, j) - (j == static_cast<size_t>(target) ? 1.0f : 0.0f)) / N;
            }
        }
    }
}
