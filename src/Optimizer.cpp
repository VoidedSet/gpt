#include "Optimizer.hpp"
#include <cmath>

AdamW::AdamW(float lr, float beta1, float beta2, float eps, float weight_decay)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), step(0) {}

void AdamW::update(const std::vector<Parameter*>& params) {
    if (m.size() != params.size()) {
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            size_t size = params[i]->value.size();
            m[i].assign(size, 0.0f);
            v[i].assign(size, 0.0f);
        }
    }

    step++;
    float bias_correction1 = 1.0f - std::pow(beta1, step);
    float bias_correction2 = 1.0f - std::pow(beta2, step);

    for (size_t i = 0; i < params.size(); ++i) {
        float* w = params[i]->value.data();
        const float* g = params[i]->grad.data();
        float* m_i = m[i].data();
        float* v_i = v[i].data();
        size_t size = params[i]->value.size();

        for (size_t k = 0; k < size; ++k) {
            float gk = g[k];
            m_i[k] = beta1 * m_i[k] + (1.0f - beta1) * gk;
            v_i[k] = beta2 * v_i[k] + (1.0f - beta2) * gk * gk;
            float m_hat = m_i[k] / bias_correction1;
            float v_hat = v_i[k] / bias_correction2;
            w[k] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * w[k]);
        }
    }
}

void AdamW::zero_grad(const std::vector<Parameter*>& params) {
    for (size_t i = 0; i < params.size(); ++i) {
        float* g = params[i]->grad.data();
        size_t size = params[i]->grad.size();
        for (size_t k = 0; k < size; ++k) {
            g[k] = 0.0f;
        }
    }
}
