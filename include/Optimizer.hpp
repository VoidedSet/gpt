#pragma once

#include "NastyTensors.hpp"
#include <vector>
#include <cmath>

class AdamW {
private:
    std::vector<NastyTensors*> params_;
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int t_;

    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;

public:
    AdamW(const std::vector<NastyTensors*>& params, 
          float lr = 1e-4f, 
          float beta1 = 0.9f, 
          float beta2 = 0.99f, 
          float eps = 1e-8f, 
          float weight_decay = 0.01f)
        : params_(params), 
          lr_(lr), 
          beta1_(beta1), 
          beta2_(beta2), 
          eps_(eps), 
          weight_decay_(weight_decay), 
          t_(0) {
        
        m_.resize(params_.size());
        v_.resize(params_.size());
        for (size_t i = 0; i < params_.size(); ++i) {
            size_t size = params_[i]->size();
            m_[i].resize(size, 0.0f);
            v_[i].resize(size, 0.0f);
        }
    }

    void step() {
        t_++;
        float bias_correction1 = 1.0f - std::pow(beta1_, t_);
        float bias_correction2 = 1.0f - std::pow(beta2_, t_);

        for (size_t i = 0; i < params_.size(); ++i) {
            NastyTensors* p = params_[i];
            float* w = p->data();
            const float* g = p->grad();
            if (!w || !g) continue;

            size_t size = p->size();
            for (size_t j = 0; j < size; ++j) {
                float grad = g[j];
                
                m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * grad;
                v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * grad * grad;
                
                float m_hat = m_[i][j] / bias_correction1;
                float v_hat = v_[i][j] / bias_correction2;

                w[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_) + weight_decay_ * w[j]);
            }
        }
    }

    void zero_grad() {
        for (NastyTensors* p : params_) {
            p->zero_grad();
        }
    }
};
