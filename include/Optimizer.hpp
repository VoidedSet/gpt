#pragma once

#include "NastyTensors.hpp"
#include "CudaKernels.hpp"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

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

    // GPU optimizer state pointers
    std::vector<float*> d_m_;
    std::vector<float*> d_v_;

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
        d_m_.resize(params_.size(), nullptr);
        d_v_.resize(params_.size(), nullptr);
        for (size_t i = 0; i < params_.size(); ++i) {
            size_t size = params_[i]->size();
            m_[i].resize(size, 0.0f);
            v_[i].resize(size, 0.0f);
        }
    }

    ~AdamW() {
        for (size_t i = 0; i < params_.size(); ++i) {
            if (d_m_[i]) cudaFree(d_m_[i]);
            if (d_v_[i]) cudaFree(d_v_[i]);
        }
    }

    void step() {
        t_++;
        float bias_correction1 = 1.0f - std::pow(beta1_, t_);
        float bias_correction2 = 1.0f - std::pow(beta2_, t_);

        for (size_t i = 0; i < params_.size(); ++i) {
            NastyTensors* p = params_[i];
            
            if (p->device_data() != nullptr) {
                // GPU Path
                size_t size = p->size();
                if (d_m_[i] == nullptr) {
                    cudaMalloc(&d_m_[i], size * sizeof(float));
                    cudaMemset(d_m_[i], 0, size * sizeof(float));
                }
                if (d_v_[i] == nullptr) {
                    cudaMalloc(&d_v_[i], size * sizeof(float));
                    cudaMemset(d_v_[i], 0, size * sizeof(float));
                }

                launch_adamw_step(
                    p->device_data(),
                    p->device_grad(),
                    d_m_[i],
                    d_v_[i],
                    size,
                    lr_,
                    beta1_,
                    beta2_,
                    eps_,
                    weight_decay_,
                    bias_correction1,
                    bias_correction2
                );
                continue;
            }

            // CPU Path
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

    void set_lr(float lr) {
        lr_ = lr;
    }
};
