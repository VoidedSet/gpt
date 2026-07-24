#pragma once

#include "NastyTensors.hpp"
#include <vector>

namespace ops {
    void matmul(const NastyTensors& A, const NastyTensors& B, NastyTensors& C);
    
    void layernorm_forward(const NastyTensors& x, const NastyTensors& gamma, const NastyTensors& beta, 
                           NastyTensors& y, NastyTensors& mean, NastyTensors& rstd, float eps = 1e-5f);
    
    void layernorm_backward(const NastyTensors& dy, const NastyTensors& x, const NastyTensors& gamma,
                            const NastyTensors& mean, const NastyTensors& rstd, 
                            NastyTensors& dx, NastyTensors& dgamma, NastyTensors& dbeta);

    void gelu_forward(const NastyTensors& x, NastyTensors& y);
    
    void gelu_backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx);

    void causal_softmax_forward(const NastyTensors& x, NastyTensors& y);
    
    void causal_softmax_backward(const NastyTensors& dy, const NastyTensors& y, NastyTensors& dx);

    void linear_forward(const NastyTensors& X, const NastyTensors& W, const NastyTensors& b, NastyTensors& Y);
    
    void linear_backward(const NastyTensors& dY, const NastyTensors& X, const NastyTensors& W,
                         NastyTensors& dX, NastyTensors& dW, NastyTensors& db);

    void matmul_batched_Q_KT(const NastyTensors& Q, const NastyTensors& K, NastyTensors& S);
    
    void matmul_batched_Q_KT_backward(const NastyTensors& dS, const NastyTensors& Q, const NastyTensors& K,
                                      NastyTensors& dQ, NastyTensors& dK);

    void matmul_batched_S_V(const NastyTensors& S, const NastyTensors& V, NastyTensors& O);
    
    void matmul_batched_S_V_backward(const NastyTensors& dO, const NastyTensors& S, const NastyTensors& V,
                                     NastyTensors& dS, NastyTensors& dV);

    void add_forward(const NastyTensors& x1, const NastyTensors& x2, NastyTensors& y);
    
    void add_backward(const NastyTensors& dy, NastyTensors& dx1, NastyTensors& dx2);

    float cross_entropy_forward(const NastyTensors& logits, const std::vector<int>& targets, NastyTensors& probs);
    
    void cross_entropy_backward(const NastyTensors& probs, const std::vector<int>& targets, NastyTensors& dlogits);
}
