#include "MLP.hpp"
#include "Ops.hpp"
#include <random>

MLP::MLP(size_t C_dim) : C_dim(C_dim) {
    W_fc = Parameter({C_dim, 4 * C_dim});
    b_fc = Parameter({4 * C_dim});
    W_proj = Parameter({4 * C_dim, C_dim});
    b_proj = Parameter({C_dim});

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    float* w_fc_ptr = W_fc.value.data();
    for (size_t i = 0; i < W_fc.value.size(); ++i) {
        w_fc_ptr[i] = dist(rng);
    }
    float* w_proj_ptr = W_proj.value.data();
    for (size_t i = 0; i < W_proj.value.size(); ++i) {
        w_proj_ptr[i] = dist(rng);
    }
}

void MLP::forward(const NastyTensors& x, NastyTensors& y) {
    size_t M = x.shape()[0];
    
    if (fc_act.shape() != std::vector<size_t>{M, 4 * C_dim}) {
        fc_act = NastyTensors({M, 4 * C_dim});
        gelu_act = NastyTensors({M, 4 * C_dim});
    }

    ops::linear_forward(x, W_fc.value, b_fc.value, fc_act);
    ops::gelu_forward(fc_act, gelu_act);
    ops::linear_forward(gelu_act, W_proj.value, b_proj.value, y);
}

void MLP::backward(const NastyTensors& dy, const NastyTensors& x, NastyTensors& dx) {
    size_t M = x.shape()[0];

    if (d_gelu_act.shape() != std::vector<size_t>{M, 4 * C_dim}) {
        d_gelu_act = NastyTensors({M, 4 * C_dim}, 0.0f);
        d_fc_act = NastyTensors({M, 4 * C_dim}, 0.0f);
    } else {
        float* p1 = d_gelu_act.data(); for (size_t i = 0; i < d_gelu_act.size(); ++i) p1[i] = 0.0f;
        float* p2 = d_fc_act.data();   for (size_t i = 0; i < d_fc_act.size(); ++i) p2[i] = 0.0f;
    }

    ops::linear_backward(dy, gelu_act, W_proj.value, d_gelu_act, W_proj.grad, b_proj.grad);
    ops::gelu_backward(d_gelu_act, fc_act, d_fc_act);
    ops::linear_backward(d_fc_act, x, W_fc.value, dx, W_fc.grad, b_fc.grad);
}

std::vector<Parameter*> MLP::get_parameters() {
    return { &W_fc, &b_fc, &W_proj, &b_proj };
}
