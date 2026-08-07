#include "Block.hpp"
#include <cuda_runtime.h>

Block::Block(size_t embedding_dim, size_t num_heads)
    : ln1_(embedding_dim),
      attn_(embedding_dim, num_heads),
      ln2_(embedding_dim),
      mlp_(embedding_dim) {}

NastyTensors Block::forward(const NastyTensors& X) const {
    NastyTensors norm_X = X.clone();
    ln1_.forward(norm_X);
    NastyTensors attn_out = attn_.forward(norm_X);
    attn_out += X;

    NastyTensors norm_X1 = attn_out.clone();
    ln2_.forward(norm_X1);
    NastyTensors mlp_out = mlp_.forward(norm_X1);
    mlp_out += attn_out;

    return mlp_out;
}

void Block::backward(const NastyTensors& dY, NastyTensors& dX) {
    size_t B = dY.shape()[0];
    size_t T = dY.shape()[1];
    size_t C = dY.shape()[2];

    NastyTensors dMLP_in({B, T, C});
    if (dY.device_data() != nullptr) {
        dMLP_in.to_gpu(DATA_);
    }
    mlp_.backward(dY, dMLP_in);

    NastyTensors dX_mid_ln({B, T, C});
    if (dY.device_data() != nullptr) {
        dX_mid_ln.to_gpu(DATA_);
    }
    ln2_.backward(dMLP_in, dX_mid_ln);

    NastyTensors dX_mid = dY.clone();
    dX_mid += dX_mid_ln;

    NastyTensors dAttn_in({B, T, C});
    if (dY.device_data() != nullptr) {
        dAttn_in.to_gpu(DATA_);
    }
    attn_.backward(dX_mid, dAttn_in);

    NastyTensors dX_ln({B, T, C});
    if (dY.device_data() != nullptr) {
        dX_ln.to_gpu(DATA_);
    }
    ln1_.backward(dAttn_in, dX_ln);

    if (dY.device_data() != nullptr) {
        dX.to_gpu(DATA_);
    }

    size_t total_elements = dX.size();
    if (dX.device_data() != nullptr) {
        cudaMemcpy(dX.device_data(), dX_mid.device_data(), total_elements * sizeof(float), cudaMemcpyDeviceToDevice);
        dX += dX_ln;
        return;
    }

    float* dest = dX.data();
    const float* src_mid = dX_mid.data();
    const float* src_ln = dX_ln.data();
    if (dest && src_mid && src_ln) {
        for (size_t i = 0; i < total_elements; ++i) {
            dest[i] = src_mid[i] + src_ln[i];
        }
    }
}
