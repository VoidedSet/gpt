#include "CausalSelfAttention.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// A helper function to compare two float arrays with a tiny tolerance.
bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[*] Starting Attention Unit Test...\n";

    // 1. Initialize CausalSelfAttention layer
    int B = 2;
    int T = 4;
    int C = 16;
    int num_heads = 2;
    CausalSelfAttention attn(C, num_heads);

    // Fill weights/biases with non-trivial values
    float* w_qkv = attn.w_qkv().data();
    float* b_qkv = attn.b_qkv().data();
    float* w_proj = attn.w_proj().data();
    float* b_proj = attn.b_proj().data();

    for (size_t i = 0; i < attn.w_qkv().size(); ++i) w_qkv[i] = std::sin(static_cast<float>(i)) * 0.1f;
    for (size_t i = 0; i < attn.b_qkv().size(); ++i) b_qkv[i] = static_cast<float>(i) * 0.05f;
    for (size_t i = 0; i < attn.w_proj().size(); ++i) w_proj[i] = std::cos(static_cast<float>(i)) * 0.1f;
    for (size_t i = 0; i < attn.b_proj().size(); ++i) b_proj[i] = static_cast<float>(i) * 0.02f;

    // 2. Generate random input tensor X
    NastyTensors X({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    float* x_ptr = X.data();
    for (size_t i = 0; i < X.size(); ++i) {
        x_ptr[i] = std::sin(static_cast<float>(i) * 0.5f) * 1.5f;
    }

    // ==========================================
    // Part 1: CPU Forward Pass (Ground Truth)
    // ==========================================
    NastyTensors Y_cpu = attn.forward(X);

    // ==========================================
    // Part 2: GPU Forward Pass
    // ==========================================
    attn.w_qkv().to_gpu(DATA_);
    attn.b_qkv().to_gpu(DATA_);
    attn.w_proj().to_gpu(DATA_);
    attn.b_proj().to_gpu(DATA_);

    NastyTensors X_gpu = X.clone();
    X_gpu.to_gpu(DATA_);

    // Run forward pass on GPU
    NastyTensors Y_gpu = attn.forward(X_gpu);

    // Copy back
    Y_gpu.to_cpu(DATA_);

    // ==========================================
    // Part 3: Forward Verification
    // ==========================================
    for (size_t i = 0; i < Y_cpu.size(); ++i) {
        if (!approx_equal(Y_cpu.data()[i], Y_gpu.data()[i])) {
            std::cerr << "[-] FORWARD PASS MISMATCH at index " << i 
                      << ": CPU=" << Y_cpu.data()[i] 
                      << ", GPU=" << Y_gpu.data()[i] << "\n";
            return 1;
        }
    }
    std::cout << "[+] Forward Pass Verification Successful!\n";

    // ==========================================
    // Part 4: Backward Pass Verification
    // ==========================================
    std::cout << "[*] Starting Backward Pass Verification...\n";

    // Create incoming gradients (dY)
    NastyTensors dY({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    float* dy_ptr = dY.data();
    for (size_t i = 0; i < dY.size(); ++i) {
        dy_ptr[i] = std::cos(static_cast<float>(i) * 0.3f) * 0.5f;
    }

    // Run CPU Backward
    NastyTensors dX_cpu({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    attn.backward(dY, dX_cpu);

    // Clone CPU gradients
    NastyTensors w_qkv_g_cpu = attn.w_qkv().clone();
    NastyTensors b_qkv_g_cpu = attn.b_qkv().clone();
    NastyTensors w_proj_g_cpu = attn.w_proj().clone();
    NastyTensors b_proj_g_cpu = attn.b_proj().clone();

    // Reset gradients
    attn.w_qkv().zero_grad();
    attn.b_qkv().zero_grad();
    attn.w_proj().zero_grad();
    attn.b_proj().zero_grad();

    // Run GPU Backward
    NastyTensors dY_gpu = dY.clone();
    dY_gpu.to_gpu(DATA_);

    NastyTensors dX_gpu({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    attn.backward(dY_gpu, dX_gpu);

    // Copy back
    dX_gpu.to_cpu(DATA_);
    attn.w_qkv().to_cpu(GRAD_);
    attn.b_qkv().to_cpu(GRAD_);
    attn.w_proj().to_cpu(GRAD_);
    attn.b_proj().to_cpu(GRAD_);

    // 1. Verify dX
    for (size_t i = 0; i < dX_cpu.size(); ++i) {
        if (!approx_equal(dX_cpu.data()[i], dX_gpu.data()[i])) {
            std::cerr << "[-] BACKWARD INPUT GRADIENT (dX) MISMATCH at index " << i 
                      << ": CPU=" << dX_cpu.data()[i] 
                      << ", GPU=" << dX_gpu.data()[i] << "\n";
            return 1;
        }
    }

    // 2. Verify w_qkv gradients
    for (size_t i = 0; i < attn.w_qkv().size(); ++i) {
        if (!approx_equal(w_qkv_g_cpu.grad()[i], attn.w_qkv().grad()[i])) {
            std::cerr << "[-] BACKWARD W_QKV GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << w_qkv_g_cpu.grad()[i] 
                      << ", GPU=" << attn.w_qkv().grad()[i] << "\n";
            return 1;
        }
    }

    // 3. Verify b_qkv gradients
    for (size_t i = 0; i < attn.b_qkv().size(); ++i) {
        if (!approx_equal(b_qkv_g_cpu.grad()[i], attn.b_qkv().grad()[i])) {
            std::cerr << "[-] BACKWARD B_QKV GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << b_qkv_g_cpu.grad()[i] 
                      << ", GPU=" << attn.b_qkv().grad()[i] << "\n";
            return 1;
        }
    }

    // 4. Verify w_proj gradients
    for (size_t i = 0; i < attn.w_proj().size(); ++i) {
        if (!approx_equal(w_proj_g_cpu.grad()[i], attn.w_proj().grad()[i])) {
            std::cerr << "[-] BACKWARD W_PROJ GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << w_proj_g_cpu.grad()[i] 
                      << ", GPU=" << attn.w_proj().grad()[i] << "\n";
            return 1;
        }
    }

    // 5. Verify b_proj gradients
    for (size_t i = 0; i < attn.b_proj().size(); ++i) {
        if (!approx_equal(b_proj_g_cpu.grad()[i], attn.b_proj().grad()[i])) {
            std::cerr << "[-] BACKWARD B_PROJ GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << b_proj_g_cpu.grad()[i] 
                      << ", GPU=" << attn.b_proj().grad()[i] << "\n";
            return 1;
        }
    }

    std::cout << "[+] Backward Pass Verification Successful!\n";
    std::cout << "[+] All Attention Unit Tests Passed!\n";

    return 0;
}
