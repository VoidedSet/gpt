#include "FeedForward.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// A helper function to compare two float arrays with a tiny tolerance.
bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[*] Starting FeedForward Unit Test...\n";

    // 1. Initialize a small FeedForward layer
    int B = 2;
    int T = 4;
    int C = 16;
    FeedForward ffn(C);

    // Fill weights/biases with non-trivial values
    float* w_fc = ffn.w_fc().data();
    float* b_fc = ffn.b_fc().data();
    float* w_proj = ffn.w_proj().data();
    float* b_proj = ffn.b_proj().data();

    for (size_t i = 0; i < ffn.w_fc().size(); ++i) w_fc[i] = std::sin(static_cast<float>(i)) * 0.1f;
    for (size_t i = 0; i < ffn.b_fc().size(); ++i) b_fc[i] = static_cast<float>(i) * 0.05f;
    for (size_t i = 0; i < ffn.w_proj().size(); ++i) w_proj[i] = std::cos(static_cast<float>(i)) * 0.1f;
    for (size_t i = 0; i < ffn.b_proj().size(); ++i) b_proj[i] = static_cast<float>(i) * 0.02f;

    // 2. Generate random input tensor X
    NastyTensors X({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    float* x_ptr = X.data();
    for (size_t i = 0; i < X.size(); ++i) {
        x_ptr[i] = std::sin(static_cast<float>(i) * 0.5f) * 1.5f;
    }

    // ==========================================
    // Part 1: CPU Forward Pass (Ground Truth)
    // ==========================================
    NastyTensors Y_cpu = ffn.forward(X);

    // ==========================================
    // Part 2: GPU Forward Pass
    // ==========================================
    // Move all weights to GPU
    ffn.w_fc().to_gpu(DATA_);
    ffn.b_fc().to_gpu(DATA_);
    ffn.w_proj().to_gpu(DATA_);
    ffn.b_proj().to_gpu(DATA_);

    NastyTensors X_gpu = X.clone();
    X_gpu.to_gpu(DATA_);

    // Run forward pass on GPU
    NastyTensors Y_gpu = ffn.forward(X_gpu);

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
    ffn.backward(dY, dX_cpu);

    // Clone CPU gradients
    NastyTensors w_fc_g_cpu = ffn.w_fc().clone();
    NastyTensors b_fc_g_cpu = ffn.b_fc().clone();
    NastyTensors w_proj_g_cpu = ffn.w_proj().clone();
    NastyTensors b_proj_g_cpu = ffn.b_proj().clone();

    // Reset gradients
    ffn.w_fc().zero_grad();
    ffn.b_fc().zero_grad();
    ffn.w_proj().zero_grad();
    ffn.b_proj().zero_grad();

    // Run GPU Backward
    NastyTensors dY_gpu = dY.clone();
    dY_gpu.to_gpu(DATA_);

    NastyTensors dX_gpu({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    ffn.backward(dY_gpu, dX_gpu);

    // Copy back
    dX_gpu.to_cpu(DATA_);
    ffn.w_fc().to_cpu(GRAD_);
    ffn.b_fc().to_cpu(GRAD_);
    ffn.w_proj().to_cpu(GRAD_);
    ffn.b_proj().to_cpu(GRAD_);

    // 1. Verify dX
    for (size_t i = 0; i < dX_cpu.size(); ++i) {
        if (!approx_equal(dX_cpu.data()[i], dX_gpu.data()[i])) {
            std::cerr << "[-] BACKWARD INPUT GRADIENT (dX) MISMATCH at index " << i 
                      << ": CPU=" << dX_cpu.data()[i] 
                      << ", GPU=" << dX_gpu.data()[i] << "\n";
            return 1;
        }
    }

    // 2. Verify w_fc gradients
    for (size_t i = 0; i < ffn.w_fc().size(); ++i) {
        if (!approx_equal(w_fc_g_cpu.grad()[i], ffn.w_fc().grad()[i])) {
            std::cerr << "[-] BACKWARD W_FC GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << w_fc_g_cpu.grad()[i] 
                      << ", GPU=" << ffn.w_fc().grad()[i] << "\n";
            return 1;
        }
    }

    // 3. Verify b_fc gradients
    for (size_t i = 0; i < ffn.b_fc().size(); ++i) {
        if (!approx_equal(b_fc_g_cpu.grad()[i], ffn.b_fc().grad()[i])) {
            std::cerr << "[-] BACKWARD B_FC GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << b_fc_g_cpu.grad()[i] 
                      << ", GPU=" << ffn.b_fc().grad()[i] << "\n";
            return 1;
        }
    }

    // 4. Verify w_proj gradients
    for (size_t i = 0; i < ffn.w_proj().size(); ++i) {
        if (!approx_equal(w_proj_g_cpu.grad()[i], ffn.w_proj().grad()[i])) {
            std::cerr << "[-] BACKWARD W_PROJ GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << w_proj_g_cpu.grad()[i] 
                      << ", GPU=" << ffn.w_proj().grad()[i] << "\n";
            return 1;
        }
    }

    // 5. Verify b_proj gradients
    for (size_t i = 0; i < ffn.b_proj().size(); ++i) {
        if (!approx_equal(b_proj_g_cpu.grad()[i], ffn.b_proj().grad()[i])) {
            std::cerr << "[-] BACKWARD B_PROJ GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << b_proj_g_cpu.grad()[i] 
                      << ", GPU=" << ffn.b_proj().grad()[i] << "\n";
            return 1;
        }
    }

    std::cout << "[+] Backward Pass Verification Successful!\n";
    std::cout << "[+] All FeedForward Unit Tests Passed!\n";

    return 0;
}
