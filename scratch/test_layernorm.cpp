#include "LayerNorm.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// A helper function to compare two float arrays with a tiny tolerance.
bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[*] Starting LayerNorm Unit Test...\n";

    // 1. Initialize a small LayerNorm layer
    int B = 2;
    int T = 4;
    int C = 16;
    LayerNorm ln(C);

    // Fill gamma and beta weights with some non-trivial values
    float* gamma_ptr = ln.gamma().data();
    float* beta_ptr = ln.beta().data();
    for (int i = 0; i < C; ++i) {
        gamma_ptr[i] = 1.0f + static_cast<float>(i) * 0.05f;
        beta_ptr[i] = static_cast<float>(i) * 0.02f;
    }

    // 2. Generate random input tensor X
    NastyTensors X({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    float* x_ptr = X.data();
    for (size_t i = 0; i < X.size(); ++i) {
        x_ptr[i] = std::sin(static_cast<float>(i)) * 2.0f; // nice dynamic range
    }

    // ==========================================
    // Part 1: CPU Forward Pass (Ground Truth)
    // ==========================================
    NastyTensors X_cpu = X.clone();
    ln.forward(X_cpu);

    // ==========================================
    // Part 2: GPU Forward Pass
    // ==========================================
    // Upload LayerNorm weights to the GPU (activates GPU path)
    ln.gamma().to_gpu(DATA_);
    ln.beta().to_gpu(DATA_);

    NastyTensors X_gpu = X.clone();
    // Copy the input activations to the GPU
    X_gpu.to_gpu(DATA_);

    // Run the forward pass on GPU
    ln.forward(X_gpu);

    // Copy results back to host for verification
    X_gpu.to_cpu(DATA_);

    // ==========================================
    // Part 3: Forward Verification
    // ==========================================
    for (size_t i = 0; i < X_cpu.size(); ++i) {
        if (!approx_equal(X_cpu.data()[i], X_gpu.data()[i])) {
            std::cerr << "[-] FORWARD PASS MISMATCH at index " << i 
                      << ": CPU=" << X_cpu.data()[i] 
                      << ", GPU=" << X_gpu.data()[i] << "\n";
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
        dy_ptr[i] = std::cos(static_cast<float>(i)) * 0.5f;
    }

    // Run CPU Backward
    NastyTensors dX_cpu({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});
    ln.backward(dY, dX_cpu);

    // Clone CPU gradients for comparison
    NastyTensors dgamma_g_cpu = ln.gamma().clone();
    NastyTensors dbeta_g_cpu = ln.beta().clone();

    // Reset gradients
    ln.gamma().zero_grad();
    ln.beta().zero_grad();

    // Run GPU Backward
    NastyTensors dY_gpu = dY.clone();
    dY_gpu.to_gpu(DATA_);

    NastyTensors dX_gpu({static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(C)});

    ln.backward(dY_gpu, dX_gpu);

    // Copy results back to CPU
    dX_gpu.to_cpu(DATA_);
    ln.gamma().to_cpu(GRAD_);
    ln.beta().to_cpu(GRAD_);

    // 1. Verify dX
    for (size_t i = 0; i < dX_cpu.size(); ++i) {
        if (!approx_equal(dX_cpu.data()[i], dX_gpu.data()[i])) {
            std::cerr << "[-] BACKWARD INPUT GRADIENT (dX) MISMATCH at index " << i 
                      << ": CPU=" << dX_cpu.data()[i] 
                      << ", GPU=" << dX_gpu.data()[i] << "\n";
            return 1;
        }
    }

    // 2. Verify dgamma
    for (size_t i = 0; i < ln.gamma().size(); ++i) {
        if (!approx_equal(dgamma_g_cpu.grad()[i], ln.gamma().grad()[i])) {
            std::cerr << "[-] BACKWARD GAMMA GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << dgamma_g_cpu.grad()[i] 
                      << ", GPU=" << ln.gamma().grad()[i] << "\n";
            return 1;
        }
    }

    // 3. Verify dbeta
    for (size_t i = 0; i < ln.beta().size(); ++i) {
        if (!approx_equal(dbeta_g_cpu.grad()[i], ln.beta().grad()[i])) {
            std::cerr << "[-] BACKWARD BETA GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << dbeta_g_cpu.grad()[i] 
                      << ", GPU=" << ln.beta().grad()[i] << "\n";
            return 1;
        }
    }

    std::cout << "[+] Backward Pass Verification Successful!\n";
    std::cout << "[+] All LayerNorm Unit Tests Passed!\n";

    return 0;
}
