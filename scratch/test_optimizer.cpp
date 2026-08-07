#include "Optimizer.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[*] Starting AdamW Optimizer Unit Test...\n";

    size_t size = 100;
    NastyTensors w_cpu({size});
    NastyTensors w_gpu({size});

    // Initialize both parameters with identical values
    float* w_cpu_ptr = w_cpu.data();
    float* w_gpu_ptr = w_gpu.data();
    for (size_t i = 0; i < size; ++i) {
        float val = std::sin(static_cast<float>(i)) * 0.1f;
        w_cpu_ptr[i] = val;
        w_gpu_ptr[i] = val;
    }

    w_cpu.init_grad();
    w_gpu.init_grad();

    // Create optimizers
    std::vector<NastyTensors*> params_cpu = {&w_cpu};
    std::vector<NastyTensors*> params_gpu = {&w_gpu};

    AdamW opt_cpu(params_cpu, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);
    AdamW opt_gpu(params_gpu, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);

    // Set GPU weights to GPU mode
    w_gpu.to_gpu(DATA_);

    // Run 5 optimizer steps with deterministic grads
    for (int step = 1; step <= 5; ++step) {
        // Set grads on CPU and GPU
        float* g_cpu = w_cpu.grad();
        for (size_t i = 0; i < size; ++i) {
            g_cpu[i] = std::cos(static_cast<float>(i + step)) * 0.01f;
        }

        // Upload grads to GPU
        w_gpu.to_gpu(GRAD_);
        float* g_gpu = w_gpu.grad();
        for (size_t i = 0; i < size; ++i) {
            g_gpu[i] = g_cpu[i];
        }
        // Force upload grads to GPU
        cudaMemcpy(w_gpu.device_grad(), g_gpu, size * sizeof(float), cudaMemcpyHostToDevice);

        // Run step
        opt_cpu.step();
        opt_gpu.step();

        // Download weights from GPU
        w_gpu.to_cpu(DATA_);

        // Verify
        float max_diff = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float diff = std::abs(w_cpu.data()[i] - w_gpu.data()[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        std::cout << "[Step " << step << "] Max Weights Absolute Difference: " << max_diff << "\n";
        assert(max_diff < 1e-5f && "Optimizer update mismatch!");
    }

    std::cout << "[+] AdamW Optimizer Unit Test Passed!\n";
    return 0;
}
