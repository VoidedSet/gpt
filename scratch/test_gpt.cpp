#include "GPT.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// A helper function to compare two float arrays with a tiny tolerance.
bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[*] Starting GPT End-to-End Unit Test...\n";

    // 1. Initialize GPT model with a small scale
    int vocab_size = 65;
    int max_seq_len = 32;
    int embedding_dim = 16;
    int num_heads = 2;
    int num_layers = 1;

    GPT gpt(vocab_size, max_seq_len, embedding_dim, num_heads, num_layers);

    // Initialize all parameters with non-trivial deterministic values
    int p_idx = 0;
    for (auto* param : gpt.get_parameters()) {
        float* data = param->data();
        for (size_t i = 0; i < param->size(); ++i) {
            data[i] = std::sin(static_cast<float>(p_idx++)) * 0.05f;
        }
    }

    // 2. Generate random input tokens & targets
    int B = 2;
    int T = 4;
    std::vector<int> input_tokens(B * T);
    std::vector<int> targets(B * T);
    for (int i = 0; i < B * T; ++i) {
        input_tokens[i] = (i * 7) % vocab_size;
        targets[i] = (i * 11 + 3) % vocab_size;
    }

    // ==========================================
    // Part 1: CPU Forward & Backward (Ground Truth)
    // ==========================================
    NastyTensors Y_cpu = gpt.forward(input_tokens, B, T);
    float loss_cpu = gpt.backward(targets);

    // Save CPU parameter gradients for verification
    std::vector<NastyTensors> grads_cpu;
    for (auto* param : gpt.get_parameters()) {
        grads_cpu.push_back(param->clone());
    }

    // ==========================================
    // Part 2: Move parameters to GPU & Reset Gradients
    // ==========================================
    for (auto* param : gpt.get_parameters()) {
        param->zero_grad();
        param->to_gpu(DATA_);
        // Also zero gradients on GPU
        param->zero_grad(); 
    }

    // ==========================================
    // Part 3: GPU Forward & Backward
    // ==========================================
    NastyTensors Y_gpu = gpt.forward(input_tokens, B, T);
    Y_gpu.to_cpu(DATA_);

    float loss_gpu = gpt.backward(targets);

    // Copy gradients back to CPU
    for (auto* param : gpt.get_parameters()) {
        param->to_cpu(GRAD_);
    }

    // ==========================================
    // Part 4: Verification
    // ==========================================
    
    // 1. Verify Logits (Forward Pass)
    float max_logit_diff = 0.0f;
    for (size_t i = 0; i < Y_cpu.size(); ++i) {
        float diff = std::abs(Y_cpu.data()[i] - Y_gpu.data()[i]);
        if (diff > max_logit_diff) {
            max_logit_diff = diff;
        }
    }
    std::cout << "[*] Max Logit Absolute Difference: " << max_logit_diff << "\n";
    if (max_logit_diff > 1e-4f) {
        std::cerr << "[-] LOGITS MISMATCH: Exceeded tolerance!\n";
        return 1;
    }
    std::cout << "[+] Logits (Forward Pass) Verification Successful!\n";

    // 2. Verify Cross-Entropy Loss
    float loss_diff = std::abs(loss_cpu - loss_gpu);
    std::cout << "[*] Loss Absolute Difference: " << loss_diff << "\n";
    if (loss_diff > 1e-4f) {
        std::cerr << "[-] LOSS MISMATCH: Exceeded tolerance!\n";
        return 1;
    }
    std::cout << "[+] Loss Verification Successful! (Loss = " << loss_gpu << ")\n";

    // 3. Verify Parameter Gradients
    auto params = gpt.get_parameters();
    bool any_grad_failed = false;
    for (size_t p = 0; p < params.size(); ++p) {
        NastyTensors* param = params[p];
        const float* g_cpu = grads_cpu[p].grad();
        const float* g_gpu = param->grad();
        float max_grad_diff = 0.0f;

        assert(grads_cpu[p].size() == param->size());
        for (size_t i = 0; i < param->size(); ++i) {
            float diff = std::abs(g_cpu[i] - g_gpu[i]);
            if (diff > max_grad_diff) {
                max_grad_diff = diff;
            }
        }
        std::cout << "[*] Parameter " << p << " Max Gradient Absolute Difference: " << max_grad_diff << "\n";
        if (max_grad_diff > 1e-4f) {
            std::cerr << "[-] GRADIENT MISMATCH in parameter " << p << ": Exceeded tolerance!\n";
            any_grad_failed = true;
        }
    }
    if (any_grad_failed) {
        return 1;
    }
    std::cout << "[+] All Parameter Gradients (Backward Pass) Verification Successful!\n";
    std::cout << "[+] Full GPT End-to-End Test Passed!\n";

    return 0;
}
