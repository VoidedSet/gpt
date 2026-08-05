#include "Embedding.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[*] Starting Embedding Unit Test...\n";

    int vocab_size = 10;
    int max_seq_len = 8;
    int embedding_dim = 16;
    Embedding emb(vocab_size, max_seq_len, embedding_dim);

    std::vector<int> tokens = {1, 3, 5, 2, 4, 0, 7, 6};

    // cpu forward pass for baseline data
    NastyTensors out_cpu = emb.forward(tokens, 1, 8);

    //gpu forward pass
    emb.wte().to_gpu(DATA_);
    emb.wpe().to_gpu(DATA_);

    NastyTensors out_gpu = emb.forward(tokens, 1, 8);

    out_gpu.to_cpu(DATA_);

    //compare
    size_t size = out_cpu.size();
    const float* cpu_data = out_cpu.data();
    const float* gpu_data = out_gpu.data();

    for (size_t i = 0; i < size; ++i) {
        if (!approx_equal(cpu_data[i], gpu_data[i])) {
            std::cerr << "[-] FORWARD PASS MISMATCH at index " << i 
                      << ": CPU=" << cpu_data[i] 
                      << ", GPU=" << gpu_data[i] << "\n";
            return 1; 
        }
    }
    std::cout << "[+] Forward Pass Verification Successful!\n";

    //backward pass
    std::cout << "[*] Starting Backward Pass Verification...\n";

    NastyTensors dY({1, 8, 16});
    float* dy_ptr = dY.data();
    for (size_t i = 0; i < dY.size(); ++i) {
        dy_ptr[i] = static_cast<float>(i) * 0.01f;
    }

    // run on cpu 
    emb.backward(dY);

    NastyTensors wte_grad_cpu = emb.wte().clone(),
                 wpe_grad_cpu = emb.wpe().clone();

    emb.wte().zero_grad();
    emb.wpe().zero_grad();

    // run on gpu
    dY.to_gpu(DATA_);
    emb.backward(dY);

    emb.wte().to_cpu(GRAD_);
    emb.wpe().to_cpu(GRAD_);

    //compare
    const float* wte_g_cpu = wte_grad_cpu.grad();
    const float* wte_g_gpu = emb.wte().grad();
    for (size_t i = 0; i < emb.wte().size(); ++i) {
        if (!approx_equal(wte_g_cpu[i], wte_g_gpu[i])) {
            std::cerr << "[-] BACKWARD WTE GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << wte_g_cpu[i] 
                      << ", GPU=" << wte_g_gpu[i] << "\n";
            return 1;
        }
    }

    const float* wpe_g_cpu = wpe_grad_cpu.grad();
    const float* wpe_g_gpu = emb.wpe().grad();
    for (size_t i = 0; i < emb.wpe().size(); ++i) {
        if (!approx_equal(wpe_g_cpu[i], wpe_g_gpu[i])) {
            std::cerr << "[-] BACKWARD WPE GRADIENT MISMATCH at index " << i 
                      << ": CPU=" << wpe_g_cpu[i] 
                      << ", GPU=" << wpe_g_gpu[i] << "\n";
            return 1;
        }
    }

    std::cout << "[+] Backward Pass Verification Successful!\n";
    std::cout << "[+] All Embedding Unit Tests Passed!\n";

    return 0;
}
