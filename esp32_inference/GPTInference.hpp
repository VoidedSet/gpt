#pragma once

#include <stddef.h>
#include <stdint.h>

struct GPTConfig {
    int magic;
    int version;
    int vocab_size;
    int max_seq_len;
    int embedding_dim;
    int num_heads;
    int num_layers;
};

class GPTInference {
public:
    GPTConfig config;
    char* id_to_char = nullptr;

    // Model weight pointers
    float* wte = nullptr;
    float* wpe = nullptr;

    struct BlockWeights {
        float* ln1_gamma = nullptr;
        float* ln1_beta = nullptr;
        float* w_qkv = nullptr;
        float* b_qkv = nullptr;
        float* w_proj = nullptr;
        float* b_proj = nullptr;
        float* ln2_gamma = nullptr;
        float* ln2_beta = nullptr;
        float* w_fc = nullptr;
        float* b_fc = nullptr;
        float* w_proj_mlp = nullptr;
        float* b_proj_mlp = nullptr;
    };

    BlockWeights* blocks = nullptr;

    float* ln_f_gamma = nullptr;
    float* ln_f_beta = nullptr;

    // Buffer to hold weights
    void* weights_buffer = nullptr;
    size_t weights_size_bytes = 0;

    // Temporary memory buffers for activations (allocated in internal fast SRAM)
    float* x_buffer = nullptr;        // Shape [max_seq_len * embedding_dim]
    float* x2_buffer = nullptr;       // Shape [max_seq_len * embedding_dim]
    float* qkv_buffer = nullptr;      // Shape [max_seq_len * 3 * embedding_dim]
    float* att_scores = nullptr;      // Shape [num_heads * max_seq_len * max_seq_len]
    float* att_out = nullptr;         // Shape [max_seq_len * embedding_dim]
    float* mlp_h = nullptr;           // Shape [max_seq_len * 4 * embedding_dim]

    GPTInference();
    ~GPTInference();

    // Loads the model file from standard C FILE* API
    bool load_model(const char* filepath);

    // Forward pass
    // Runs inference on the input token sequence of length T.
    // out_logits must point to a buffer of size vocab_size.
    // To save computation and memory, we only calculate logits for the LAST token (T - 1).
    void forward(const int* input_tokens, int T, float* out_logits);

    // Utility functions for matrix math
    static void matmul(const float* A, const float* B, float* C, int M, int K, int N);
    static void matmul_transposed_b(const float* A, const float* B, float* C, int M, int K, int N);
    static void layernorm(const float* x, const float* gamma, const float* beta, float* out, int T, int C);
    static void gelu(float* x, int size);
    static void softmax(float* x, int size);
};
