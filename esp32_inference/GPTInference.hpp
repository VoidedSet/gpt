#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <string>

struct GPTConfig {
    int magic;
    int version;
    int vocab_size;
    int max_seq_len;
    int embedding_dim;
    int num_heads;
    int num_layers;
    int tokenizer_type;     // 0 = CHAR, 1 = BPE
    int quantization_level; // 0 = FP32, 1 = BF16, 2 = INT8
};

struct QuantizedTensor {
    float scale = 1.0f;     // Used only for INT8
    void* data = nullptr;   // Pointer to raw elements (float, uint16_t, or int8_t)
};

class GPTInference {
public:
    GPTConfig config;
    
    // Character Tokenizer vocab
    char* id_to_char = nullptr;
    
    // BPE Tokenizer merges and vocab
    int num_merges = 0;
    int* merge_left = nullptr;
    int* merge_right = nullptr;
    std::vector<std::string> bpe_vocab;

    // Model weight pointers
    QuantizedTensor wte;
    QuantizedTensor wpe;

    struct BlockWeights {
        QuantizedTensor ln1_gamma;
        QuantizedTensor ln1_beta;
        QuantizedTensor w_qkv;
        QuantizedTensor b_qkv;
        QuantizedTensor w_proj;
        QuantizedTensor b_proj;
        QuantizedTensor ln2_gamma;
        QuantizedTensor ln2_beta;
        QuantizedTensor w_fc;
        QuantizedTensor b_fc;
        QuantizedTensor w_proj_mlp;
        QuantizedTensor b_proj_mlp;
    };

    BlockWeights* blocks = nullptr;

    QuantizedTensor ln_f_gamma;
    QuantizedTensor ln_f_beta;

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
    void forward(const int* input_tokens, int T, float* out_logits);

    // Helper functions for BPE decode
    std::string decode_token(int token_id) const;

    // Utility functions for matrix math
    static void matmul(const float* A, const QuantizedTensor& B, float* C, int M, int K, int N, int quant_level);
    static void matmul_transposed_b(const float* A, const QuantizedTensor& B, float* C, int M, int K, int N, int quant_level);
    static void layernorm(const float* x, const QuantizedTensor& gamma, const QuantizedTensor& beta, float* out, int T, int C, int quant_level);
    static void gelu(float* x, int size);
    static void softmax(float* x, int size);
    
    // Helper to dequantize BF16 to float32
    static inline float dequantize_bf16(uint16_t val) {
        uint32_t bits = (static_cast<uint32_t>(val) << 16);
        float f;
        __builtin_memcpy(&f, &bits, sizeof(float));
        return f;
    }
};
