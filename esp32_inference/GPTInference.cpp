#include "GPTInference.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

GPTInference::GPTInference() {}

GPTInference::~GPTInference() {
    if (blocks) delete[] blocks;
    if (id_to_char) free(id_to_char);

#ifdef ESP_PLATFORM
    if (weights_buffer) heap_caps_free(weights_buffer);
#else
    if (weights_buffer) free(weights_buffer);
#endif

    if (x_buffer) free(x_buffer);
    if (x2_buffer) free(x2_buffer);
    if (qkv_buffer) free(qkv_buffer);
    if (att_scores) free(att_scores);
    if (att_out) free(att_out);
    if (mlp_h) free(mlp_h);
}

bool GPTInference::load_model(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("[-] Error: Failed to open model file: %s\n", filepath);
        return false;
    }

    // Read header
    if (fread(&config, sizeof(GPTConfig), 1, f) != 1) {
        printf("[-] Error: Failed to read model header.\n");
        fclose(f);
        return false;
    }

    // Verify magic and version
    if (config.magic != 0x47505432) {
        printf("[-] Error: Invalid magic number (0x%X). Expected 0x47505432.\n", config.magic);
        fclose(f);
        return false;
    }
    if (config.version != 1) {
        printf("[-] Error: Unsupported model version (%d).\n", config.version);
        fclose(f);
        return false;
    }

    printf("[+] Model configuration loaded:\n");
    printf("    vocab_size: %d\n", config.vocab_size);
    printf("    max_seq_len: %d\n", config.max_seq_len);
    printf("    embedding_dim: %d\n", config.embedding_dim);
    printf("    num_heads: %d\n", config.num_heads);
    printf("    num_layers: %d\n", config.num_layers);

    // Read vocabulary mapping
    id_to_char = (char*)malloc(config.vocab_size * sizeof(char));
    if (fread(id_to_char, sizeof(char), config.vocab_size, f) != (size_t)config.vocab_size) {
        printf("[-] Error: Failed to read vocabulary mapping.\n");
        fclose(f);
        return false;
    }

    // Read padding bytes to align to 4-byte boundaries
    size_t vocab_bytes = config.vocab_size * sizeof(char);
    size_t padding = (4 - (vocab_bytes % 4)) % 4;
    if (padding > 0) {
        char pad[3];
        if (fread(pad, sizeof(char), padding, f) != padding) {
            printf("[-] Error: Failed to read padding bytes.\n");
            fclose(f);
            return false;
        }
    }

    // Calculate parameter sizes
    size_t vocab_size = config.vocab_size;
    size_t max_seq_len = config.max_seq_len;
    size_t embedding_dim = config.embedding_dim;
    size_t num_layers = config.num_layers;

    size_t num_floats = 0;
    num_floats += vocab_size * embedding_dim; // wte
    num_floats += max_seq_len * embedding_dim; // wpe

    for (size_t i = 0; i < num_layers; ++i) {
        num_floats += embedding_dim; // ln1_gamma
        num_floats += embedding_dim; // ln1_beta
        num_floats += embedding_dim * 3 * embedding_dim; // w_qkv
        num_floats += 3 * embedding_dim; // b_qkv
        num_floats += embedding_dim * embedding_dim; // w_proj
        num_floats += embedding_dim; // b_proj
        num_floats += embedding_dim; // ln2_gamma
        num_floats += embedding_dim; // ln2_beta
        num_floats += embedding_dim * 4 * embedding_dim; // w_fc
        num_floats += 4 * embedding_dim; // b_fc
        num_floats += 4 * embedding_dim * embedding_dim; // w_proj_mlp
        num_floats += embedding_dim; // b_proj_mlp
    }

    num_floats += embedding_dim; // ln_f_gamma
    num_floats += embedding_dim; // ln_f_beta

    weights_size_bytes = num_floats * sizeof(float);
    printf("[+] Model weight floats: %zu (~%.2f MB)\n", num_floats, (double)weights_size_bytes / (1024.0 * 1024.0));

    // Allocate memory for weights (prefer PSRAM if on ESP32)
#ifdef ESP_PLATFORM
    printf("[*] Allocating weights buffer in PSRAM...\n");
    weights_buffer = heap_caps_malloc(weights_size_bytes, MALLOC_CAP_SPIRAM);
#else
    printf("[*] Allocating weights buffer on Host CPU...\n");
    weights_buffer = malloc(weights_size_bytes);
#endif

    if (!weights_buffer) {
        printf("[-] Error: Failed to allocate weights buffer of size %zu.\n", weights_size_bytes);
        fclose(f);
        return false;
    }

    // Read weights from file
    if (fread(weights_buffer, sizeof(float), num_floats, f) != num_floats) {
        printf("[-] Error: Failed to read model weights.\n");
        fclose(f);
        return false;
    }
    fclose(f);
    printf("[+] Model weights loaded successfully.\n");

    // Assign weight pointers
    float* ptr = (float*)weights_buffer;
    wte = ptr; ptr += vocab_size * embedding_dim;
    wpe = ptr; ptr += max_seq_len * embedding_dim;

    blocks = new BlockWeights[num_layers];
    for (size_t i = 0; i < num_layers; ++i) {
        blocks[i].ln1_gamma = ptr; ptr += embedding_dim;
        blocks[i].ln1_beta = ptr; ptr += embedding_dim;
        blocks[i].w_qkv = ptr; ptr += embedding_dim * 3 * embedding_dim;
        blocks[i].b_qkv = ptr; ptr += 3 * embedding_dim;
        blocks[i].w_proj = ptr; ptr += embedding_dim * embedding_dim;
        blocks[i].b_proj = ptr; ptr += embedding_dim;
        blocks[i].ln2_gamma = ptr; ptr += embedding_dim;
        blocks[i].ln2_beta = ptr; ptr += embedding_dim;
        blocks[i].w_fc = ptr; ptr += embedding_dim * 4 * embedding_dim;
        blocks[i].b_fc = ptr; ptr += 4 * embedding_dim;
        blocks[i].w_proj_mlp = ptr; ptr += 4 * embedding_dim * embedding_dim;
        blocks[i].b_proj_mlp = ptr; ptr += embedding_dim;
    }
    ln_f_gamma = ptr; ptr += embedding_dim;
    ln_f_beta = ptr; ptr += embedding_dim;

    // Verify pointer arithmetic matches num_floats
    if (ptr - (float*)weights_buffer != (ptrdiff_t)num_floats) {
        printf("[-] Error: Weight pointer offset mismatch! Offset diff: %td, expected: %zu\n", 
               ptr - (float*)weights_buffer, num_floats);
        return false;
    }

    // Allocate memory for activations (Internal fast SRAM)
    // We optimize memory by reusing buffers
    x_buffer = (float*)malloc(max_seq_len * embedding_dim * sizeof(float));
    x2_buffer = (float*)malloc(max_seq_len * embedding_dim * sizeof(float));
    qkv_buffer = (float*)malloc(max_seq_len * 4 * embedding_dim * sizeof(float)); // Shared scratch buffer for qkv and mlp_h
    att_scores = (float*)malloc(max_seq_len * max_seq_len * sizeof(float));        // Causal attention matrix for a single head
    
    if (!x_buffer || !x2_buffer || !qkv_buffer || !att_scores) {
        printf("[-] Error: Failed to allocate activation buffers in internal memory.\n");
        return false;
    }
    printf("[+] Activation buffers allocated successfully (~308 KB in internal memory).\n");

    return true;
}

void GPTInference::forward(const int* input_tokens, int T, float* out_logits) {
    if (T > config.max_seq_len) {
        printf("[-] Warning: sequence length %d exceeds max %d. Truncating.\n", T, config.max_seq_len);
        T = config.max_seq_len;
    }

    int C = config.embedding_dim;
    int V = config.vocab_size;
    int num_heads = config.num_heads;
    int head_dim = C / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // 1. Embedding lookup: x = wte[token] + wpe[pos]
    for (int t = 0; t < T; ++t) {
        int token = input_tokens[t];
        const float* wte_row = wte + token * C;
        const float* wpe_row = wpe + t * C;
        float* x_row = x_buffer + t * C;
        for (int c = 0; c < C; ++c) {
            x_row[c] = wte_row[c] + wpe_row[c];
        }
    }

    // 2. Process Blocks
    for (int l = 0; l < config.num_layers; ++l) {
        const BlockWeights& block = blocks[l];

        // LayerNorm 1
        layernorm(x_buffer, block.ln1_gamma, block.ln1_beta, x2_buffer, T, C);

        // QKV Projection: x2_buffer (T x C) * w_qkv (C x 3C) + b_qkv (3C) -> qkv_buffer (T x 3C)
        matmul(x2_buffer, block.w_qkv, qkv_buffer, T, C, 3 * C);
        for (int t = 0; t < T; ++t) {
            float* qkv_row = qkv_buffer + t * 3 * C;
            for (int i = 0; i < 3 * C; ++i) {
                qkv_row[i] += block.b_qkv[i];
            }
        }

        // Multi-head attention
        // qkv_buffer is layout: [T, 3, num_heads, head_dim]
        // Q: qkv_buffer[t, 0, h, d] -> qkv_buffer + t*3C + 0*C + h*head_dim + d
        // K: qkv_buffer[t, 1, h, d] -> qkv_buffer + t*3C + 1*C + h*head_dim + d
        // V: qkv_buffer[t, 2, h, d] -> qkv_buffer + t*3C + 2*C + h*head_dim + d
        for (int h = 0; h < num_heads; ++h) {
            // Compute causal attention matrix [T, T] for this head
            for (int q_t = 0; q_t < T; ++q_t) {
                const float* Q = qkv_buffer + q_t * 3 * C + h * head_dim;
                for (int k_t = 0; k_t <= q_t; ++k_t) {
                    const float* K = qkv_buffer + k_t * 3 * C + C + h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += Q[d] * K[d];
                    }
                    att_scores[q_t * T + k_t] = dot * scale;
                }
                // Softmax on key dimensions 0..q_t
                softmax(att_scores + q_t * T, q_t + 1);
            }

            // Compute attention output O: x2_buffer (T x C)
            for (int q_t = 0; q_t < T; ++q_t) {
                float* O = x2_buffer + q_t * C + h * head_dim;
                for (int d = 0; d < head_dim; ++d) O[d] = 0.0f;

                for (int k_t = 0; k_t <= q_t; ++k_t) {
                    float att_weight = att_scores[q_t * T + k_t];
                    const float* V = qkv_buffer + k_t * 3 * C + 2 * C + h * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        O[d] += att_weight * V[d];
                    }
                }
            }
        }

        // Project attention output: x2_buffer (T x C) * w_proj (C x C) + b_proj -> qkv_buffer (first T x C)
        matmul(x2_buffer, block.w_proj, qkv_buffer, T, C, C);
        // Add bias and add to residual stream x_buffer
        for (int t = 0; t < T; ++t) {
            float* x_row = x_buffer + t * C;
            const float* proj_row = qkv_buffer + t * C;
            for (int c = 0; c < C; ++c) {
                x_row[c] += proj_row[c] + block.b_proj[c];
            }
        }

        // LayerNorm 2
        layernorm(x_buffer, block.ln2_gamma, block.ln2_beta, x2_buffer, T, C);

        // MLP first linear layer: x2_buffer (T x C) * w_fc (C x 4C) + b_fc -> qkv_buffer (T x 4C)
        matmul(x2_buffer, block.w_fc, qkv_buffer, T, C, 4 * C);
        for (int t = 0; t < T; ++t) {
            float* fc_row = qkv_buffer + t * 4 * C;
            for (int i = 0; i < 4 * C; ++i) {
                fc_row[i] += block.b_fc[i];
            }
        }
        // GeLU
        gelu(qkv_buffer, T * 4 * C);

        // MLP second linear layer: qkv_buffer (T x 4C) * w_proj_mlp (4C x C) + b_proj_mlp -> x2_buffer (T x C)
        matmul(qkv_buffer, block.w_proj_mlp, x2_buffer, T, 4 * C, C);
        // Add bias and add to residual stream x_buffer
        for (int t = 0; t < T; ++t) {
            float* x_row = x_buffer + t * C;
            const float* proj_row = x2_buffer + t * C;
            for (int c = 0; c < C; ++c) {
                x_row[c] += proj_row[c] + block.b_proj_mlp[c];
            }
        }
    }

    // 3. Final LayerNorm (only need for the last token T-1!)
    float* last_x_ln = x2_buffer; // Shape: [1, C]
    layernorm(x_buffer + (T - 1) * C, ln_f_gamma, ln_f_beta, last_x_ln, 1, C);

    // 4. Classifier Head: logits = last_x_ln (1 x C) * wte^T (C x V)
    // We compute matmul with wte transposed (wte original shape was V x C).
    // result: out_logits (1 x V)
    matmul_transposed_b(last_x_ln, wte, out_logits, 1, C, V);
}

// Optimized matrix multiplication (i-k-j loop order for cache friendliness)
void GPTInference::matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float val = A[i * K + k];
            const float* b_row = B + k * N;
            float* c_row = C + i * N;
            for (int j = 0; j < N; ++j) {
                c_row[j] += val * b_row[j];
            }
        }
    }
}

// Matrix multiplication with transposed B: C = A * B^T
// A is M x K, B is N x K (since B is transposed, its original shape before transpose was N x K)
// C is M x N
void GPTInference::matmul_transposed_b(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        for (int j = 0; j < N; ++j) {
            const float* b_row = B + j * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            c_row[j] = sum;
        }
    }
}

void GPTInference::layernorm(const float* x, const float* gamma, const float* beta, float* out, int T, int C) {
    for (int t = 0; t < T; ++t) {
        const float* x_row = x + t * C;
        float* out_row = out + t * C;

        // Calculate mean
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) {
            mean += x_row[c];
        }
        mean /= C;

        // Calculate variance
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float diff = x_row[c] - mean;
            var += diff * diff;
        }
        var /= C;

        // Normalize and scale/shift
        float std_dev = 1.0f / sqrtf(var + 1e-5f);
        for (int c = 0; c < C; ++c) {
            out_row[c] = (x_row[c] - mean) * std_dev * gamma[c] + beta[c];
        }
    }
}

void GPTInference::gelu(float* x, int size) {
    for (int i = 0; i < size; ++i) {
        float val = x[i];
        x[i] = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
    }
}

void GPTInference::softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum_exp += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= sum_exp;
    }
}
