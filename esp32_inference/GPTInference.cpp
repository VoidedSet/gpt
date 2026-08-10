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
    if (merge_left) free(merge_left);
    if (merge_right) free(merge_right);

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

// Helper to determine the parameter sizes (number of elements)
static size_t get_param_size_bytes(size_t num_elements, int quant_level) {
    if (quant_level == 0) {
        return num_elements * sizeof(float);
    } else if (quant_level == 1) {
        size_t bytes = num_elements * sizeof(uint16_t);
        if (bytes % 4 != 0) bytes += 2; // Add alignment padding
        return bytes;
    } else { // quant_level == 2 (INT8)
        size_t bytes = sizeof(float) + num_elements * sizeof(int8_t);
        size_t padding = (4 - (bytes % 4)) % 4;
        return bytes + padding;
    }
}

// Helper to assign pointers and parse parameter layouts
static QuantizedTensor assign_param(uint8_t*& current_ptr, size_t num_elements, int quant_level) {
    QuantizedTensor tensor;
    if (quant_level == 0) {
        tensor.scale = 1.0f;
        tensor.data = (void*)current_ptr;
        current_ptr += num_elements * sizeof(float);
    } else if (quant_level == 1) {
        tensor.scale = 1.0f;
        tensor.data = (void*)current_ptr;
        current_ptr += num_elements * sizeof(uint16_t);
        if ((num_elements * sizeof(uint16_t)) % 4 != 0) {
            current_ptr += 2; // skip padding
        }
    } else { // quant_level == 2 (INT8)
        // Read scale factor (4 bytes float)
        tensor.scale = *(float*)current_ptr;
        current_ptr += sizeof(float);
        
        tensor.data = (void*)current_ptr;
        current_ptr += num_elements * sizeof(int8_t);
        
        // Align to 4-byte boundary
        size_t bytes_written = sizeof(float) + num_elements * sizeof(int8_t);
        size_t padding = (4 - (bytes_written % 4)) % 4;
        current_ptr += padding;
    }
    return tensor;
}

bool GPTInference::load_model(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("[-] Error: Failed to open model file: %s\n", filepath);
        return false;
    }

    // Read header up to original size
    // Note: We need to parse magic and version first to handle layout correctly
    int magic = 0;
    int version = 0;
    if (fread(&magic, sizeof(int), 1, f) != 1 || fread(&version, sizeof(int), 1, f) != 1) {
        printf("[-] Error: Failed to read model metadata.\n");
        fclose(f);
        return false;
    }

    if (magic != 0x47505432) {
        printf("[-] Error: Invalid magic number (0x%X). Expected 0x47505432.\n", magic);
        fclose(f);
        return false;
    }

    config.magic = magic;
    config.version = version;

    // Read metadata values depending on version
    if (fread(&config.vocab_size, sizeof(int), 1, f) != 1 ||
        fread(&config.max_seq_len, sizeof(int), 1, f) != 1 ||
        fread(&config.embedding_dim, sizeof(int), 1, f) != 1 ||
        fread(&config.num_heads, sizeof(int), 1, f) != 1 ||
        fread(&config.num_layers, sizeof(int), 1, f) != 1) {
        printf("[-] Error: Failed to read model dimensions.\n");
        fclose(f);
        return false;
    }

    if (version == 1) {
        config.tokenizer_type = 0;     // Default CHAR
        config.quantization_level = 0; // Default FP32
    } else if (version == 2) {
        if (fread(&config.tokenizer_type, sizeof(int), 1, f) != 1 ||
            fread(&config.quantization_level, sizeof(int), 1, f) != 1) {
            printf("[-] Error: Failed to read version 2 metadata flags.\n");
            fclose(f);
            return false;
        }
    } else {
        printf("[-] Error: Unsupported model version (%d).\n", version);
        fclose(f);
        return false;
    }

    printf("[+] Model configuration loaded:\n");
    printf("    vocab_size: %d\n", config.vocab_size);
    printf("    max_seq_len: %d\n", config.max_seq_len);
    printf("    embedding_dim: %d\n", config.embedding_dim);
    printf("    num_heads: %d\n", config.num_heads);
    printf("    num_layers: %d\n", config.num_layers);
    printf("    tokenizer_type: %s\n", config.tokenizer_type == 0 ? "CHAR" : "BPE");
    printf("    quantization_level: %d (%s)\n", config.quantization_level,
           config.quantization_level == 0 ? "FP32" : (config.quantization_level == 1 ? "BF16" : "INT8"));

    // Read Vocabulary
    if (config.tokenizer_type == 0) {
        id_to_char = (char*)malloc(config.vocab_size * sizeof(char));
        if (fread(id_to_char, sizeof(char), config.vocab_size, f) != (size_t)config.vocab_size) {
            printf("[-] Error: Failed to read character vocabulary mapping.\n");
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
    } else {
        // Load BPE vocabulary merges
        if (fread(&num_merges, sizeof(int), 1, f) != 1) {
            printf("[-] Error: Failed to read number of BPE merges.\n");
            fclose(f);
            return false;
        }
        
        merge_left = (int*)malloc(num_merges * sizeof(int));
        merge_right = (int*)malloc(num_merges * sizeof(int));
        for (int i = 0; i < num_merges; ++i) {
            if (fread(&merge_left[i], sizeof(int), 1, f) != 1 ||
                fread(&merge_right[i], sizeof(int), 1, f) != 1) {
                printf("[-] Error: Failed to read BPE merges rule at index %d.\n", i);
                fclose(f);
                return false;
            }
        }
        
        // Reconstruct BPE vocab mappings from merges
        bpe_vocab.resize(256);
        for (int i = 0; i < 256; ++i) {
            bpe_vocab[i] = std::string(1, (char)i);
        }
        for (int i = 0; i < num_merges; ++i) {
            int left = merge_left[i];
            int right = merge_right[i];
            if (left < 0 || left >= (int)bpe_vocab.size() || right < 0 || right >= (int)bpe_vocab.size()) {
                printf("[-] Error: BPE merge out of bounds: %d, %d at index %d\n", left, right, i);
                fclose(f);
                return false;
            }
            std::string merged_str = bpe_vocab[left] + bpe_vocab[right];
            bpe_vocab.push_back(merged_str);
        }
        printf("[+] BPE vocabulary expanded successfully to %zu tokens.\n", bpe_vocab.size());
    }

    // Calculate parameter sizes
    size_t vocab_size = config.vocab_size;
    size_t max_seq_len = config.max_seq_len;
    size_t embedding_dim = config.embedding_dim;
    size_t num_layers = config.num_layers;
    int quant = config.quantization_level;

    std::vector<size_t> param_sizes;
    param_sizes.push_back(vocab_size * embedding_dim); // wte
    param_sizes.push_back(max_seq_len * embedding_dim); // wpe

    for (size_t i = 0; i < num_layers; ++i) {
        param_sizes.push_back(embedding_dim); // ln1_gamma
        param_sizes.push_back(embedding_dim); // ln1_beta
        param_sizes.push_back(embedding_dim * 3 * embedding_dim); // w_qkv
        param_sizes.push_back(3 * embedding_dim); // b_qkv
        param_sizes.push_back(embedding_dim * embedding_dim); // w_proj
        param_sizes.push_back(embedding_dim); // b_proj
        param_sizes.push_back(embedding_dim); // ln2_gamma
        param_sizes.push_back(embedding_dim); // ln2_beta
        param_sizes.push_back(embedding_dim * 4 * embedding_dim); // w_fc
        param_sizes.push_back(4 * embedding_dim); // b_fc
        param_sizes.push_back(4 * embedding_dim * embedding_dim); // w_proj_mlp
        param_sizes.push_back(embedding_dim); // b_proj_mlp
    }

    param_sizes.push_back(embedding_dim); // ln_f_gamma
    param_sizes.push_back(embedding_dim); // ln_f_beta

    size_t total_bytes = 0;
    for (size_t size : param_sizes) {
        total_bytes += get_param_size_bytes(size, quant);
    }
    weights_size_bytes = total_bytes;
    printf("[+] Allocated weights binary size: %zu bytes (~%.2f MB)\n", 
           weights_size_bytes, (double)weights_size_bytes / (1024.0 * 1024.0));

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
    if (fread(weights_buffer, 1, weights_size_bytes, f) != weights_size_bytes) {
        printf("[-] Error: Failed to read model weights.\n");
        fclose(f);
        return false;
    }
    fclose(f);
    printf("[+] Model weights loaded successfully.\n");

    // Assign weight pointers
    uint8_t* current_ptr = (uint8_t*)weights_buffer;
    wte = assign_param(current_ptr, vocab_size * embedding_dim, quant);
    wpe = assign_param(current_ptr, max_seq_len * embedding_dim, quant);

    blocks = new BlockWeights[num_layers];
    for (size_t i = 0; i < num_layers; ++i) {
        blocks[i].ln1_gamma = assign_param(current_ptr, embedding_dim, quant);
        blocks[i].ln1_beta = assign_param(current_ptr, embedding_dim, quant);
        blocks[i].w_qkv = assign_param(current_ptr, embedding_dim * 3 * embedding_dim, quant);
        blocks[i].b_qkv = assign_param(current_ptr, 3 * embedding_dim, quant);
        blocks[i].w_proj = assign_param(current_ptr, embedding_dim * embedding_dim, quant);
        blocks[i].b_proj = assign_param(current_ptr, embedding_dim, quant);
        blocks[i].ln2_gamma = assign_param(current_ptr, embedding_dim, quant);
        blocks[i].ln2_beta = assign_param(current_ptr, embedding_dim, quant);
        blocks[i].w_fc = assign_param(current_ptr, embedding_dim * 4 * embedding_dim, quant);
        blocks[i].b_fc = assign_param(current_ptr, 4 * embedding_dim, quant);
        blocks[i].w_proj_mlp = assign_param(current_ptr, 4 * embedding_dim * embedding_dim, quant);
        blocks[i].b_proj_mlp = assign_param(current_ptr, embedding_dim, quant);
    }
    ln_f_gamma = assign_param(current_ptr, embedding_dim, quant);
    ln_f_beta = assign_param(current_ptr, embedding_dim, quant);

    // Verify pointer arithmetic matches total bytes
    size_t processed_bytes = current_ptr - (uint8_t*)weights_buffer;
    if (processed_bytes != weights_size_bytes) {
        printf("[-] Error: Weight pointer offset mismatch! Offset diff: %zu, expected: %zu\n", 
               processed_bytes, weights_size_bytes);
        return false;
    }

    // Allocate memory for activations (Internal fast SRAM)
    x_buffer = (float*)malloc(max_seq_len * embedding_dim * sizeof(float));
    x2_buffer = (float*)malloc(max_seq_len * embedding_dim * sizeof(float));
    qkv_buffer = (float*)malloc(max_seq_len * 4 * embedding_dim * sizeof(float)); // Shared MLP/QKV buffer
    att_scores = (float*)malloc(max_seq_len * max_seq_len * sizeof(float));
    
    if (!x_buffer || !x2_buffer || !qkv_buffer || !att_scores) {
        printf("[-] Error: Failed to allocate activation buffers in internal memory.\n");
        return false;
    }
    printf("[+] Activation buffers allocated successfully (~308 KB in internal memory).\n");

    return true;
}

void GPTInference::forward(const int* input_tokens, int T, float* out_logits) {
    if (T > config.max_seq_len) {
        T = config.max_seq_len;
    }

    int C = config.embedding_dim;
    int V = config.vocab_size;
    int num_heads = config.num_heads;
    int head_dim = C / num_heads;
    float head_scale = 1.0f / sqrtf((float)head_dim);
    int quant = config.quantization_level;

    // 1. Embedding lookup: x = wte[token] + wpe[pos]
    for (int t = 0; t < T; ++t) {
        int token = input_tokens[t];
        float* x_row = x_buffer + t * C;
        
        // Lookup WTE
        if (quant == 0) {
            const float* wte_data = (const float*)wte.data;
            for (int c = 0; c < C; ++c) x_row[c] = wte_data[token * C + c];
        } else if (quant == 1) {
            const uint16_t* wte_data = (const uint16_t*)wte.data;
            for (int c = 0; c < C; ++c) x_row[c] = dequantize_bf16(wte_data[token * C + c]);
        } else {
            const int8_t* wte_data = (const int8_t*)wte.data;
            for (int c = 0; c < C; ++c) x_row[c] = wte_data[token * C + c] * wte.scale;
        }

        // Lookup WPE
        if (quant == 0) {
            const float* wpe_data = (const float*)wpe.data;
            for (int c = 0; c < C; ++c) x_row[c] += wpe_data[t * C + c];
        } else if (quant == 1) {
            const uint16_t* wpe_data = (const uint16_t*)wpe.data;
            for (int c = 0; c < C; ++c) x_row[c] += dequantize_bf16(wpe_data[t * C + c]);
        } else {
            const int8_t* wpe_data = (const int8_t*)wpe.data;
            for (int c = 0; c < C; ++c) x_row[c] += wpe_data[t * C + c] * wpe.scale;
        }
    }

    // 2. Process Blocks
    for (int l = 0; l < config.num_layers; ++l) {
        const BlockWeights& block = blocks[l];

        // LayerNorm 1
        layernorm(x_buffer, block.ln1_gamma, block.ln1_beta, x2_buffer, T, C, quant);

        // QKV Projection: x2_buffer (T x C) * w_qkv (C x 3C) + b_qkv (3C) -> qkv_buffer (T x 3C)
        matmul(x2_buffer, block.w_qkv, qkv_buffer, T, C, 3 * C, quant);
        for (int t = 0; t < T; ++t) {
            float* qkv_row = qkv_buffer + t * 3 * C;
            if (quant == 0) {
                const float* b_qkv = (const float*)block.b_qkv.data;
                for (int i = 0; i < 3 * C; ++i) qkv_row[i] += b_qkv[i];
            } else if (quant == 1) {
                const uint16_t* b_qkv = (const uint16_t*)block.b_qkv.data;
                for (int i = 0; i < 3 * C; ++i) qkv_row[i] += dequantize_bf16(b_qkv[i]);
            } else {
                const int8_t* b_qkv = (const int8_t*)block.b_qkv.data;
                for (int i = 0; i < 3 * C; ++i) qkv_row[i] += b_qkv[i] * block.b_qkv.scale;
            }
        }

        // Multi-head attention
        for (int h = 0; h < num_heads; ++h) {
            for (int q_t = 0; q_t < T; ++q_t) {
                const float* Q = qkv_buffer + q_t * 3 * C + h * head_dim;
                for (int k_t = 0; k_t <= q_t; ++k_t) {
                    const float* K = qkv_buffer + k_t * 3 * C + C + h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += Q[d] * K[d];
                    }
                    att_scores[q_t * T + k_t] = dot * head_scale;
                }
                softmax(att_scores + q_t * T, q_t + 1);
            }

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

        // Project attention output: x2_buffer (T x C) * w_proj (C x C) + b_proj -> qkv_buffer (T x C)
        matmul(x2_buffer, block.w_proj, qkv_buffer, T, C, C, quant);
        for (int t = 0; t < T; ++t) {
            float* x_row = x_buffer + t * C;
            const float* proj_row = qkv_buffer + t * C;
            if (quant == 0) {
                const float* b_proj = (const float*)block.b_proj.data;
                for (int c = 0; c < C; ++c) x_row[c] += proj_row[c] + b_proj[c];
            } else if (quant == 1) {
                const uint16_t* b_proj = (const uint16_t*)block.b_proj.data;
                for (int c = 0; c < C; ++c) x_row[c] += proj_row[c] + dequantize_bf16(b_proj[c]);
            } else {
                const int8_t* b_proj = (const int8_t*)block.b_proj.data;
                for (int c = 0; c < C; ++c) x_row[c] += proj_row[c] + b_proj[c] * block.b_proj.scale;
            }
        }

        // LayerNorm 2
        layernorm(x_buffer, block.ln2_gamma, block.ln2_beta, x2_buffer, T, C, quant);

        // MLP first linear layer: x2_buffer (T x C) * w_fc (C x 4C) + b_fc -> qkv_buffer (T x 4C)
        matmul(x2_buffer, block.w_fc, qkv_buffer, T, C, 4 * C, quant);
        for (int t = 0; t < T; ++t) {
            float* fc_row = qkv_buffer + t * 4 * C;
            if (quant == 0) {
                const float* b_fc = (const float*)block.b_fc.data;
                for (int i = 0; i < 4 * C; ++i) fc_row[i] += b_fc[i];
            } else if (quant == 1) {
                const uint16_t* b_fc = (const uint16_t*)block.b_fc.data;
                for (int i = 0; i < 4 * C; ++i) fc_row[i] += dequantize_bf16(b_fc[i]);
            } else {
                const int8_t* b_fc = (const int8_t*)block.b_fc.data;
                for (int i = 0; i < 4 * C; ++i) fc_row[i] += b_fc[i] * block.b_fc.scale;
            }
        }
        gelu(qkv_buffer, T * 4 * C);

        // MLP second linear layer: qkv_buffer (T x 4C) * w_proj_mlp (4C x C) + b_proj_mlp -> x2_buffer (T x C)
        matmul(qkv_buffer, block.w_proj_mlp, x2_buffer, T, 4 * C, C, quant);
        for (int t = 0; t < T; ++t) {
            float* x_row = x_buffer + t * C;
            const float* proj_row = x2_buffer + t * C;
            if (quant == 0) {
                const float* b_proj_mlp = (const float*)block.b_proj_mlp.data;
                for (int c = 0; c < C; ++c) x_row[c] += proj_row[c] + b_proj_mlp[c];
            } else if (quant == 1) {
                const uint16_t* b_proj_mlp = (const uint16_t*)block.b_proj_mlp.data;
                for (int c = 0; c < C; ++c) x_row[c] += proj_row[c] + dequantize_bf16(b_proj_mlp[c]);
            } else {
                const int8_t* b_proj_mlp = (const int8_t*)block.b_proj_mlp.data;
                for (int c = 0; c < C; ++c) x_row[c] += proj_row[c] + b_proj_mlp[c] * block.b_proj_mlp.scale;
            }
        }
    }

    // 3. Final LayerNorm for last token
    float* last_x_ln = x2_buffer;
    layernorm(x_buffer + (T - 1) * C, ln_f_gamma, ln_f_beta, last_x_ln, 1, C, quant);

    // 4. Classifier Head: logits = last_x_ln * wte^T
    matmul_transposed_b(last_x_ln, wte, out_logits, 1, C, V, quant);
}

void GPTInference::matmul(const float* A, const QuantizedTensor& B, float* C, int M, int K, int N, int quant_level) {
    memset(C, 0, M * N * sizeof(float));
    
    if (quant_level == 0) {
        const float* B_data = (const float*)B.data;
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float val = A[i * K + k];
                const float* b_row = B_data + k * N;
                float* c_row = C + i * N;
                for (int j = 0; j < N; ++j) {
                    c_row[j] += val * b_row[j];
                }
            }
        }
    } else if (quant_level == 1) {
        const uint16_t* B_data = (const uint16_t*)B.data;
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float val = A[i * K + k];
                const uint16_t* b_row = B_data + k * N;
                float* c_row = C + i * N;
                for (int j = 0; j < N; ++j) {
                    c_row[j] += val * dequantize_bf16(b_row[j]);
                }
            }
        }
    } else if (quant_level == 2) {
        const int8_t* B_data = (const int8_t*)B.data;
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float val = A[i * K + k];
                const int8_t* b_row = B_data + k * N;
                float* c_row = C + i * N;
                for (int j = 0; j < N; ++j) {
                    c_row[j] += val * b_row[j];
                }
            }
        }
        // Multiply entire output by scale factor
        float scale = B.scale;
        for (int i = 0; i < M * N; ++i) {
            C[i] *= scale;
        }
    }
}

void GPTInference::matmul_transposed_b(const float* A, const QuantizedTensor& B, float* C, int M, int K, int N, int quant_level) {
    if (quant_level == 0) {
        const float* B_data = (const float*)B.data;
        for (int i = 0; i < M; ++i) {
            const float* a_row = A + i * K;
            float* c_row = C + i * N;
            for (int j = 0; j < N; ++j) {
                const float* b_row = B_data + j * K;
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_row[k] * b_row[k];
                }
                c_row[j] = sum;
            }
        }
    } else if (quant_level == 1) {
        const uint16_t* B_data = (const uint16_t*)B.data;
        for (int i = 0; i < M; ++i) {
            const float* a_row = A + i * K;
            float* c_row = C + i * N;
            for (int j = 0; j < N; ++j) {
                const uint16_t* b_row = B_data + j * K;
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_row[k] * dequantize_bf16(b_row[k]);
                }
                c_row[j] = sum;
            }
        }
    } else if (quant_level == 2) {
        const int8_t* B_data = (const int8_t*)B.data;
        float scale = B.scale;
        for (int i = 0; i < M; ++i) {
            const float* a_row = A + i * K;
            float* c_row = C + i * N;
            for (int j = 0; j < N; ++j) {
                const int8_t* b_row = B_data + j * K;
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_row[k] * b_row[k];
                }
                c_row[j] = sum * scale;
            }
        }
    }
}

void GPTInference::layernorm(const float* x, const QuantizedTensor& gamma, const QuantizedTensor& beta, float* out, int T, int C, int quant_level) {
    for (int t = 0; t < T; ++t) {
        const float* x_row = x + t * C;
        float* out_row = out + t * C;

        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += x_row[c];
        mean /= C;

        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float diff = x_row[c] - mean;
            var += diff * diff;
        }
        var /= C;

        float std_dev = 1.0f / sqrtf(var + 1e-5f);
        
        if (quant_level == 0) {
            const float* g_data = (const float*)gamma.data;
            const float* b_data = (const float*)beta.data;
            for (int c = 0; c < C; ++c) {
                out_row[c] = (x_row[c] - mean) * std_dev * g_data[c] + b_data[c];
            }
        } else if (quant_level == 1) {
            const uint16_t* g_data = (const uint16_t*)gamma.data;
            const uint16_t* b_data = (const uint16_t*)beta.data;
            for (int c = 0; c < C; ++c) {
                out_row[c] = (x_row[c] - mean) * std_dev * dequantize_bf16(g_data[c]) + dequantize_bf16(b_data[c]);
            }
        } else if (quant_level == 2) {
            const int8_t* g_data = (const int8_t*)gamma.data;
            const int8_t* b_data = (const int8_t*)beta.data;
            float g_scale = gamma.scale;
            float b_scale = beta.scale;
            for (int c = 0; c < C; ++c) {
                out_row[c] = (x_row[c] - mean) * std_dev * (g_data[c] * g_scale) + (b_data[c] * b_scale);
            }
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

std::string GPTInference::decode_token(int token_id) const {
    if (token_id < 0 || token_id >= config.vocab_size) return "";
    if (config.tokenizer_type == 0) {
        return std::string(1, id_to_char[token_id]);
    } else {
        if (token_id < (int)bpe_vocab.size()) {
            return bpe_vocab[token_id];
        }
        return "";
    }
}
