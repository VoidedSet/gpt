#include "GPTInference.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Self-contained BPE encoder helper using model merges
std::vector<int> bpe_encode(const std::string& text, int num_merges, const int* merge_left, const int* merge_right) {
    std::vector<int> result;
    result.reserve(text.size());
    for (char c : text) {
        result.push_back((unsigned char)c);
    }
    for (int m = 0; m < num_merges; ++m) {
        int left = merge_left[m];
        int right = merge_right[m];
        int merged = 256 + m;
        std::vector<int> next_res;
        next_res.reserve(result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            if (i + 1 < result.size() && result[i] == left && result[i+1] == right) {
                next_res.push_back(merged);
                i++;
            } else {
                next_res.push_back(result[i]);
            }
        }
        result = std::move(next_res);
    }
    return result;
}

int main() {
    srand(1337);
    GPTInference model;
    
    std::cout << "[*] Loading model...\n";
    if (!model.load_model("../dataset/macbeth2.bin")) {
        std::cerr << "[-] Error loading model.\n";
        return 1;
    }
    
    std::cout << "[+] Model loaded successfully!\n";
    
    // Start generation from a prompt
    std::string prompt = "The ";
    std::vector<int> tokens;
    
    if (model.config.tokenizer_type == 0) {
        // CHAR mode
        for (char c : prompt) {
            int token_id = -1;
            for (int i = 0; i < model.config.vocab_size; ++i) {
                if (model.id_to_char[i] == c) {
                    token_id = i;
                    break;
                }
            }
            if (token_id == -1) token_id = 0;
            tokens.push_back(token_id);
        }
    } else {
        // BPE mode
        tokens = bpe_encode(prompt, model.num_merges, model.merge_left, model.merge_right);
    }
    
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Tokens: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n\n--- Generating ---\n";
    
    std::string generated_text = prompt;
    std::vector<float> logits(model.config.vocab_size);
    for (int i = 0; i < 80; ++i) {
        model.forward(tokens.data(), tokens.size(), logits.data());
        
        float max_logit = logits[0];
        for (int j = 1; j < model.config.vocab_size; ++j) {
            if (logits[j] > max_logit) max_logit = logits[j];
        }
        
        std::vector<float> probs(model.config.vocab_size);
        float sum_exp = 0.0f;
        for (int j = 0; j < model.config.vocab_size; ++j) {
            probs[j] = std::exp(logits[j] - max_logit);
            sum_exp += probs[j];
        }
        
        float r = ((float)rand() / RAND_MAX) * sum_exp;
        float running_sum = 0.0f;
        int next_token = 0;
        for (int j = 0; j < model.config.vocab_size; ++j) {
            running_sum += probs[j];
            if (r <= running_sum) {
                next_token = j;
                break;
            }
        }
        
        std::string next_str = model.decode_token(next_token);
        std::cout << next_str << std::flush;
        generated_text += next_str;
        
        tokens.push_back(next_token);
        if (tokens.size() > (size_t)model.config.max_seq_len) {
            tokens.erase(tokens.begin());
        }
    }
    
    std::cout << "\n------------------\n";
    std::cout << "Final Generated Text:\n" << generated_text << "\n";
    std::cout << "------------------\n";
    
    return 0;
}
