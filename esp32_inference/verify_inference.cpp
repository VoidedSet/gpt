#include "GPTInference.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>

int main() {
    srand(1337);
    GPTInference model;
    
    std::cout << "[*] Loading model...\n";
    if (!model.load_model("../dataset/macbeth.bin")) {
        std::cerr << "[-] Error loading model.\n";
        return 1;
    }
    
    std::cout << "[+] Model loaded successfully!\n";
    
    // Start generation from a prompt
    std::string prompt = "The ";
    std::vector<int> tokens;
    for (char c : prompt) {
        int token_id = -1;
        for (int i = 0; i < model.config.vocab_size; ++i) {
            if (model.id_to_char[i] == c) {
                token_id = i;
                break;
            }
        }
        if (token_id == -1) {
            token_id = 0;
        }
        tokens.push_back(token_id);
    }
    
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Tokens: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n\n--- Generating ---\n";
    
    std::string generated_text = prompt;
    std::vector<float> logits(model.config.vocab_size);
    for (int i = 0; i < 150; ++i) {
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
        
        generated_text += model.id_to_char[next_token];
        
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
