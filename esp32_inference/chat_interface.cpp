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
    srand(time(nullptr));
    GPTInference model;
    
    std::cout << "[*] Loading model weights from data/macbeth.bin...\n";
    if (!model.load_model("data/macbeth.bin")) {
        std::cerr << "[-] Error: Failed to open model weights file: data/macbeth.bin\n";
        std::cerr << "    Make sure to run this tool from within the 'esp32_inference' directory.\n";
        return 1;
    }
    
    std::cout << "[+] Model loaded successfully!\n\n";
    std::cout << "=========================================================\n";
    std::cout << "       Macbeth GPT Interactive Chat (BPE + INT8)        \n";
    std::cout << "=========================================================\n";
    std::cout << "Type a character name or prompt to chat with the model.\n";
    std::cout << "Type 'exit' or 'quit' to close the interface.\n\n";

    std::vector<int> context_tokens;

    while (true) {
        std::cout << "\nYou: ";
        std::string user_input;
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "exit" || user_input == "quit") break;
        if (user_input.empty()) continue;

        // Format user input as dialogue structure (ensure it ends with a newline)
        if (user_input.back() != '\n') {
            user_input += "\n";
        }

        // BPE Encode the user input
        std::vector<int> input_tokens;
        if (model.config.tokenizer_type == 0) {
            for (char c : user_input) {
                int token_id = -1;
                for (int i = 0; i < model.config.vocab_size; ++i) {
                    if (model.id_to_char[i] == c) {
                        token_id = i;
                        break;
                    }
                }
                if (token_id == -1) token_id = 0;
                input_tokens.push_back(token_id);
            }
        } else {
            input_tokens = bpe_encode(user_input, model.num_merges, model.merge_left, model.merge_right);
        }

        // Append to context window
        context_tokens.insert(context_tokens.end(), input_tokens.begin(), input_tokens.end());
        
        // Truncate context if it exceeds model context length limits
        if (context_tokens.size() > (size_t)model.config.max_seq_len) {
            context_tokens.erase(context_tokens.begin(), context_tokens.end() - model.config.max_seq_len);
        }

        std::cout << "GPT: " << std::flush;

        // Generate response tokens
        int max_response_tokens = 120;
        std::vector<float> logits(model.config.vocab_size);
        float temperature = 0.85f;

        for (int i = 0; i < max_response_tokens; ++i) {
            model.forward(context_tokens.data(), context_tokens.size(), logits.data());
            
            float max_logit = logits[0];
            for (int j = 1; j < model.config.vocab_size; ++j) {
                if (logits[j] > max_logit) max_logit = logits[j];
            }
            
            std::vector<float> probs(model.config.vocab_size);
            float sum_exp = 0.0f;
            for (int j = 0; j < model.config.vocab_size; ++j) {
                probs[j] = std::exp((logits[j] - max_logit) / temperature);
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

            context_tokens.push_back(next_token);
            if (context_tokens.size() > (size_t)model.config.max_seq_len) {
                context_tokens.erase(context_tokens.begin());
            }

            // Stop generating early if the model creates a double newline (indicating play turn change)
            if (next_str == "\n\n") {
                break;
            }
        }
        std::cout << "\n";
    }

    return 0;
}
