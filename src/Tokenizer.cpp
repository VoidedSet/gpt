#include "Tokenizer.hpp"


bool Tokenizer::load_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[-] Error: Failed to open " << filepath << "\n";
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    raw_text = buffer.str();

    std::cout << "[+] Loaded " << filepath << " (" << raw_text.size() << " bytes)\n";
    return true;
}

void Tokenizer::build_vocab() {
    type = CHAR;
    std::set<char> unique_chars(raw_text.begin(), raw_text.end());
    vocab_size = unique_chars.size();

    id_to_char.assign(unique_chars.begin(), unique_chars.end());
    for (int i = 0; i < vocab_size; ++i) {
        char_to_id[id_to_char[i]] = i;
    }

    std::cout << "[+] Vocab size: " << vocab_size << " unique characters\n";
}

void Tokenizer::build_vocab_bpe(int target_vocab_size) {
    type = BPE;
    vocab_size = 256;
    
    // 1. Initialize base byte vocabulary
    id_to_token.clear();
    token_to_id.clear();
    for (int i = 0; i < 256; ++i) {
        std::string s(1, (char)i);
        id_to_token.push_back(s);
        token_to_id[s] = i;
    }
    
    // 2. Initialize tokens as individual bytes
    std::vector<int> current_tokens;
    current_tokens.reserve(raw_text.size());
    for (char c : raw_text) {
        current_tokens.push_back((unsigned char)c);
    }
    
    // 3. Iteratively find most frequent adjacent pair and merge it
    int num_merges = target_vocab_size - 256;
    merges.clear();
    
    std::cout << "[*] Training BPE tokenizer on text size of " << raw_text.size() << " bytes...\n";
    
    for (int m = 0; m < num_merges; ++m) {
        // Count frequencies of adjacent pairs
        std::unordered_map<std::pair<int, int>, int, pair_hash> pair_counts;
        for (size_t i = 0; i + 1 < current_tokens.size(); ++i) {
            std::pair<int, int> p = {current_tokens[i], current_tokens[i+1]};
            pair_counts[p]++;
        }
        
        // Find most frequent pair
        std::pair<int, int> best_pair = {-1, -1};
        int max_count = 0;
        for (auto const& [pair, count] : pair_counts) {
            if (count > max_count) {
                max_count = count;
                best_pair = pair;
            }
        }
        
        // If no pairs left, stop
        if (max_count == 0 || best_pair.first == -1) {
            std::cout << "[!] Stopping BPE training early after " << m << " merges.\n";
            break;
        }
        
        int new_token_id = 256 + m;
        merges.push_back(best_pair);
        
        // Reconstruct new token string representation
        std::string new_token_str = id_to_token[best_pair.first] + id_to_token[best_pair.second];
        id_to_token.push_back(new_token_str);
        token_to_id[new_token_str] = new_token_id;
        
        // Merge this pair in current_tokens list
        std::vector<int> merged_tokens;
        merged_tokens.reserve(current_tokens.size());
        for (size_t i = 0; i < current_tokens.size(); ++i) {
            if (i + 1 < current_tokens.size() && 
                current_tokens[i] == best_pair.first && 
                current_tokens[i+1] == best_pair.second) {
                merged_tokens.push_back(new_token_id);
                i++; // skip next element
            } else {
                merged_tokens.push_back(current_tokens[i]);
            }
        }
        current_tokens = std::move(merged_tokens);
        
        if (m % 50 == 0 || m == num_merges - 1) {
            std::cout << "  Merge " << m << "/" << num_merges 
                      << ": (" << best_pair.first << ", " << best_pair.second 
                      << ") -> " << new_token_id << " (freq: " << max_count << ")\n";
        }
    }
    
    vocab_size = id_to_token.size();
    std::cout << "[+] BPE Vocab trained successfully. Total vocab size: " << vocab_size << "\n";
}

void Tokenizer::encode() {
    if (type == CHAR) {
        tokens.clear();
        tokens.reserve(raw_text.size());
        for (char c : raw_text) {
            tokens.push_back(char_to_id[c]);
        }
        std::cout << "[+] Encoded " << tokens.size() << " character tokens.\n";
    } else {
        tokens.clear();
        tokens.reserve(raw_text.size());
        for (char c : raw_text) {
            tokens.push_back((unsigned char)c);
        }
        
        // Apply BPE merges in sequence
        for (size_t m = 0; m < merges.size(); ++m) {
            int left = merges[m].first;
            int right = merges[m].second;
            int merged = 256 + m;
            
            std::vector<int> next_tokens;
            next_tokens.reserve(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i + 1 < tokens.size() && tokens[i] == left && tokens[i+1] == right) {
                    next_tokens.push_back(merged);
                    i++; // skip next
                } else {
                    next_tokens.push_back(tokens[i]);
                }
            }
            tokens = std::move(next_tokens);
        }
        std::cout << "[+] Encoded " << tokens.size() << " BPE tokens.\n";
    }
}

std::vector<int> Tokenizer::encode_string(const std::string& text) const {
    if (type == CHAR) {
        std::vector<int> result;
        result.reserve(text.size());
        for (char c : text) {
            auto it = char_to_id.find(c);
            if (it != char_to_id.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(0); // Fallback
            }
        }
        return result;
    } else {
        std::vector<int> result;
        result.reserve(text.size());
        for (char c : text) {
            result.push_back((unsigned char)c);
        }
        
        for (size_t m = 0; m < merges.size(); ++m) {
            int left = merges[m].first;
            int right = merges[m].second;
            int merged = 256 + m;
            
            std::vector<int> next_result;
            next_result.reserve(result.size());
            for (size_t i = 0; i < result.size(); ++i) {
                if (i + 1 < result.size() && result[i] == left && result[i+1] == right) {
                    next_result.push_back(merged);
                    i++;
                } else {
                    next_result.push_back(result[i]);
                }
            }
            result = std::move(next_result);
        }
        return result;
    }
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string output;
    output.reserve(tokens.size());

    for (int token : tokens) {
        if (token < 0 || token >= vocab_size) continue;
        if (type == CHAR) {
            output += id_to_char[token];
        } else {
            output += id_to_token[token];
        }
    }

    return output;
}

void Tokenizer::print_sample_pair(int start_index, int block_size) {
    std::cout << "\n--- Sample Training Pair (block_size = " << block_size << ") ---\n";
    std::cout << "Input  (x): ";
    for (int i = 0; i < block_size; ++i) {
        std::cout << tokens[start_index + i] << " ";
    }
    std::cout << "\nTarget (y): ";
    for (int i = 0; i < block_size; ++i) {
        std::cout << tokens[start_index + i + 1] << " ";
    }
    std::cout << "\n-----------------------------------------\n";
}
