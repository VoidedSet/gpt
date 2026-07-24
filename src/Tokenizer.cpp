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
    std::set<char> unique_chars(raw_text.begin(), raw_text.end());
    vocab_size = unique_chars.size();

    id_to_char.assign(unique_chars.begin(), unique_chars.end());
    for (int i = 0; i < vocab_size; ++i) {
        char_to_id[id_to_char[i]] = i;
    }

    std::cout << "[+] Vocab size: " << vocab_size << " unique characters\n";
}

void Tokenizer::encode() {
    tokens.reserve(raw_text.size());
    for (char c : raw_text) {
        tokens.push_back(char_to_id[c]);
    }
    std::cout << "[+] Encoded " << tokens.size() << " tokens.\n";
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
