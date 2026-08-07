#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cassert>

enum TokenizerType {
    CHAR = 0,
    BPE = 1
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return (std::size_t)pair.first * 131071 + pair.second;
    }
};

class Tokenizer {
private:
    std::string raw_text;
    
    // Character Tokenizer State
    std::vector<char> id_to_char;
    std::unordered_map<char, int> char_to_id;
    
    // BPE Tokenizer State
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::pair<int, int>> merges;

    std::vector<int> tokens;
    int vocab_size = 0;
    TokenizerType type = CHAR;

public:
    bool load_file(const std::string& filepath);
    void build_vocab(); // Character vocab builder
    void build_vocab_bpe(int target_vocab_size); // BPE vocab builder
    
    void encode(); // Default encode based on type
    std::string decode(const std::vector<int>& tokens) const;
    
    std::vector<int> encode_string(const std::string& text) const;

    void print_sample_pair(int start_index, int block_size);

    const std::string& get_raw_text() const { return raw_text; }
    const std::vector<int>& get_tokens() const { return tokens; }
    int get_vocab_size() const { return vocab_size; }
    
    TokenizerType get_type() const { return type; }
    void set_type(TokenizerType t) { type = t; }
    
    // Serialization Getters
    const std::vector<char>& get_id_to_char() const { return id_to_char; }
    const std::vector<std::string>& get_id_to_token() const { return id_to_token; }
    const std::vector<std::pair<int, int>>& get_merges() const { return merges; }

    // Deserialization Setters
    void load_bpe_vocab(const std::vector<std::string>& vocab, const std::vector<std::pair<int, int>>& merge_rules) {
        type = BPE;
        id_to_token = vocab;
        merges = merge_rules;
        vocab_size = vocab.size();
        token_to_id.clear();
        for (size_t i = 0; i < vocab.size(); ++i) {
            token_to_id[vocab[i]] = i;
        }
    }
    
    void load_char_vocab(const std::vector<char>& vocab) {
        type = CHAR;
        id_to_char = vocab;
        vocab_size = vocab.size();
        char_to_id.clear();
        for (size_t i = 0; i < vocab.size(); ++i) {
            char_to_id[vocab[i]] = i;
        }
    }

    int char_to_token(char c) const {
        if (type == CHAR) {
            auto it = char_to_id.find(c);
            if (it != char_to_id.end()) return it->second;
            return 0;
        } else {
            // For BPE, bytes 0-255 are mapped to IDs 0-255
            return (unsigned char)c;
        }
    }
};

#endif