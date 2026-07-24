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

class Tokenizer {
private:
    std::string raw_text;
    std::vector<char> id_to_char;
    std::unordered_map<char, int> char_to_id;
    std::vector<int> tokens;
    int vocab_size = 0;

public:
    bool load_file(const std::string& filepath);
    void build_vocab();
    void encode();
    std::string decode(const std::vector<int>& tokens);
    
    void print_sample_pair(int start_index, int block_size);

    const std::string& get_raw_text() const { return raw_text; }
    const std::vector<int>& get_tokens() const { return tokens; }
    int get_vocab_size() const { return vocab_size; }
};

#endif