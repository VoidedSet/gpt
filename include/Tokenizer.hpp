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

using namespace std;

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
        void print_sample_pair(int start_index, int block_size);

        const string& get_raw_text(){
            return raw_text;
        }
};

#endif