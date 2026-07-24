#include "Embedding.hpp"

void Embedding::forward(const std::vector<int>& input_tokens, std::vector<float>& out_data){
        int N = input_tokens.size();
        out_data.resize(N * embedding_dim);

        for(int i = 0; i < N; i++){
            int token_id = input_tokens[i];

            const float* src = &weights[token_id * embedding_dim];
            float* dest = &out_data[i * embedding_dim];

            for(int c = 0; c < embedding_dim; ++c)
                dest[c] = src[c];
        }
    }