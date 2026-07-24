#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include <vector>
#include <random>
#include <cmath>

class Embedding{
    private:
        int num_embeddings, // vocab or block size
            embedding_dim;  // n_embd or C -> channel
        std::vector<float> weights; // num_embd * embd_dim

    public:
        Embedding(int num_embeddings, int embedding_dim)
            : num_embeddings(num_embeddings), embedding_dim(embedding_dim){
            weights.resize(num_embeddings * embedding_dim);

            std::mt19937 gen(1337);
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for(auto& w : weights)
                w = dist(gen);   
        }
        void forward(const std::vector<int>& input_tokens, std::vector<float>& out_data);

        const std::vector<float>& get_weights() const { return weights; }
};

#endif