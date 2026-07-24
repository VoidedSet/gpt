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

        void forward(const std::vector<int>& input_tokens, std::vector<float>& out_data){
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

        const std::vector<float>& get_weights() const { return weights; }
};