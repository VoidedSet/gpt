#include "Tokenizer.hpp"
#include "DataLoader.hpp"
#include "BigramModel.hpp"
#include "NastyTensors.hpp"
#include "Embedding.hpp"
#include "LayerNorm.hpp"
#include "CausalSelfAttention.hpp"
#include "FeedForward.hpp"
#include "Block.hpp"
#include "GPT.hpp"
#include "Optimizer.hpp"

#include <chrono>
#include <iostream>

using namespace std;

int main() {

    auto start = chrono::high_resolution_clock::now();

    std::cout << "[*] Running GPT Training Loop...\n";
    Tokenizer tokenizer;
    if (!tokenizer.load_file("dataset/input.txt"))
        return 1;

    tokenizer.build_vocab();
    tokenizer.encode();

    size_t B_train = 4;
    size_t T_train = 8;
    DataLoader loader(tokenizer.get_tokens(), B_train, T_train);

    size_t vocab_size = tokenizer.get_vocab_size();
    size_t max_seq_len = 16;
    size_t embedding_dim = 16;
    size_t num_heads = 2;
    size_t num_layers = 1;

    std::cout << "Creating GPT Model (vocab_size=" << vocab_size 
              << ", layers=" << num_layers << ", dim=" << embedding_dim << ")...\n";
    GPT gpt_model(vocab_size, max_seq_len, embedding_dim, num_heads, num_layers);

    std::vector<NastyTensors*> params = gpt_model.get_parameters();
    std::cout << "Number of learnable parameters: " << params.size() << "\n";

    AdamW optimizer(params, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.01f);

    std::vector<int> X_train, Y_train;
    for (int step = 0; step < 20; ++step) {
        loader.get_batch(X_train, Y_train);
        
        optimizer.zero_grad();

        NastyTensors train_logits = gpt_model.forward(X_train, B_train, T_train);

        float loss = gpt_model.backward(Y_train);

        optimizer.step();

        if (step % 2 == 0) {
            std::cout << "  Step " << step << " | Loss: " << loss << "\n";
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, micro> elapsed = end - start;

    std::cout << "\n[+] Execution Complete in " << elapsed.count() / 1000 << "ms! \n";

    return 0;
}