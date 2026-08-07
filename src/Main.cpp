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
    // Load complete Works of Shakespeare if available, fallback to tiny shakespeare
    if (!tokenizer.load_file("dataset/complete_shakespeare.txt")) {
        if (!tokenizer.load_file("dataset/input.txt"))
            return 1;
    }

    int target_vocab_size = 512; // BPE vocabulary size
    tokenizer.build_vocab_bpe(target_vocab_size);
    tokenizer.encode();

    size_t B_train = 16;
    size_t T_train = 64;
    DataLoader loader(tokenizer.get_tokens(), B_train, T_train);

    size_t vocab_size = tokenizer.get_vocab_size();
    size_t max_seq_len = 64;
    size_t embedding_dim = 256;
    size_t num_heads = 8;
    size_t num_layers = 6;

    std::cout << "Creating GPT Model (vocab_size=" << vocab_size 
              << ", layers=" << num_layers << ", dim=" << embedding_dim << ")...\n";
    GPT gpt_model(vocab_size, max_seq_len, embedding_dim, num_heads, num_layers);

    std::vector<NastyTensors*> params = gpt_model.get_parameters();
    std::cout << "Number of learnable parameters: " << params.size() << "\n";

    // Move all model parameters to the GPU
    for (auto* param : params) {
        param->to_gpu(DATA_);
    }

    AdamW optimizer(params, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.01f);

    std::vector<int> X_train, Y_train;
    
    srand(1337);

    std::cout << "\n--- Generating with untrained model ---\n";
    std::vector<int> prompt = tokenizer.encode_string("The ");
    std::vector<int> gen_tokens = gpt_model.generate(prompt, 100);
    std::cout << tokenizer.decode(gen_tokens) << "\n";
    std::cout << "---------------------------------------\n\n";

    std::cout << "[*] Training starting...\n";
    auto train_start = chrono::high_resolution_clock::now();
    
    int total_steps = 10000;
    for (int step = 0; step < total_steps; ++step) {
        auto step_start = chrono::high_resolution_clock::now();
        
        loader.get_batch(X_train, Y_train);
        
        optimizer.zero_grad();

        NastyTensors train_logits = gpt_model.forward(X_train, B_train, T_train);

        float loss = gpt_model.backward(Y_train);

        optimizer.step();

        auto step_end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> step_dur = step_end - step_start;
        double step_ms = step_dur.count();
        double tokens_per_sec = (B_train * T_train) / (step_ms / 1000.0);

        if (step % 50 == 0 || step == total_steps - 1) {
            std::cout << "  Step " << step << " | Loss: " << loss 
                      << " | Speed: " << step_ms << " ms/step (" << tokens_per_sec << " tok/sec)\n";
        }

        if ((step > 0 && step % 300 == 0) || step == total_steps - 1) {
            std::cout << "\n  --- [Step " << step << "] Intermediate Generation snippet ---\n";
            std::vector<int> intermediate_gen = gpt_model.generate(prompt, 80);
            std::cout << tokenizer.decode(intermediate_gen) << "\n";
            std::cout << "  -----------------------------------------------------\n\n";
        }
    }
    
    auto train_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_dur = train_end - train_start;
    std::cout << "\n[+] Training completed in " << total_dur.count() << " seconds.\n";

    std::cout << "\n--- Generating with final trained model ---\n";
    gen_tokens = gpt_model.generate(prompt, 200);
    std::cout << tokenizer.decode(gen_tokens) << "\n";
    std::cout << "-------------------------------------\n";
    
    gpt_model.save_binary("dataset/macbeth2.bin", tokenizer, 2); // 2 = INT8 Quantization
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    std::cout << "\n[+] Main Completed in " << elapsed.count() << " seconds.\n";

    return 0;
}