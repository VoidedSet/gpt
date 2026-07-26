#include "Tokenizer.hpp"
#include "DataLoader.hpp"
#include "BigramModel.hpp"
#include "NastyTensors.hpp"
#include "Embedding.hpp"
#include "LayerNorm.hpp"

#include <chrono>
#include <iostream>

using namespace std;

int main() {

    std::cout << "[*] Testing Embedding Layer...\n";
    Embedding emb(10, 8, 4);
    std::vector<int> test_tokens = {1, 3, 2, 4, 0, 5}; // B=2, T=3
    NastyTensors emb_out = emb.forward(test_tokens, 2, 3);
    std::cout << "Embedding Output shape: [" 
              << emb_out.shape()[0] << ", " 
              << emb_out.shape()[1] << ", " 
              << emb_out.shape()[2] << "]\n";
    emb_out.print();

    std::cout << "Checking manual sum at (0, 0, 0):\n";
    std::cout << "  wte(1, 0) = " << emb.wte()(1, 0) << "\n";
    std::cout << "  wpe(0, 0) = " << emb.wpe()(0, 0) << "\n";
    std::cout << "  sum       = " << emb.wte()(1, 0) + emb.wpe()(0, 0) << "\n";
    std::cout << "  emb_out   = " << emb_out(0, 0, 0) << "\n";
    std::cout << "-----------------------------------\n\n";

    std::cout << "[*] Testing LayerNorm (In-place)...\n";
    LayerNorm ln(4);
    ln.forward(emb_out);
    std::cout << "Normalized Embedding Output:\n";
    emb_out.print();

    std::cout << "Verifying mean and variance for Batch 0:\n";
    for (size_t t = 0; t < 3; ++t) {
        float sum = 0.0f;
        for (size_t c = 0; c < 4; ++c) {
            sum += emb_out(0, t, c);
        }
        float mean = sum / 4.0f;

        float sum_sq_diff = 0.0f;
        for (size_t c = 0; c < 4; ++c) {
            float diff = emb_out(0, t, c) - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / 4.0f;

        std::cout << "  t = " << t 
                  << " | Mean = " << (std::abs(mean) < 1e-5f ? 0.0f : mean)
                  << " | Var = " << var << "\n";
    }
    std::cout << "-----------------------------------\n\n";


    auto start = chrono::high_resolution_clock::now();

    Tokenizer tokenizer;
    if (!tokenizer.load_file("dataset/input.txt"))
        return 1;

    tokenizer.build_vocab();
    tokenizer.encode();

    int B = 32;
    int T = 8;
    DataLoader loader(tokenizer.get_tokens(), B, T);
    BigramModel model(tokenizer.get_vocab_size());

    std::vector<int> X, Y;
    loader.get_batch(X, Y);

    float initial_loss = model.compute_loss(X, Y);
    std::cout << "\n[+] Initial Untrained Loss: " << initial_loss 
              << " (Theoretical Random Target: ~4.17)\n";

    std::cout << "[*] Training Bigram Model...\n";
    float lr = 1.0f;
    for (int step = 0; step < 2500; ++step) {
        loader.get_batch(X, Y);
        model.train_step(X, Y, lr);

        if ((step + 1) % 200 == 0) {
            float loss = model.compute_loss(X, Y);
            std::cout << "    Step " << (step + 1) << " | Loss: " << loss << "\n";
        }
    }

    std::cout << "\n[+] Generating sample output after Bigram training:" << endl;
    std::vector<int> sample_tokens = model.generate(64, 1000);
 
    std::cout << tokenizer.decode(sample_tokens);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, micro> elapsed = end - start;

    std::cout << "\n[+] Bigram Baseline Complete in " << elapsed.count() / 1000 << "ms! \n";

    return 0;
}