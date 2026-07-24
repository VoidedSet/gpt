#include <iostream>
#include "Tokenizer.hpp"
#include "DataLoader.hpp"
#include "BigramModel.hpp"

using namespace std;

int main() {
    Tokenizer tokenizer;
    if (!tokenizer.load_file("dataset/input.txt")) {
        return 1;
    }

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
    // for (int id : sample_tokens) {
    //     std::cout << id << " ";
    // }

    std::cout << tokenizer.decode(sample_tokens);
    
    std::cout << "\n[+] Bigram Baseline Complete!\n";

    return 0;
}