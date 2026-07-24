#include "Tokenizer.hpp"
#include "DataLoader.hpp"
#include "BigramModel.hpp"
#include "NastyTensors.hpp"

using namespace std;

int main() {
    std::cout << "[*] Testing NastyTensors...\n";
    NastyTensors t({2, 3}, 1.5f);
    t(0, 1) = 9.9f;
    t(1, 2) = -4.2f;
    std::cout << "Original 2D tensor:\n";
    t.print();

    std::cout << "\nSlice at index 1:\n";
    NastyTensors slice1 = t.slice(1);
    slice1.print();
    std::cout << "slice1(2) (should be -4.2000): " << slice1(2) << "\n";

    std::cout << "\nModifying slice1(0) = 88.8...\n";
    slice1(0) = 88.8f;
    std::cout << "Original tensor after modifying slice (should show 88.8000 at row 1, col 0):\n";
    t.print();
    std::cout << "-----------------------------------\n\n";

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