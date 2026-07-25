#include "Tokenizer.hpp"
#include "DataLoader.hpp"
#include "BigramModel.hpp"
#include "NastyTensors.hpp"

using namespace std;

int main() {

    std::cout << "[*] testing matmul \n";
    NastyTensors A({2, 2});
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    NastyTensors b({2, 2});
    b(0, 0) = 5.0f; b(0, 1) = 6.0f;
    b(1, 0) = 7.0f; b(1, 1) = 8.0f;

    std::cout << "Matrix A with stride: \n";
    A.print();

    std::cout << "\nMatrix B with: \n";
    b.print();

    std::cout << "\n C = A.matmul(B)\n";
    NastyTensors C = A.matmul(b);
    C.print();


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