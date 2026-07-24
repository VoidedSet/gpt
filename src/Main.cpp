#include "Tokenizer.hpp"
#include "DataLoader.hpp"
#include "BigramModel.hpp"
#include "NastyTensors.hpp"
#include "Ops.hpp"
#include "GPT.hpp"
#include "Optimizer.hpp"

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

    std::cout << "[*] Testing MatMul (C = A * B)...\n";
    NastyTensors test_A({2, 3});
    test_A(0, 0) = 1.0f; test_A(0, 1) = 2.0f; test_A(0, 2) = 3.0f;
    test_A(1, 0) = 4.0f; test_A(1, 1) = 5.0f; test_A(1, 2) = 6.0f;

    NastyTensors test_B({3, 2});
    test_B(0, 0) = 7.0f;  test_B(0, 1) = 8.0f;
    test_B(1, 0) = 9.0f;  test_B(1, 1) = 10.0f;
    test_B(2, 0) = 11.0f; test_B(2, 1) = 12.0f;

    NastyTensors test_C({2, 2});
    ops::matmul(test_A, test_B, test_C);
    std::cout << "Result C (expected [[58.0000, 64.0000], [139.0000, 154.0000]]):\n";
    test_C.print();
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

    std::cout << "\n===================================\n";
    std::cout << "[*] Starting GPT Engine...\n";
    std::cout << "===================================\n";

    int gpt_B = 16;
    int gpt_T = 64;
    DataLoader gpt_loader(tokenizer.get_tokens(), gpt_B, gpt_T);
    GPT gpt_model(tokenizer.get_vocab_size(), 256, 128, 4, 4);
    AdamW opt(1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);

    std::vector<int> X_gpt, Y_gpt;
    gpt_loader.get_batch(X_gpt, Y_gpt);
    float gpt_initial_loss = gpt_model.forward(X_gpt, Y_gpt, gpt_B, gpt_T);
    std::cout << "[+] Initial GPT Untrained Loss: " << gpt_initial_loss << "\n";

    std::cout << "[*] Training GPT Model (2000 steps)...\n";
    for (int step = 0; step < 2000; ++step) {
        gpt_loader.get_batch(X_gpt, Y_gpt);
        opt.zero_grad(gpt_model.get_parameters());
        float loss = gpt_model.forward(X_gpt, Y_gpt, gpt_B, gpt_T);
        gpt_model.backward(X_gpt, Y_gpt, gpt_B, gpt_T);
        opt.update(gpt_model.get_parameters());

        if ((step + 1) % 50 == 0) {
            std::cout << "    Step " << (step + 1) << " | Loss: " << loss << "\n";
        }
    }

    std::cout << "\n[+] Generating sample output from trained GPT model:\n";
    std::mt19937 gpt_rng(1337);
    std::vector<int> gpt_sample = gpt_model.generate(X_gpt[0], 300, gpt_rng);
    std::cout << "-----------------------------------\n";
    std::cout << tokenizer.decode(gpt_sample) << "\n";
    std::cout << "-----------------------------------\n";
    std::cout << "[+] GPT Engine Complete!\n";

    return 0;
}