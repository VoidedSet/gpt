
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Tokenizer.hpp"

using namespace std;

int main() {

    Tokenizer tokenizer;

    std::cout << "[*] Starting execution...\n";

    // timerrr
    auto start = std::chrono::high_resolution_clock::now();
    
    if(tokenizer.load_file("dataset/input.txt")){
        cout << tokenizer.get_raw_text().substr(0, 100) << endl;
      
        tokenizer.build_vocab();
        tokenizer.encode();
        
        vector<int> tokens = tokenizer.get_tokens();

        cout << endl << "First 20 tokens" << endl;
        for(int i = 0; i < 20; i++)
            cout << tokens[i] << ", ";

        cout << endl << "[-] Decoding Sample" << endl;
        std::cout << tokenizer.decode({18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 0, 14, 43, 44, 53, 56}) << endl;
        
        tokenizer.print_sample_pair(0, 8);
    }

    //end of exec

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[+] Completed in " << elapsed.count() << " us.\n";

    return 0;
}