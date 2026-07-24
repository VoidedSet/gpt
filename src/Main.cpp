#include "Core.hpp"
#include <chrono>
#include <iostream>

using namespace std;

int main() {
    Engine::System sys;
    sys.initialize();

    std::cout << "[*] Starting execution...\n";

    // High-precision timing harness
    auto start = std::chrono::high_resolution_clock::now();

    // --- YOUR HARDWARE / SYSTEM LOGIC HERE ---
    sys.process();
    
    // ----------------------------------------

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[+] Completed in " << elapsed.count() << " us.\n";

    return 0;
}