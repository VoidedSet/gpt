#include "Core.hpp"

namespace Engine {

System::System() {
    // Allocation / setup logic
}

System::~System() {
    // Cleanup
}

void System::initialize() {
    std::cout << "[+] System initialized.\n";
}

void System::process() {
    // Raw execution loop goes here
}

} // namespace Engine