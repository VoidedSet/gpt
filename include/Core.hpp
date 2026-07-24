#ifndef CORE_HPP
#define CORE_HPP

#include <iostream>
#include <vector>
#include <memory>

namespace Engine {

class System {
public:
    System();
    ~System();

    void initialize();
    void process();
};

}

#endif