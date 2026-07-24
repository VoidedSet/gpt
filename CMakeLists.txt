cmake_minimum_required(VERSION 3.20)
project(Engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(engine ${SOURCES}
    include/Tokenizer.hpp
    include/DataLoader.hpp
    src/Embedding.cpp
    include/BigramModel.hpp)