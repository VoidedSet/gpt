cmake_minimum_required(VERSION 3.20)
project(Engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -fopenmp)
endif()

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(engine ${SOURCES}
    include/Tokenizer.hpp
    include/DataLoader.hpp
    src/Embedding.cpp
    include/BigramModel.hpp
    include/Embedding.hpp
    src/BigramModel.cpp
    include/NastyTensors.hpp
    include/LayerNorm.hpp
    include/CausalSelfAttention.hpp
    include/FeedForward.hpp
    include/Block.hpp
    include/GPT.hpp
    include/Optimizer.hpp
    src/CudaKernels.cu
    include/CudaKernels.hpp)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(engine PRIVATE OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)