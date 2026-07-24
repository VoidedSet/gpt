#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <vector>
#include <random>
#include <iostream>
#include <cassert>

class DataLoader {
    private:
        const std::vector<int>& tokens;
        int B;
        int T;
        std::mt19937 rng;

    public:
        DataLoader(const std::vector<int>& dataset_tokens, int batch_size, int block_size)
         : tokens(dataset_tokens), B(batch_size), T(block_size), rng(1337) {
            assert(tokens.size() > static_cast<size_t>(B * T) && "Dataset too small for batch configuration!");
        }

        void get_batch(std::vector<int>& x_batch, std::vector<int>& y_batch){
            x_batch.resize(B * T);
            y_batch.resize(B * T);

            int max_start = tokens.size() - T - 1;
            std::uniform_int_distribution<int> dist(0, max_start);

            for(int b = 0; b < B; b++){
                int start_idx = dist(rng);
                for(int t = 0; t < T; t++){
                    int flat_index = b * T + t;
                    x_batch[flat_index] = tokens[start_idx + t];
                    y_batch[flat_index] = tokens[start_idx + t  + 1];
                }
            }
        }

        int get_batch_size() const { return B; }
        int get_block_size() const { return T; }
};

#endif