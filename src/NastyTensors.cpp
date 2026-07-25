#include "NastyTensors.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>

std::vector<size_t> NastyTensors::calculate_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

NastyTensors::NastyTensors() 
    : data_(nullptr), offset_(0), shape_({}), strides_({}) {}

NastyTensors::NastyTensors(const std::vector<size_t>& shape) 
    : shape_(shape), strides_(calculate_strides(shape)), offset_(0) {
    size_t total_size = size();
    data_ = std::make_shared<std::vector<float>>(total_size, 0.0f);
}

NastyTensors::NastyTensors(const std::vector<size_t>& shape, float fill_value) 
    : shape_(shape), strides_(calculate_strides(shape)), offset_(0) {
    size_t total_size = size();
    data_ = std::make_shared<std::vector<float>>(total_size, fill_value);
}

NastyTensors::NastyTensors(const std::vector<size_t>& shape, 
                           std::shared_ptr<std::vector<float>> data, 
                           size_t offset) 
    : shape_(shape), strides_(calculate_strides(shape)), data_(data), offset_(offset) {}

NastyTensors NastyTensors::clone() const {
    NastyTensors cloned(shape_);
    float* dest = cloned.data();
    const float* src = this->data();
    size_t num_elements = this->size();
    if (src && dest) {
        for (size_t i = 0; i < num_elements; ++i) {
            dest[i] = src[i];
        }
    }
    return cloned;
}

NastyTensors NastyTensors::slice(size_t index) const {
    if (shape_.empty()) {
        throw std::runtime_error("Cannot slice a 0-dimensional tensor.");
    }
    if (index >= shape_[0]) {
        throw std::out_of_range("Slice index out of bounds.");
    }
    
    std::vector<size_t> new_shape(shape_.begin() + 1, shape_.end());
    size_t new_offset = offset_ + index * strides_[0];
    
    return NastyTensors(new_shape, data_, new_offset);
}

size_t NastyTensors::size() const {
    if (shape_.empty()) return data_ ? 1 : 0;
    size_t total = 1;
    for (size_t dim : shape_) {
        total *= dim;
    }
    return total;
}

float& NastyTensors::operator()(size_t i) {
    return (*data_)[offset_ + i * strides_[0]];
}

const float& NastyTensors::operator()(size_t i) const {
    return (*data_)[offset_ + i * strides_[0]];
}

float& NastyTensors::operator()(size_t i, size_t j) {
    return (*data_)[offset_ + i * strides_[0] + j * strides_[1]];
}

const float& NastyTensors::operator()(size_t i, size_t j) const {
    return (*data_)[offset_ + i * strides_[0] + j * strides_[1]];
}

float& NastyTensors::operator()(size_t i, size_t j, size_t k) {
    return (*data_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]];
}

const float& NastyTensors::operator()(size_t i, size_t j, size_t k) const {
    return (*data_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]];
}

NastyTensors NastyTensors::matmul(const NastyTensors &other) const
{   
    assert(this->ndim() == 2 && other.ndim() == 2 && "Not a 2d Matrix!");
    assert(this->shape()[1] == other.shape()[0] && "Dimensions dont match");

    size_t M = this->shape()[0], K = this->shape()[1], N = other.shape()[1];

    NastyTensors output({M, N}, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float val = (*this)(i, k);
            for (size_t j = 0; j < N; ++j) {
                output(i, j) += val * other(k, j);
            }
        }
    }

    return output;
}

void NastyTensors::print() const
{
    std::cout << "(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    std::cout << "], offset=" << offset_ << ")\n";

    if (shape_.empty() || size() == 0) {
        std::cout << "[]\n";
        return;
    }

    const float* ptr = data();
    if (!ptr) {
        std::cout << "[null]\n";
        return;
    }

    if (shape_.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            std::cout << std::fixed << std::setprecision(4) << ptr[i] << (i < shape_[0] - 1 ? ", " : "");
        }
        std::cout << "]\n";
    } else if (shape_.size() == 2) {
        std::cout << "[\n";
        for (size_t i = 0; i < shape_[0]; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < shape_[1]; ++j) {
                std::cout << std::fixed << std::setprecision(4) << (*this)(i, j) << (j < shape_[1] - 1 ? ", " : "");
            }
            std::cout << "]" << (i < shape_[0] - 1 ? ",\n" : "\n");
        }
        std::cout << "]\n";
    } else {
        std::cout << "[Flattened view]:\n  ";
        for (size_t i = 0; i < size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << ptr[i] << (i < size() - 1 ? ", " : "");
        }
        std::cout << "\n";
    }
}