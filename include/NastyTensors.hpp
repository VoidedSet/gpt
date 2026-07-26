#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <numeric>

class NastyTensors {
private:
    std::shared_ptr<std::vector<float>> data_;
    size_t offset_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    static std::vector<size_t> calculate_strides(const std::vector<size_t>& shape);

public:
    NastyTensors();
    explicit NastyTensors(const std::vector<size_t>& shape);
    NastyTensors(const std::vector<size_t>& shape, float fill_value);
    NastyTensors(const std::vector<size_t>& shape, 
                 std::shared_ptr<std::vector<float>> data, 
                 size_t offset);
    ~NastyTensors() = default;

    NastyTensors clone() const;
    NastyTensors slice(size_t index) const;
    NastyTensors matmul(const NastyTensors& other) const;
    void gelu();

    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t offset() const { return offset_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const;

    float* data() { return data_ ? (data_->data() + offset_) : nullptr; }
    const float* data() const { return data_ ? (data_->data() + offset_) : nullptr; }

    float& operator()(size_t i);
    const float& operator()(size_t i) const;

    float& operator()(size_t i, size_t j);
    const float& operator()(size_t i, size_t j) const;

    float& operator()(size_t i, size_t j, size_t k);
    const float& operator()(size_t i, size_t j, size_t k) const;

    void print() const;
};