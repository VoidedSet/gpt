#pragma once

#define DATA_ 0
#define GRAD_ 1

#include <vector>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <numeric>

class NastyTensors {
private:
    std::shared_ptr<std::vector<float>> data_;
    std::shared_ptr<std::vector<float>> grad_;

    std::shared_ptr<float> d_data_, d_grad_;
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
                 std::shared_ptr<std::vector<float>> grad, 
                 std::shared_ptr<float> d_data, 
                 std::shared_ptr<float> d_grad, 
                 size_t offset);
    ~NastyTensors() = default;

    NastyTensors clone() const;
    NastyTensors slice(size_t index) const;
    NastyTensors matmul(const NastyTensors& other) const;
    void gelu();
    NastyTensors reshape(const std::vector<size_t>& new_shape) const;
    NastyTensors& operator+=(const NastyTensors& other);
    NastyTensors matmul_transposed_b(const NastyTensors& other) const;

    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t offset() const { return offset_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const;

    // == CUDA WRAPPERS == //
    void to_gpu(int type);
    void to_cpu(int type);

    float* device_data() {return d_data_ ? (d_data_.get() + offset_) : nullptr; }
    const float* device_data() const { return d_data_ ? (d_data_.get() + offset_) : nullptr; }

    float* device_grad() { return d_grad_ ? (d_grad_.get() + offset_) : nullptr; }
    const float* device_grad() const { return d_grad_ ? (d_grad_.get() + offset_) : nullptr; }
    
    // == END OF CUDA WRAPPERS == //
    
    float* data() { return data_ ? (data_->data() + offset_) : nullptr; }
    const float* data() const { return data_ ? (data_->data() + offset_) : nullptr; }

    float* grad() { return grad_ ? (grad_->data() + offset_) : nullptr; }
    const float* grad() const { return grad_ ? (grad_->data() + offset_) : nullptr; }

    void init_grad();
    void zero_grad();

    float& operator()(size_t i);
    const float& operator()(size_t i) const;

    float& operator()(size_t i, size_t j);
    const float& operator()(size_t i, size_t j) const;

    float& operator()(size_t i, size_t j, size_t k);
    const float& operator()(size_t i, size_t j, size_t k) const;

    void print() const;
};