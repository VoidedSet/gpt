#include "NastyTensors.hpp"
#include "CudaKernels.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <cublas_v2.h>

#include <cuda_runtime.h>

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

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
    : data_(nullptr), grad_(nullptr), offset_(0), shape_({}), strides_({}) {}

NastyTensors::NastyTensors(const std::vector<size_t>& shape) 
    : data_(nullptr), grad_(nullptr), offset_(0), shape_(shape), strides_(calculate_strides(shape)) {
    size_t total_size = size();
    data_ = std::make_shared<std::vector<float>>(total_size, 0.0f);
}

NastyTensors::NastyTensors(const std::vector<size_t>& shape, float fill_value) 
    : data_(nullptr), grad_(nullptr), offset_(0), shape_(shape), strides_(calculate_strides(shape)) {
    size_t total_size = size();
    data_ = std::make_shared<std::vector<float>>(total_size, fill_value);
}

NastyTensors::NastyTensors(const std::vector<size_t>& shape, 
                           std::shared_ptr<std::vector<float>> data, 
                           std::shared_ptr<std::vector<float>> grad,
                           std::shared_ptr<float> d_data, 
                           std::shared_ptr<float> d_grad,  
                           size_t offset) 
    : data_(data), grad_(grad), d_data_(d_data), d_grad_(d_grad), offset_(offset), shape_(shape), strides_(calculate_strides(shape)) {}

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
    if (this->grad_) {
        cloned.init_grad();
        float* dest_g = cloned.grad();
        const float* src_g = this->grad();
        if (src_g && dest_g) {
            for (size_t i = 0; i < num_elements; ++i) {
                dest_g[i] = src_g[i];
            }
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
    
    return NastyTensors(new_shape, data_, grad_, d_data_, d_grad_, new_offset);
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

    if (d_data_ != nullptr) {
        output.to_gpu(DATA_);
        cublasHandle_t handle = get_cublas_handle();
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, 
                    &alpha, 
                    other.device_data(), N, 
                    this->device_data(), K, 
                    &beta, 
                    output.device_data(), N);
        return output;
    }

    const float* a = this->data();
    const float* b = other.data();
    float* c = output.data();

    size_t lda = this->strides_[0];
    size_t ldb = other.strides_[0];
    size_t ldc = output.strides_[0];

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float val = a[i * lda + k];
            __m256 val_vec = _mm256_set1_ps(val);
            size_t j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 vb = _mm256_loadu_ps(b + k * ldb + j);
                __m256 vc = _mm256_loadu_ps(c + i * ldc + j);
                vc = _mm256_fmadd_ps(val_vec, vb, vc);
                _mm256_storeu_ps(c + i * ldc + j, vc);
            }
            for (; j < N; ++j) {
                c[i * ldc + j] += val * b[k * ldb + j];
            }
        }
    }

    return output;
}

void NastyTensors::gelu() {
    if (d_data_ != nullptr) {
        launch_gelu_forward(device_data(), size());
        return;
    }

    float* ptr = data();
    size_t n = size();
    if (!ptr) return;
    for (size_t i = 0; i < n; ++i) {
        float x = ptr[i];
        ptr[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

NastyTensors NastyTensors::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = 1;
    for (size_t dim : new_shape) new_size *= dim;
    if (new_size != this->size()) {
        throw std::runtime_error("Cannot reshape: size mismatch.");
    }
    return NastyTensors(new_shape, data_, grad_, d_data_, d_grad_, offset_);
}

NastyTensors& NastyTensors::operator+=(const NastyTensors& other) {
    assert(this->size() == other.size());
    float* dest = this->data();
    const float* src = other.data();
    size_t n = this->size();
    if (dest && src) {
        for (size_t i = 0; i < n; ++i) {
            dest[i] += src[i];
        }
    }
    return *this;
}

void NastyTensors::init_grad() {
    if (!grad_ && data_) {
        grad_ = std::make_shared<std::vector<float>>(data_->size(), 0.0f);
    }
}

void NastyTensors::zero_grad() {
    if (grad_) {
        float* g = grad();
        size_t n = size();
        if (g) {
            std::fill(g, g + n, 0.0f);
        }
    }
    if(d_grad_)
        cudaMemset(d_grad_.get(), 0, size() * sizeof(float));
}

NastyTensors NastyTensors::matmul_transposed_b(const NastyTensors& other) const {
    assert(this->ndim() == 2 && other.ndim() == 2);
    assert(this->shape()[1] == other.shape()[1]);

    size_t M = this->shape()[0];
    size_t K = this->shape()[1];
    size_t N = other.shape()[0];

    NastyTensors output({M, N}, 0.0f);

    if (d_data_ != nullptr) {
        output.to_gpu(DATA_);
        cublasHandle_t handle = get_cublas_handle();
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K, 
                    &alpha, 
                    other.device_data(), K, 
                    this->device_data(), K, 
                    &beta, 
                    output.device_data(), N);
        return output;
    }

    const float* a = this->data();
    const float* b = other.data();
    float* c = output.data();

    size_t lda = this->strides_[0];
    size_t ldb = other.strides_[0];
    size_t ldc = output.strides_[0];

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const float* row_a = a + i * lda;
            const float* row_b = b + j * ldb;

            __m256 sum = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 va = _mm256_loadu_ps(row_a + k);
                __m256 vb = _mm256_loadu_ps(row_b + k);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            
            alignas(32) float temp[8];
            _mm256_storeu_ps(temp, sum);
            float dot = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            
            for (; k < K; ++k) {
                dot += row_a[k] * row_b[k];
            }
            
            c[i * ldc + j] = dot;
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

void NastyTensors::to_gpu(int type)
{
    if(type == DATA_){
        if(!d_data_){
            float* raw_gpu_ptr = nullptr;
            cudaMalloc(&raw_gpu_ptr, (size() * sizeof(float)));
            d_data_ = std::shared_ptr<float>(raw_gpu_ptr, [](float* p){
                if(p) cudaFree(p);
            });
        }        
        cudaMemcpy(d_data_.get(), data(), (size() * sizeof(float)), cudaMemcpyHostToDevice);
    } else if(type == GRAD_){
        if(!d_grad_){
            float* raw_gpu_ptr = nullptr;
            cudaMalloc(&raw_gpu_ptr, (size() * sizeof(float)));
            d_grad_ = std::shared_ptr<float>(raw_gpu_ptr, [](float* p){
                if (p) cudaFree(p);
            });
        }
        cudaMemcpy(d_grad_.get(), grad(), (size() * sizeof(float)), cudaMemcpyHostToDevice);
    }
}

void NastyTensors::to_cpu(int type){
    if(type == DATA_ && d_data_)
        cudaMemcpy(data(), d_data_.get(), (size() * sizeof(float)), cudaMemcpyDeviceToHost);
    else if(type == GRAD_ && d_grad_)
        cudaMemcpy(grad(), d_grad_.get(), (size() * sizeof(float)), cudaMemcpyDeviceToHost);
}

void NastyTensors::gemm(bool transA, bool transB,
                        size_t M, size_t N, size_t K,
                        float alpha,
                        const float* A, size_t lda,
                        const float* B, size_t ldb,
                        float beta,
                        float* C, size_t ldc) {
    cublasHandle_t handle = get_cublas_handle();
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(handle, opB, opA, 
                static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), 
                &alpha, 
                B, static_cast<int>(ldb), 
                A, static_cast<int>(lda), 
                &beta, 
                C, static_cast<int>(ldc));
}
