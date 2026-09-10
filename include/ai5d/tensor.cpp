#include "ai5d/tensor.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace ai5d {

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape)
{
    data_.resize(calculate_size(shape_));
}

Tensor::Tensor(std::initializer_list<float> data)
    : data_(data)
{
    shape_ = {data_.size()};
}

Tensor::~Tensor() = default;

float* Tensor::data()
{
    return data_.data();
}

const float* Tensor::data() const
{
    return data_.data();
}

std::size_t Tensor::size() const
{
    return data_.size();
}

std::size_t Tensor::ndim() const
{
    return shape_.size();
}

const std::vector<std::size_t>& Tensor::shape() const
{
    return shape_;
}

float& Tensor::operator[](std::size_t index)
{
    if (index >= data_.size()) {
        throw std::out_of_range("AI5D: tensor index out of range.");
    }

    return data_[index];
}

const float& Tensor::operator[](std::size_t index) const
{
    if (index >= data_.size()) {
        throw std::out_of_range("AI5D: tensor index out of range.");
    }

    return data_[index];
}

void Tensor::reshape(const std::vector<std::size_t>& new_shape)
{
    const std::size_t new_size = calculate_size(new_shape);

    if (new_size != data_.size()) {
        throw std::invalid_argument(
            "AI5D: reshape cannot change the number of elements."
        );
    }

    shape_ = new_shape;
}

bool Tensor::empty() const
{
    return data_.empty();
}

void Tensor::clear()
{
    data_.clear();
    shape_.clear();
}

std::vector<float>& Tensor::vector()
{
    return data_;
}

const std::vector<float>& Tensor::vector() const
{
    return data_;
}

std::size_t Tensor::calculate_size(
    const std::vector<std::size_t>& shape
)
{
    if (shape.empty()) {
        return 0;
    }

    return std::accumulate(
        shape.begin(),
        shape.end(),
        std::size_t{1},
        [](std::size_t total, std::size_t dimension) {
            return total * dimension;
        }
    );
}

} // namespace ai5d
