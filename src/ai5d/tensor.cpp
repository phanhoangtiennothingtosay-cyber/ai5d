#include "ai5d/tensor.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai5d {

Tensor::Tensor(
    const std::vector<std::size_t>& shape
)
    : data_{},
      shape_{}
{
    const std::size_t total_size =
        calculate_size(shape);

    shape_ = shape;
    data_.resize(total_size);
}

Tensor::Tensor(
    std::initializer_list<float> data
)
    : data_(data),
      shape_{data.size()}
{
}

Tensor Tensor::from_shape(
    const std::vector<std::size_t>& shape
)
{
    return Tensor(shape);
}

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
        throw std::out_of_range(
            "AI5D: Tensor index out of range."
        );
    }

    return data_[index];
}

const float& Tensor::operator[](
    std::size_t index
) const
{
    if (index >= data_.size()) {
        throw std::out_of_range(
            "AI5D: Tensor index out of range."
        );
    }

    return data_[index];
}

void Tensor::reshape(
    const std::vector<std::size_t>& new_shape
)
{
    const std::size_t new_size =
        calculate_size(new_shape);

    if (new_size != data_.size()) {
        throw std::invalid_argument(
            "AI5D: reshape cannot change the number "
            "of tensor elements."
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

bool Tensor::has_nan() const
{
    for (const float value : data_) {
        if (std::isnan(value)) {
            return true;
        }
    }

    return false;
}

bool Tensor::has_inf() const
{
    for (const float value : data_) {
        if (std::isinf(value)) {
            return true;
        }
    }

    return false;
}

std::size_t Tensor::calculate_size(
    const std::vector<std::size_t>& shape
)
{
    if (shape.empty()) {
        return 0;
    }

    std::size_t total = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "AI5D: Tensor dimensions cannot be zero."
            );
        }

        if (
            total >
            std::numeric_limits<std::size_t>::max()
                / dimension
        ) {
            throw std::overflow_error(
                "AI5D: Tensor size overflow."
            );
        }

        total *= dimension;
    }

    return total;
}

} // namespace ai5d
