#include "ai5d/layers/layer1d.hpp"

#include <cmath>

namespace ai5d::layers {

Tensor Layer1D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = process(input[i]);
    }

    return output;
}

float Layer1D::process(float value) const
{
    return std::tanh(value);
}

std::size_t Layer1D::input_size() const
{
    return input_size_;
}

void Layer1D::set_input_size(std::size_t size)
{
    input_size_ = size;
}

} // namespace ai5d::layers
