#pragma once

#include <cstddef>

#include "ai5d/tensor.hpp"

namespace ai5d::math {

enum class ActivationType {
    NONE,
    RELU,
    GELU,
    SIGMOID,
    TANH
};

float activate(float value, ActivationType type);

Tensor apply_activation(
    const Tensor& input,
    ActivationType type
);

const char* activation_name(ActivationType type);

} // namespace ai5d::math
