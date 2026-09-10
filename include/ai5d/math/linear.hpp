#pragma once

#include <cstddef>

#include "ai5d/tensor.hpp"

namespace ai5d::math {

Tensor linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias
);

Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs
);

Tensor add_bias(
    const Tensor& input,
    const Tensor& bias
);

float dot(
    const float* lhs,
    const float* rhs,
    std::size_t size
);

} // namespace ai5d::math
