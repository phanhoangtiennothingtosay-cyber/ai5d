#pragma once

#include "ai5d/tensor.hpp"

namespace ai5d::math {

Tensor layer_norm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon = 1e-6f
);

Tensor normalize(
    const Tensor& input,
    float epsilon = 1e-6f
);

float mean(const Tensor& input);

float variance(
    const Tensor& input,
    float mean_value
);

} // namespace ai5d::math
