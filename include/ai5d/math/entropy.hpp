#pragma once

#include <cstddef>

#include "ai5d/tensor.hpp"

namespace ai5d::math {

float entropy(const Tensor& probabilities);

float normalized_entropy(const Tensor& probabilities);

float confidence(const Tensor& probabilities);

float margin(const Tensor& probabilities);

bool is_uncertain(
    const Tensor& probabilities,
    float entropy_threshold,
    float confidence_threshold
);

} // namespace ai5d::math
