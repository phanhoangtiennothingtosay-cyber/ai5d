#pragma once

#include <cstddef>

#include "ai5d/tensor.hpp"

namespace ai5d::math {

Tensor attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
);

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale
);

Tensor softmax(
    const Tensor& input,
    std::size_t axis = 0
);

Tensor attention_weights(
    const Tensor& query,
    const Tensor& key,
    float scale
);

} // namespace ai5d::math
