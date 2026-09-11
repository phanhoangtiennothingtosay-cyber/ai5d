#include "ai5d/math/linear.hpp"

#include <stdexcept>

namespace ai5d::math {

Tensor linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias
)
{
    Tensor output = matmul(input, weights);

    if (!bias.empty()) {
        output = add_bias(output, bias);
    }

    return output;
}

Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs
)
{
    if (lhs.empty() || rhs.empty()) {
        return Tensor{};
    }

    if (lhs.ndim() != 2 || rhs.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: matmul requires two 2D tensors."
        );
    }

    const std::size_t lhs_rows = lhs.shape()[0];
    const std::size_t lhs_cols = lhs.shape()[1];

    const std::size_t rhs_rows = rhs.shape()[0];
    const std::size_t rhs_cols = rhs.shape()[1];

    if (lhs_cols != rhs_rows) {
        throw std::invalid_argument(
            "AI5D: incompatible tensor shapes for matmul."
        );
    }

    Tensor output({lhs_rows, rhs_cols});

    for (std::size_t i = 0; i < lhs_rows; ++i) {
        for (std::size_t j = 0; j < rhs_cols; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < lhs_cols; ++k) {
                sum += lhs[i * lhs_cols + k]
                     * rhs[k * rhs_cols + j];
            }

            output[i * rhs_cols + j] = sum;
        }
    }

    return output;
}

Tensor add_bias(
    const Tensor& input,
    const Tensor& bias
)
{
    if (input.empty()) {
        return Tensor{};
    }

    if (bias.empty()) {
        return input;
    }

    if (bias.size() != input.size()) {
        throw std::invalid_argument(
            "AI5D: bias size must match input size."
        );
    }

    Tensor output = input;

    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] += bias[i];
    }

    return output;
}

float dot(
    const float* lhs,
    const float* rhs,
    std::size_t size
)
{
    if (lhs == nullptr || rhs == nullptr) {
        throw std::invalid_argument(
            "AI5D: dot received a null pointer."
        );
    }

    float result = 0.0f;

    for (std::size_t i = 0; i < size; ++i) {
        result += lhs[i] * rhs[i];
    }

    return result;
}

} // namespace ai5d::math