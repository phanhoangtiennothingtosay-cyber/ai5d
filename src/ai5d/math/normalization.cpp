#include "ai5d/math/normalization.hpp"

#include <cmath>
#include <stdexcept>

namespace ai5d::math {

Tensor layer_norm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon
)
{
    if (input.empty()) {
        return Tensor{};
    }

    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "AI5D: epsilon must be greater than zero."
        );
    }

    if (gamma.size() != input.size() ||
        beta.size() != input.size()) {
        throw std::invalid_argument(
            "AI5D: gamma and beta must match input size."
        );
    }

    const float input_mean = mean(input);
    const float input_variance = variance(input, input_mean);

    const float inverse_std =
        1.0f / std::sqrt(input_variance + epsilon);

    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        const float normalized =
            (input[i] - input_mean) * inverse_std;

        output[i] = gamma[i] * normalized + beta[i];
    }

    return output;
}

Tensor normalize(
    const Tensor& input,
    float epsilon
)
{
    if (input.empty()) {
        return Tensor{};
    }

    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "AI5D: epsilon must be greater than zero."
        );
    }

    const float input_mean = mean(input);
    const float input_variance = variance(input, input_mean);

    const float inverse_std =
        1.0f / std::sqrt(input_variance + epsilon);

    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] =
            (input[i] - input_mean) * inverse_std;
    }

    return output;
}

float mean(const Tensor& input)
{
    if (input.empty()) {
        return 0.0f;
    }

    float total = 0.0f;

    for (std::size_t i = 0; i < input.size(); ++i) {
        total += input[i];
    }

    return total / static_cast<float>(input.size());
}

float variance(
    const Tensor& input,
    float mean_value
)
{
    if (input.empty()) {
        return 0.0f;
    }

    float total = 0.0f;

    for (std::size_t i = 0; i < input.size(); ++i) {
        const float difference = input[i] - mean_value;
        total += difference * difference;
    }

    return total / static_cast<float>(input.size());
}

} // namespace ai5d::math