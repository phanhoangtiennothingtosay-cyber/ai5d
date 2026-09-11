#include "ai5d/math/activation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ai5d::math {

float activate(float value, ActivationType type)
{
    switch (type) {
    case ActivationType::NONE:
        return value;

    case ActivationType::RELU:
        return std::max(0.0f, value);

    case ActivationType::GELU:
        return 0.5f * value *
               (1.0f + std::erf(value / std::sqrt(2.0f)));

    case ActivationType::SIGMOID:
        return 1.0f / (1.0f + std::exp(-value));

    case ActivationType::TANH:
        return std::tanh(value);
    }

    throw std::invalid_argument("AI5D: unknown activation type.");
}

Tensor apply_activation(
    const Tensor& input,
    ActivationType type
)
{
    if (input.empty()) {
        return Tensor{};
    }

    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = activate(input[i], type);
    }

    return output;
}

const char* activation_name(ActivationType type)
{
    switch (type) {
    case ActivationType::NONE:
        return "NONE";

    case ActivationType::RELU:
        return "RELU";

    case ActivationType::GELU:
        return "GELU";

    case ActivationType::SIGMOID:
        return "SIGMOID";

    case ActivationType::TANH:
        return "TANH";
    }

    return "UNKNOWN";
}

} // namespace ai5d::math 