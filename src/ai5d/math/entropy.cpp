#include "ai5d/math/entropy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai5d::math {

float entropy(const Tensor& probabilities)
{
    if (probabilities.empty()) {
        return 0.0f;
    }

    constexpr float epsilon = 1e-12f;
    float result = 0.0f;

    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const float probability =
            std::max(probabilities[i], epsilon);

        result -= probability * std::log(probability);
    }

    return result;
}

float normalized_entropy(const Tensor& probabilities)
{
    if (probabilities.empty() || probabilities.size() <= 1) {
        return 0.0f;
    }

    const float maximum_entropy =
        std::log(static_cast<float>(probabilities.size()));

    if (maximum_entropy <= 0.0f) {
        return 0.0f;
    }

    return entropy(probabilities) / maximum_entropy;
}

float confidence(const Tensor& probabilities)
{
    if (probabilities.empty()) {
        return 0.0f;
    }

    float best = probabilities[0];

    for (std::size_t i = 1; i < probabilities.size(); ++i) {
        best = std::max(best, probabilities[i]);
    }

    return best;
}

float margin(const Tensor& probabilities)
{
    if (probabilities.empty()) {
        return 0.0f;
    }

    if (probabilities.size() == 1) {
        return probabilities[0];
    }

    float best = -std::numeric_limits<float>::infinity();
    float second_best = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const float value = probabilities[i];

        if (value > best) {
            second_best = best;
            best = value;
        } else if (value > second_best) {
            second_best = value;
        }
    }

    return best - second_best;
}

bool is_uncertain(
    const Tensor& probabilities,
    float entropy_threshold,
    float confidence_threshold
)
{
    if (probabilities.empty()) {
        return true;
    }

    const float current_entropy =
        normalized_entropy(probabilities);

    const float current_confidence =
        confidence(probabilities);

    return current_entropy >= entropy_threshold ||
           current_confidence <= confidence_threshold;
}

} // namespace ai5d::math