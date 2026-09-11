#include "ai5d/math/attention.hpp"

#include <cmath>
#include <stdexcept>

namespace ai5d::math {

Tensor attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
)
{
    if (query.empty() || key.empty() || value.empty()) {
        return Tensor{};
    }

    const float scale =
        1.0f / std::sqrt(static_cast<float>(query.shape().back()));

    return scaled_dot_product_attention(
        query,
        key,
        value,
        scale
    );
}

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale
)
{
    if (query.empty() || key.empty() || value.empty()) {
        return Tensor{};
    }

    if (query.ndim() != 2 ||
        key.ndim() != 2 ||
        value.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: attention requires 2D tensors."
        );
    }

    if (query.shape()[1] != key.shape()[1]) {
        throw std::invalid_argument(
            "AI5D: query and key dimensions must match."
        );
    }

    if (key.shape()[0] != value.shape()[0]) {
        throw std::invalid_argument(
            "AI5D: key and value sequence dimensions must match."
        );
    }

    if (scale <= 0.0f) {
        throw std::invalid_argument(
            "AI5D: attention scale must be greater than zero."
        );
    }

    const std::size_t query_rows = query.shape()[0];
    const std::size_t key_rows = key.shape()[0];
    const std::size_t dimension = query.shape()[1];
    const std::size_t value_dim = value.shape()[1];

    Tensor scores({query_rows, key_rows});

    for (std::size_t i = 0; i < query_rows; ++i) {
        for (std::size_t j = 0; j < key_rows; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < dimension; ++k) {
                sum += query[i * dimension + k]
                     * key[j * dimension + k];
            }

            scores[i * key_rows + j] = sum * scale;
        }
    }

    Tensor weights = softmax(scores, 1);

    Tensor output({query_rows, value_dim});

    for (std::size_t i = 0; i < query_rows; ++i) {
        for (std::size_t j = 0; j < value_dim; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < key_rows; ++k) {
                sum += weights[i * key_rows + k]
                     * value[k * value_dim + j];
            }

            output[i * value_dim + j] = sum;
        }
    }

    return output;
}

Tensor softmax(
    const Tensor& input,
    std::size_t axis
)
{
    if (input.empty()) {
        return Tensor{};
    }

    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: softmax currently requires a 2D tensor."
        );
    }

    if (axis >= input.ndim()) {
        throw std::invalid_argument(
            "AI5D: softmax axis is out of range."
        );
    }

    const std::size_t rows = input.shape()[0];
    const std::size_t cols = input.shape()[1];

    Tensor output(input.shape());

    if (axis == 0) {
        for (std::size_t j = 0; j < cols; ++j) {
            float max_value = input[j];

            for (std::size_t i = 1; i < rows; ++i) {
                max_value =
                    std::max(max_value, input[i * cols + j]);
            }

            float sum = 0.0f;

            for (std::size_t i = 0; i < rows; ++i) {
                const float value =
                    std::exp(input[i * cols + j] - max_value);

                output[i * cols + j] = value;
                sum += value;
            }

            for (std::size_t i = 0; i < rows; ++i) {
                output[i * cols + j] /= sum;
            }
        }

        return output;
    }

    for (std::size_t i = 0; i < rows; ++i) {
        const std::size_t offset = i * cols;
        float max_value = input[offset];

        for (std::size_t j = 1; j < cols; ++j) {
            max_value =
                std::max(max_value, input[offset + j]);
        }

        float sum = 0.0f;

        for (std::size_t j = 0; j < cols; ++j) {
            const float value =
                std::exp(input[offset + j] - max_value);

            output[offset + j] = value;
            sum += value;
        }

        for (std::size_t j = 0; j < cols; ++j) {
            output[offset + j] /= sum;
        }
    }

    return output;
}

Tensor attention_weights(
    const Tensor& query,
    const Tensor& key,
    float scale
)
{
    if (query.empty() || key.empty()) {
        return Tensor{};
    }

    if (query.ndim() != 2 || key.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: attention weights require 2D tensors."
        );
    }

    if (query.shape()[1] != key.shape()[1]) {
        throw std::invalid_argument(
            "AI5D: query and key dimensions must match."
        );
    }

    if (scale <= 0.0f) {
        throw std::invalid_argument(
            "AI5D: attention scale must be greater than zero."
        );
    }

    const std::size_t query_rows = query.shape()[0];
    const std::size_t key_rows = key.shape()[0];
    const std::size_t dimension = query.shape()[1];

    Tensor scores({query_rows, key_rows});

    for (std::size_t i = 0; i < query_rows; ++i) {
        for (std::size_t j = 0; j < key_rows; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < dimension; ++k) {
                sum += query[i * dimension + k]
                     * key[j * dimension + k];
            }

            scores[i * key_rows + j] = sum * scale;
        }
    }

    return softmax(scores, 1);
}

} // namespace ai5d::math