#include "ai5d/math/attention.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ai5d::math {

Tensor attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
) {
    if (query.empty() || key.empty() || value.empty()) {
        return Tensor{};
    }

    if (query.ndim() == 0) {
        throw std::invalid_argument(
            "AI5D: query must have at least one dimension."
        );
    }

    const float scale =
        1.0f / std::sqrt(
            static_cast<float>(query.shape().back())
        );

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
) {
    if (query.empty() || key.empty() || value.empty()) {
        return Tensor{};
    }

    if (query.ndim() != 2 ||
        key.ndim() != 2 ||
        value.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: scaled_dot_product_attention requires "
            "2D query, key, and value tensors."
        );
    }

    if (scale <= 0.0f) {
        throw std::invalid_argument(
            "AI5D: attention scale must be greater than zero."
        );
    }

    const std::size_t query_rows = query.shape()[0];
    const std::size_t query_dim = query.shape()[1];

    const std::size_t key_rows = key.shape()[0];
    const std::size_t key_dim = key.shape()[1];

    const std::size_t value_rows = value.shape()[0];
    const std::size_t value_dim = value.shape()[1];

    if (query_dim != key_dim) {
        throw std::invalid_argument(
            "AI5D: query and key dimensions must match."
        );
    }

    if (key_rows != value_rows) {
        throw std::invalid_argument(
            "AI5D: key and value row counts must match."
        );
    }

    // Q × K^T
    Tensor scores(
        std::vector<std::size_t>{query_rows, key_rows}
    );

    for (std::size_t i = 0; i < query_rows; ++i) {
        for (std::size_t j = 0; j < key_rows; ++j) {

            float sum = 0.0f;

            for (std::size_t k = 0; k < query_dim; ++k) {
                sum +=
                    query[i * query_dim + k] *
                    key[j * key_dim + k];
            }

            scores[i * key_rows + j] = sum * scale;
        }
    }

    // Softmax over keys for every query.
    Tensor weights = softmax(scores, 1);

    // Attention weights × V
    Tensor output(
        std::vector<std::size_t>{query_rows, value_dim}
    );

    for (std::size_t i = 0; i < query_rows; ++i) {
        for (std::size_t j = 0; j < value_dim; ++j) {

            float sum = 0.0f;

            for (std::size_t k = 0; k < value_rows; ++k) {
                sum +=
                    weights[i * value_rows + k] *
                    value[k * value_dim + j];
            }

            output[i * value_dim + j] = sum;
        }
    }

    return output;
}

Tensor softmax(
    const Tensor& input,
    std::size_t axis
) {
    if (input.empty()) {
        return Tensor{};
    }

    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: softmax currently requires a 2D tensor."
        );
    }

    if (axis > 1) {
        throw std::invalid_argument(
            "AI5D: softmax axis must be 0 or 1."
        );
    }

    Tensor output(input.shape());

    const std::size_t rows = input.shape()[0];
    const std::size_t cols = input.shape()[1];

    if (axis == 0) {

        for (std::size_t j = 0; j < cols; ++j) {

            float maximum = input[j];

            for (std::size_t i = 1; i < rows; ++i) {
                maximum = std::max(
                    maximum,
                    input[i * cols + j]
                );
            }

            float sum = 0.0f;

            for (std::size_t i = 0; i < rows; ++i) {
                const float value =
                    std::exp(
                        input[i * cols + j] - maximum
                    );

                output[i * cols + j] = value;
                sum += value;
            }

            if (sum > 0.0f) {
                for (std::size_t i = 0; i < rows; ++i) {
                    output[i * cols + j] /= sum;
                }
            }
        }

    } else {

        for (std::size_t i = 0; i < rows; ++i) {

            float maximum = input[i * cols];

            for (std::size_t j = 1; j < cols; ++j) {
                maximum = std::max(
                    maximum,
                    input[i * cols + j]
                );
            }

            float sum = 0.0f;

            for (std::size_t j = 0; j < cols; ++j) {
                const float value =
                    std::exp(
                        input[i * cols + j] - maximum
                    );

                output[i * cols + j] = value;
                sum += value;
            }

            if (sum > 0.0f) {
                for (std::size_t j = 0; j < cols; ++j) {
                    output[i * cols + j] /= sum;
                }
            }
        }
    }

    return output;
}

Tensor attention_weights(
    const Tensor& query,
    const Tensor& key,
    float scale
) {
    if (query.empty() || key.empty()) {
        return Tensor{};
    }

    if (query.ndim() != 2 ||
        key.ndim() != 2) {
        throw std::invalid_argument(
            "AI5D: attention_weights requires "
            "2D query and key tensors."
        );
    }

    if (scale <= 0.0f) {
        throw std::invalid_argument(
            "AI5D: attention scale must be greater than zero."
        );
    }

    const std::size_t query_rows = query.shape()[0];
    const std::size_t query_dim = query.shape()[1];

    const std::size_t key_rows = key.shape()[0];
    const std::size_t key_dim = key.shape()[1];

    if (query_dim != key_dim) {
        throw std::invalid_argument(
            "AI5D: query and key dimensions must match."
        );
    }

    // Q × K^T
    Tensor scores(
        std::vector<std::size_t>{query_rows, key_rows}
    );

    for (std::size_t i = 0; i < query_rows; ++i) {
        for (std::size_t j = 0; j < key_rows; ++j) {

            float sum = 0.0f;

            for (std::size_t k = 0; k < query_dim; ++k) {
                sum +=
                    query[i * query_dim + k] *
                    key[j * key_dim + k];
            }

            scores[i * key_rows + j] = sum * scale;
        }
    }

    return softmax(scores, 1);
}

} // namespace ai5d::math
