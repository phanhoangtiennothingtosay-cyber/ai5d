#include "ai5d/layers/layer2d.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ai5d::layers {

Tensor Layer2D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    if (rows_ > 0 && cols_ > 0) {
        if (input.size() != rows_ * cols_) {
            throw std::invalid_argument(
                "AI5D: Layer2D input size does not match configured shape."
            );
        }
    }

    Tensor output(input.shape());

    /*
     * Layer2D hoạt động theo từng worker.
     *
     * Mỗi worker nhìn vào một phần dữ liệu của Tensor.
     * Ở phiên bản nền này, worker tạo ra một trọng số
     * dựa trên giá trị trung bình của phần dữ liệu đó.
     *
     * Sau này phần này có thể được thay bằng matrix
     * transformation / attention mà không cần đổi public API.
     */
    const std::size_t workers =
        worker_count_ == 0 ? 1 : worker_count_;

    const std::size_t chunk =
        (input.size() + workers - 1) / workers;

    for (std::size_t worker = 0; worker < workers; ++worker) {
        const std::size_t begin = worker * chunk;

        if (begin >= input.size()) {
            break;
        }

        const std::size_t end =
            std::min(begin + chunk, input.size());

        float sum = 0.0f;

        for (std::size_t i = begin; i < end; ++i) {
            sum += input[i];
        }

        const std::size_t count = end - begin;

        const float mean =
            count > 0
                ? sum / static_cast<float>(count)
                : 0.0f;

        for (std::size_t i = begin; i < end; ++i) {
            output[i] = input[i] * mean;
        }
    }

    return output;
}

Tensor Layer2D::combine(
    const std::vector<Tensor>& candidates
) const
{
    if (candidates.empty()) {
        return Tensor{};
    }

    std::size_t best_index = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const float current_score = score(candidates[i]);

        if (current_score > best_score) {
            best_score = current_score;
            best_index = i;
        }
    }

    return candidates[best_index];
}

float Layer2D::score(const Tensor& input) const
{
    if (input.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;

    for (std::size_t i = 0; i < input.size(); ++i) {
        sum += input[i];
    }

    return sum / static_cast<float>(input.size());
}

std::size_t Layer2D::worker_count() const
{
    return worker_count_;
}

void Layer2D::set_worker_count(std::size_t count)
{
    worker_count_ = count;
}

std::size_t Layer2D::rows() const
{
    return rows_;
}

std::size_t Layer2D::cols() const
{
    return cols_;
}

void Layer2D::set_shape(
    std::size_t rows,
    std::size_t cols
)
{
    rows_ = rows;
    cols_ = cols;
}

} // namespace ai5d::layers
