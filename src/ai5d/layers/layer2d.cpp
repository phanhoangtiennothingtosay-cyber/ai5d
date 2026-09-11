#include "ai5d/layers/layer2d.hpp"

#include <algorithm>
#include <stdexcept>

namespace ai5d::layers {

Tensor Layer2D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    Tensor output = input;

    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = score(output);
        break;
    }

    return output;
}

Tensor Layer2D::combine(const std::vector<Tensor>& inputs) const
{
    if (inputs.empty()) {
        return Tensor{};
    }

    const Tensor* best = &inputs.front();
    float best_score = score(*best);

    for (std::size_t i = 1; i < inputs.size(); ++i) {
        const float current_score = score(inputs[i]);

        if (current_score > best_score) {
            best = &inputs[i];
            best_score = current_score;
        }
    }

    return *best;
}

float Layer2D::score(const Tensor& candidate) const
{
    if (candidate.empty()) {
        return 0.0f;
    }

    float total = 0.0f;

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        total += candidate[i];
    }

    return total / static_cast<float>(candidate.size());
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

void Layer2D::set_shape(std::size_t rows, std::size_t cols)
{
    rows_ = rows;
    cols_ = cols;
}

} // namespace ai5d::layers