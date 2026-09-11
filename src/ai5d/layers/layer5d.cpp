#include "ai5d/layers/layer5d.hpp"

#include <limits>
#include <stdexcept>

namespace ai5d::layers {

Tensor Layer5D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    /*
     * forward() xử lý một candidate đơn.
     *
     * Việc lựa chọn giữa nhiều candidate thuộc về select().
     */
    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i];
    }

    return output;
}

Tensor Layer5D::select(
    const std::vector<Tensor>& candidates
) const
{
    if (candidates.empty()) {
        return Tensor{};
    }

    std::size_t best_index = 0;

    float best_score =
        -std::numeric_limits<float>::infinity();

    bool found_valid_candidate = false;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].empty()) {
            continue;
        }

        const float current_score =
            evaluate(candidates[i]);

        if (!accept(current_score)) {
            continue;
        }

        if (!found_valid_candidate ||
            current_score > best_score) {

            best_score = current_score;
            best_index = i;
            found_valid_candidate = true;
        }
    }

    /*
     * Không có candidate nào đạt ngưỡng.
     * 5D từ chối toàn bộ thay vì ép chọn một kết quả xấu.
     */
    if (!found_valid_candidate) {
        return Tensor{};
    }

    return candidates[best_index];
}

float Layer5D::evaluate(const Tensor& input) const
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

bool Layer5D::accept(float score) const
{
    return score >= threshold_;
}

float Layer5D::threshold() const
{
    return threshold_;
}

void Layer5D::set_threshold(float value)
{
    if (value < 0.0f) {
        throw std::invalid_argument(
            "AI5D: Layer5D threshold cannot be negative."
        );
    }

    threshold_ = value;
}

} // namespace ai5d::layers
