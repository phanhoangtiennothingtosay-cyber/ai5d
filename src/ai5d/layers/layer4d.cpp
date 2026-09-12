#include "ai5d/layers/layer4d.hpp"

#include <algorithm>

#include "ai5d/math/topk.hpp"

namespace ai5d::layers {

Tensor Layer4D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    /*
     * Ở đây forward() xử lý một candidate đơn.
     *
     * Layer4D về bản chất là tầng quản lý candidate,
     * vì vậy phép ranking / Top-K nằm trong rank()
     * và top_k().
     */
    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i];
    }

    return output;
}

std::vector<Tensor> Layer4D::rank(
    const std::vector<Tensor>& candidates
) const
{
    if (candidates.empty()) {
        return {};
    }

    std::vector<Tensor> ranked = candidates;

    std::stable_sort(
        ranked.begin(),
        ranked.end(),
        [this](const Tensor& lhs, const Tensor& rhs) {
            return evaluate(lhs) > evaluate(rhs);
        }
    );

    return ranked;
}

std::vector<Tensor> Layer4D::top_k(
    const std::vector<Tensor>& candidates,
    std::size_t k
) const
{
    if (candidates.empty() || k == 0) {
        return {};
    }

    std::vector<math::RankedCandidate> ranked_candidates;
    ranked_candidates.reserve(candidates.size());

    for (const Tensor& candidate : candidates) {
        ranked_candidates.push_back({candidate, evaluate(candidate)});
    }

    const std::vector<math::RankedCandidate> top =
        math::top_k(ranked_candidates, k);

    std::vector<Tensor> result;
    result.reserve(top.size());

    for (const math::RankedCandidate& item : top) {
        result.push_back(item.value);
    }

    return result;
}

float Layer4D::evaluate(const Tensor& input) const
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

std::size_t Layer4D::top_k_count() const
{
    return top_k_count_;
}

void Layer4D::set_top_k_count(std::size_t count)
{
    top_k_count_ = count;
}

} // namespace ai5d::layers
