#include "ai5d/layers/layer4d.hpp"

#include <algorithm>
#include <stdexcept>

namespace ai5d::layers {

Tensor Layer4D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    return input;
}

Tensor Layer4D::evaluate(const std::vector<Tensor>& candidates) const
{
    if (candidates.empty()) {
        return Tensor{};
    }

    const Tensor* best = &candidates.front();
    float best_score = score(*best);

    for (std::size_t i = 1; i < candidates.size(); ++i) {
        const float current_score = score(candidates[i]);

        if (current_score > best_score) {
            best = &candidates[i];
            best_score = current_score;
        }
    }

    return *best;
}

Tensor Layer4D::rank(const std::vector<Tensor>& candidates) const
{
    return evaluate(candidates);
}

std::vector<Tensor> Layer4D::select_top_k(
    const std::vector<Tensor>& candidates,
    std::size_t k
) const
{
    if (candidates.empty() || k == 0) {
        return {};
    }

    std::vector<Tensor> ranked = candidates;

    std::sort(
        ranked.begin(),
        ranked.end(),
        [this](const Tensor& lhs, const Tensor& rhs) {
            return score(lhs) > score(rhs);
        }
    );

    if (ranked.size() > k) {
        ranked.resize(k);
    }

    return ranked;
}

float Layer4D::score(const Tensor& candidate) const
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

std::size_t Layer4D::worker_count() const
{
    return worker_count_;
}

void Layer4D::set_worker_count(std::size_t count)
{
    worker_count_ = count;
}

std::size_t Layer4D::top_k() const
{
    return top_k_;
}

void Layer4D::set_top_k(std::size_t k)
{
    top_k_ = k;
}

bool Layer4D::has_valid_candidate(
    const std::vector<Tensor>& candidates
) const
{
    for (const Tensor& candidate : candidates) {
        if (!candidate.empty()) {
            return true;
        }
    }

    return false;
}

} // namespace ai5d::layers