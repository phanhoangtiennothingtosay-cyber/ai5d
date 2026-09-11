#include "ai5d/layers/layer5d.hpp"

#include <algorithm>

namespace ai5d::layers {

Tensor Layer5D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    return input;
}

Tensor Layer5D::decide(const std::vector<Tensor>& candidates) const
{
    if (candidates.empty() || should_reject(candidates)) {
        return Tensor{};
    }

    const Tensor* best = &candidates.front();
    float best_score = evaluate(*best);

    for (std::size_t i = 1; i < candidates.size(); ++i) {
        const float current_score = evaluate(candidates[i]);

        if (current_score > best_score) {
            best = &candidates[i];
            best_score = current_score;
        }
    }

    return *best;
}

float Layer5D::evaluate(const Tensor& candidate) const
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

bool Layer5D::accept(const Tensor& candidate) const
{
    if (candidate.empty()) {
        return false;
    }

    return evaluate(candidate) >= acceptance_threshold_;
}

bool Layer5D::should_reject(
    const std::vector<Tensor>& candidates
) const
{
    if (candidates.empty()) {
        return true;
    }

    for (const Tensor& candidate : candidates) {
        if (accept(candidate)) {
            return false;
        }
    }

    return true;
}

std::size_t Layer5D::candidate_count() const
{
    return candidate_count_;
}

void Layer5D::set_candidate_count(std::size_t count)
{
    candidate_count_ = count;
}

float Layer5D::acceptance_threshold() const
{
    return acceptance_threshold_;
}

void Layer5D::set_acceptance_threshold(float threshold)
{
    acceptance_threshold_ = threshold;
}

} // namespace ai5d::layers