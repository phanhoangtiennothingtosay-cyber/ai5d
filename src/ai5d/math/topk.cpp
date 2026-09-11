#include "ai5d/math/topk.hpp"

#include <algorithm>

namespace ai5d::math {

std::vector<RankedCandidate> top_k(
    const std::vector<RankedCandidate>& candidates,
    std::size_t k
)
{
    if (candidates.empty() || k == 0) {
        return {};
    }

    std::vector<RankedCandidate> result = candidates;

    std::sort(
        result.begin(),
        result.end(),
        [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
            return lhs.score > rhs.score;
        }
    );

    if (result.size() > k) {
        result.resize(k);
    }

    return result;
}

std::vector<std::size_t> top_k_indices(
    const std::vector<float>& scores,
    std::size_t k
)
{
    if (scores.empty() || k == 0) {
        return {};
    }

    std::vector<std::size_t> indices(scores.size());

    for (std::size_t i = 0; i < scores.size(); ++i) {
        indices[i] = i;
    }

    std::sort(
        indices.begin(),
        indices.end(),
        [&scores](std::size_t lhs, std::size_t rhs) {
            return scores[lhs] > scores[rhs];
        }
    );

    if (indices.size() > k) {
        indices.resize(k);
    }

    return indices;
}

std::vector<float> top_k_scores(
    const std::vector<float>& scores,
    std::size_t k
)
{
    const std::vector<std::size_t> indices =
        top_k_indices(scores, k);

    std::vector<float> result;
    result.reserve(indices.size());

    for (const std::size_t index : indices) {
        result.push_back(scores[index]);
    }

    return result;
}

} // namespace ai5d::math