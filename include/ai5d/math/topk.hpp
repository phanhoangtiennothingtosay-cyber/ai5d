#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::math {

struct RankedCandidate {
    Tensor value;
    float score = 0.0f;
};

std::vector<RankedCandidate> top_k(
    const std::vector<RankedCandidate>& candidates,
    std::size_t k
);

std::vector<std::size_t> top_k_indices(
    const std::vector<float>& scores,
    std::size_t k
);

std::vector<float> top_k_scores(
    const std::vector<float>& scores,
    std::size_t k
);

} // namespace ai5d::math
