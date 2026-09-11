#include "ai5d/routing/compute_router.hpp"

#include <algorithm>

namespace ai5d::routing {

ComputeMode ComputeRouter::select_mode(const Tensor& input) const
{
    if (requires_full_compute(input)) {
        return ComputeMode::FULL;
    }

    if (requires_deep_compute(input)) {
        return ComputeMode::DEEP;
    }

    const float budget = compute_budget(input);

    if (budget <= 0.10f) {
        return ComputeMode::FAST;
    }

    return ComputeMode::NORMAL;
}

float ComputeRouter::compute_budget(const Tensor& input) const
{
    if (input.empty()) {
        return 0.0f;
    }

    const std::size_t size = input.size();

    if (size < 128) {
        return 0.10f;
    }

    if (size < 1024) {
        return 0.25f;
    }

    if (size < 4096) {
        return 0.50f;
    }

    return 1.00f;
}

std::size_t ComputeRouter::recommended_3d_count(
    const Tensor& input,
    std::size_t available_3d
) const
{
    if (input.empty() || available_3d == 0) {
        return 0;
    }

    const float budget = compute_budget(input);

    std::size_t recommended =
        static_cast<std::size_t>(
            static_cast<float>(available_3d) * budget
        );

    if (recommended == 0) {
        recommended = 1;
    }

    return std::min(recommended, available_3d);
}

bool ComputeRouter::requires_deep_compute(
    const Tensor& input
) const
{
    if (input.empty()) {
        return false;
    }

    return input.size() >= 4096;
}

bool ComputeRouter::requires_full_compute(
    const Tensor& input
) const
{
    if (input.empty()) {
        return false;
    }

    return input.size() >= 16384;
}

void ComputeRouter::set_config(const Config& config)
{
    config_ = config;
}

const Config& ComputeRouter::config() const
{
    return config_;
}

} // namespace ai5d::routing