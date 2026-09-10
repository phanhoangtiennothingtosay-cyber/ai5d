#pragma once

#include <cstddef>

#include "ai5d/config.hpp"
#include "ai5d/tensor.hpp"

namespace ai5d::routing {

class ComputeRouter {
public:
    ComputeRouter() = default;
    ~ComputeRouter() = default;

    ComputeMode select_mode(const Tensor& input) const;

    float compute_budget(const Tensor& input) const;

    std::size_t recommended_3d_count(
        const Tensor& input,
        std::size_t available_3d
    ) const;

    bool requires_deep_compute(const Tensor& input) const;

    bool requires_full_compute(const Tensor& input) const;

    void set_config(const Config& config);

    const Config& config() const;

private:
    Config config_{};
};

} // namespace ai5d::routing
