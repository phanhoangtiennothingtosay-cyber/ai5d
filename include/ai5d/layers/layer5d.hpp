#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

class Layer5D {
public:
    Layer5D() = default;
    ~Layer5D() = default;

    Tensor forward(const Tensor& input) const;

    Tensor decide(const std::vector<Tensor>& candidates) const;

    float evaluate(const Tensor& candidate) const;

    bool accept(const Tensor& candidate) const;

    bool should_reject(const std::vector<Tensor>& candidates) const;

    std::size_t candidate_count() const;

    void set_candidate_count(std::size_t count);

    float acceptance_threshold() const;

    void set_acceptance_threshold(float threshold);

private:
    std::size_t candidate_count_ = 3;
    float acceptance_threshold_ = 0.0f;
};

} // namespace ai5d::layers
