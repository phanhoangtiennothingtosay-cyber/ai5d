#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

class Layer4D {
public:
    Layer4D() = default;
    ~Layer4D() = default;

    Tensor forward(const Tensor& input) const;

    Tensor evaluate(const std::vector<Tensor>& candidates) const;

    Tensor rank(const std::vector<Tensor>& candidates) const;

    std::vector<Tensor> select_top_k(
        const std::vector<Tensor>& candidates,
        std::size_t k
    ) const;

    float score(const Tensor& candidate) const;

    std::size_t worker_count() const;
    void set_worker_count(std::size_t count);

    std::size_t top_k() const;
    void set_top_k(std::size_t k);

    bool has_valid_candidate(const std::vector<Tensor>& candidates) const;

private:
    std::size_t worker_count_ = 0;
    std::size_t top_k_ = 3;
};

} // namespace ai5d::layers
