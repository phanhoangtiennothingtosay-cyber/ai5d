#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

class Layer3D {
public:
    Layer3D() = default;
    ~Layer3D() = default;

    Tensor forward(const Tensor& input) const;

    Tensor communicate(const std::vector<Tensor>& peers) const;

    Tensor aggregate(const std::vector<Tensor>& inputs) const;

    float score(const Tensor& candidate) const;

    std::size_t group_count() const;
    void set_group_count(std::size_t count);

    std::size_t peer_count() const;
    void set_peer_count(std::size_t count);

    std::size_t candidate_count() const;
    void set_candidate_count(std::size_t count);

private:
    std::size_t group_count_ = 0;
    std::size_t peer_count_ = 0;
    std::size_t candidate_count_ = 0;
};

} // namespace ai5d::layers
