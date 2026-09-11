#include "ai5d/layers/layer3d.hpp"

#include <stdexcept>

namespace ai5d::layers {

Tensor Layer3D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    return input;
}

Tensor Layer3D::communicate(const std::vector<Tensor>& peers) const
{
    if (peers.empty()) {
        return Tensor{};
    }

    return aggregate(peers);
}

Tensor Layer3D::aggregate(const std::vector<Tensor>& inputs) const
{
    if (inputs.empty()) {
        return Tensor{};
    }

    const Tensor* best = &inputs.front();
    float best_score = score(*best);

    for (std::size_t i = 1; i < inputs.size(); ++i) {
        const float current_score = score(inputs[i]);

        if (current_score > best_score) {
            best = &inputs[i];
            best_score = current_score;
        }
    }

    return *best;
}

float Layer3D::score(const Tensor& candidate) const
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

std::size_t Layer3D::group_count() const
{
    return group_count_;
}

void Layer3D::set_group_count(std::size_t count)
{
    group_count_ = count;
}

std::size_t Layer3D::peer_count() const
{
    return peer_count_;
}

void Layer3D::set_peer_count(std::size_t count)
{
    peer_count_ = count;
}

std::size_t Layer3D::candidate_count() const
{
    return candidate_count_;
}

void Layer3D::set_candidate_count(std::size_t count)
{
    candidate_count_ = count;
}

} // namespace ai5d::layers