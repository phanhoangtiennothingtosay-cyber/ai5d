#include "ai5d/training/optimizer.hpp"

#include <stdexcept>

namespace ai5d::training {

void Optimizer::update_all(const std::vector<WeightGradientPair>& pairs)
{
    for (const WeightGradientPair& pair : pairs) {
        if (pair.weights == nullptr || pair.gradients == nullptr) {
            throw std::invalid_argument(
                "AI5D: update_all received a null weights/gradients pointer."
            );
        }

        update(*pair.weights, *pair.gradients);
    }
}

SGD::SGD(float learning_rate)
    : learning_rate_(learning_rate)
{
}

void SGD::update(Tensor& weights, const Tensor& gradients)
{
    if (weights.size() != gradients.size()) {
        throw std::invalid_argument(
            "AI5D: SGD::update requires weights and gradients of the same size."
        );
    }

    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate_ * gradients[i];
    }
}

float SGD::learning_rate() const
{
    return learning_rate_;
}

void SGD::set_learning_rate(float learning_rate)
{
    learning_rate_ = learning_rate;
}

} // namespace ai5d::training
