#include "ai5d/training/loss.hpp"

#include <stdexcept>
#include <string>

namespace ai5d::training {

namespace {

void check_same_shape(const Tensor& a, const Tensor& b, const char* function_name)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::string("AI5D: ") + function_name +
            " requires prediction and target of the same size."
        );
    }
}

} // namespace

float mse_loss(const Tensor& prediction, const Tensor& target)
{
    check_same_shape(prediction, target, "mse_loss");

    if (prediction.empty()) {
        return 0.0f;
    }

    float sum_squared_error = 0.0f;
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        const float diff = prediction[i] - target[i];
        sum_squared_error += diff * diff;
    }

    return sum_squared_error / static_cast<float>(prediction.size());
}

Tensor mse_loss_gradient(const Tensor& prediction, const Tensor& target)
{
    check_same_shape(prediction, target, "mse_loss_gradient");

    if (prediction.empty()) {
        return Tensor{};
    }

    Tensor gradient = Tensor::from_shape({prediction.size()});

    const float scale = 2.0f / static_cast<float>(prediction.size());

    for (std::size_t i = 0; i < prediction.size(); ++i) {
        gradient[i] = scale * (prediction[i] - target[i]);
    }

    return gradient;
}

float compute_reward(const Tensor& prediction, const Tensor& target)
{
    // Không cần check_same_shape() riêng — mse_loss() đã tự kiểm tra.
    return -mse_loss(prediction, target);
}

} // namespace ai5d::training
