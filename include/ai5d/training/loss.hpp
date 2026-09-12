#pragma once

#include "ai5d/tensor.hpp"

namespace ai5d::training {

/**
 * @brief Các hàm loss/reward dùng chung cho cả 2 phần của AI5D:
 *
 * - Phần "trơn tru" (1D/2D/3D, học bằng backpropagation thường):
 *   dùng mse_loss() + mse_loss_gradient().
 *
 * - Phần "rời rạc" (4D/5D, học bằng REINFORCE — xem reinforce.hpp):
 *   dùng compute_reward() để biến kết quả cuối thành 1 con số
 *   reward duy nhất.
 */

/**
 * @brief Mean Squared Error — sai số bình phương trung bình.
 *
 *     loss = mean((prediction - target)^2)
 *
 * @param prediction Kết quả mô hình đưa ra.
 * @param target Kết quả mong muốn (đáp án đúng).
 *
 * @throws std::invalid_argument nếu prediction và target
 *         không cùng shape.
 */
float mse_loss(const Tensor& prediction, const Tensor& target);

/**
 * @brief Gradient của mse_loss() theo prediction.
 *
 *     d(loss)/d(prediction_i) = 2 * (prediction_i - target_i) / N
 *
 * Kết quả này đưa thẳng vào Layer::backward() (phần trơn tru,
 * 1D-3D) để bắt đầu lan truyền ngược.
 *
 * @throws std::invalid_argument nếu prediction và target
 *         không cùng shape.
 */
Tensor mse_loss_gradient(const Tensor& prediction, const Tensor& target);

/**
 * @brief Chuyển kết quả cuối cùng của Brain (sau get_5d()) và target
 * mong muốn thành 1 con số reward duy nhất, dùng cho REINFORCE
 * (xem ai5d::training::policy_gradient trong reinforce.hpp).
 *
 * Công thức v0.1 — ĐƠN GIẢN NHẤT có thể, không cố sáng tạo công
 * thức reward phức tạp hơn khi chưa thật sự cần:
 *
 *     reward = -mse_loss(prediction, target)
 *
 * loss thấp -> reward cao (gần 0 hoặc dương);
 * loss cao -> reward thấp (âm nhiều hơn).
 *
 * @throws std::invalid_argument nếu prediction và target
 *         không cùng shape.
 */
float compute_reward(const Tensor& prediction, const Tensor& target);

} // namespace ai5d::training
