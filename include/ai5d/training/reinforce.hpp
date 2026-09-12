#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::training {

/**
 * @brief Công cụ học REINFORCE (policy gradient) — dùng riêng cho
 * các quyết định RỜI RẠC ở 4D/5D (chọn candidate, accept/reject),
 * nơi backpropagation thường không truyền được gradient qua.
 *
 * Không dùng module này cho 1D/2D/3D — những tầng đó dùng
 * mse_loss_gradient() (loss.hpp) + Optimizer (optimizer.hpp)
 * như bình thường.
 */

/**
 * @brief Chuyển 1 danh sách điểm số thô (chưa chuẩn hoá) thành
 * phân phối xác suất — softmax 1 chiều, có trừ max trước khi
 * exp để ổn định số học (numerically stable).
 *
 * @param scores Điểm số thô cho từng candidate/hành động.
 * @return Phân phối xác suất, cùng shape với scores, tổng = 1.0.
 */
Tensor to_probabilities(const Tensor& scores);

/**
 * @brief Chọn ngẫu nhiên 1 chỉ số theo phân phối xác suất đã cho.
 *
 * Đây là bước "hành động" (action) thật sự của policy — không
 * phải luôn chọn chỉ số có xác suất cao nhất (đó là hành vi cũ,
 * "chọn cứng"). Có tính ngẫu nhiên để mô hình có cơ hội khám phá
 * (exploration) trong quá trình học.
 *
 * @param probabilities Phân phối xác suất từ to_probabilities().
 * @return Chỉ số đã được chọn.
 */
std::size_t sample_index(const Tensor& probabilities);

/**
 * @brief Tính gradient của policy loss theo score gốc (trước
 * softmax), dùng công thức REINFORCE dạng đóng (closed-form)
 * cho phân phối softmax:
 *
 *     d(loss)/d(score_i) = reward * (probabilities[i] - onehot(chosen_index)[i])
 *
 * Gradient này đưa thẳng vào Optimizer::update() giống hệt
 * gradient từ backpropagation thông thường — không cần autograd,
 * vì công thức đạo hàm của softmax + log-likelihood đã có dạng
 * đóng (closed-form) sẵn.
 *
 * @param probabilities Phân phối xác suất đã dùng để sample.
 * @param chosen_index Chỉ số hành động đã chọn (từ sample_index()).
 * @param reward Reward nhận được sau khi thực hiện hành động này
 *               (từ compute_reward() trong loss.hpp).
 *               Reward dương -> tăng xác suất chọn lại hành động này.
 *               Reward âm -> giảm xác suất chọn lại hành động này.
 * @return Gradient, cùng shape với probabilities.
 *
 * @throws std::invalid_argument nếu chosen_index >= probabilities.size().
 */
Tensor policy_gradient(
    const Tensor& probabilities,
    std::size_t chosen_index,
    float reward
);

/**
 * @brief Ghi lại 1 bước quyết định (action) trong 1 episode.
 *
 * Dùng để tích luỹ nhiều quyết định (vd: nhiều lần 4D chọn Top-K
 * trong 1 lần chạy) trước khi tính baseline/gradient.
 */
struct ActionRecord {
    Tensor probabilities;
    std::size_t chosen_index = 0;
    float reward = 0.0f;
};

/**
 * @brief Tính baseline = trung bình cộng reward trong danh sách
 * ActionRecord.
 *
 * Trừ baseline vào reward từng bước trước khi tính policy_gradient()
 * là kỹ thuật chuẩn để giảm phương sai (variance) của REINFORCE,
 * giúp quá trình học ổn định hơn — không đổi kỳ vọng gradient,
 * chỉ giảm nhiễu.
 *
 * @param records Danh sách các quyết định đã ghi lại trong episode.
 * @return Baseline. Trả về 0.0f nếu records rỗng.
 */
float compute_baseline(const std::vector<ActionRecord>& records);

} // namespace ai5d::training
