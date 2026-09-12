#pragma once

#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::training {

/**
 * @brief Một cặp (trọng số, gradient) cần cập nhật cùng nhau.
 *
 * Dùng khi 1 Layer có nhiều trọng số riêng biệt (vd: weights + bias)
 * và muốn cập nhật tất cả trong 1 lệnh gọi duy nhất.
 */
struct WeightGradientPair {
    Tensor* weights = nullptr;
    const Tensor* gradients = nullptr;
};

/**
 * @brief Interface chung cho các thuật toán tối ưu (optimizer).
 *
 * Optimizer chịu trách nhiệm cập nhật trọng số dựa trên gradient
 * đã tính được từ backpropagation (phần "trơn tru" của AI5D:
 * 1D/2D/3D — không dùng cho các quyết định rời rạc ở 4D/5D,
 * xem ai5d::training::reinforce cho phần đó).
 *
 * Optimizer là thành phần NỘI BỘ của AI5D — người dùng cuối
 * (Python API) không thao tác trực tiếp với class này. Người
 * dùng chỉ gọi Brain::train(...); Brain sẽ tự dùng Optimizer
 * bên trong.
 *
 * Optimizer KHÔNG tự tính gradient — nó chỉ nhận gradient đã
 * tính sẵn (từ backward() của Layer) và quyết định cách áp dụng
 * gradient đó vào trọng số.
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /**
     * @brief Cập nhật 1 trọng số dựa trên gradient tương ứng.
     *
     * @param weights Trọng số cần cập nhật (bị thay đổi trực tiếp).
     * @param gradients Gradient của loss theo từng trọng số.
     *                  Phải cùng shape với weights.
     *
     * @throws std::invalid_argument nếu weights và gradients
     *         không cùng shape.
     */
    virtual void update(Tensor& weights, const Tensor& gradients) = 0;

    /**
     * @brief Cập nhật nhiều trọng số cùng lúc (vd: weights + bias
     *        của cùng 1 Layer).
     *
     * Mặc định: lần lượt gọi update() cho từng cặp trong danh sách.
     * Optimizer có trạng thái phức tạp hơn (vd: cần biết thứ tự
     * cập nhật liên quan tới nhau) có thể override hàm này.
     *
     * @param pairs Danh sách các cặp (weights, gradients).
     */
    virtual void update_all(const std::vector<WeightGradientPair>& pairs);

    /**
     * @brief Reset trạng thái nội bộ của optimizer (nếu có).
     *
     * Với SGD thuần, hàm này không làm gì cả — optimizer không có
     * trạng thái để reset. Optimizer có trạng thái (Momentum, Adam...)
     * ở phiên bản sau sẽ override hàm này để xóa lịch sử tích lũy.
     */
    virtual void reset() {}
};

/**
 * @brief Stochastic Gradient Descent — optimizer đơn giản nhất.
 *
 * Công thức cập nhật:
 *     weight = weight - learning_rate * gradient
 *
 * Không có trạng thái nội bộ (stateless) — reset() không cần làm gì.
 */
class SGD : public Optimizer {
public:
    /**
     * @param learning_rate Tốc độ học. Giá trị càng lớn, mỗi bước
     *                       cập nhật càng "nhảy" xa — dễ học nhanh
     *                       nhưng cũng dễ học không ổn định.
     */
    explicit SGD(float learning_rate = 0.01f);

    void update(Tensor& weights, const Tensor& gradients) override;

    float learning_rate() const;
    void set_learning_rate(float learning_rate);

private:
    float learning_rate_;
};

/*
 * GHI CHÚ CHO PHIÊN BẢN SAU (chưa implement ở v0.1):
 *
 * Các optimizer có trạng thái (Momentum, Adam...) sẽ kế thừa
 * Optimizer tương tự SGD, nhưng cần thêm Tensor nội bộ để lưu
 * lịch sử gradient (momentum/velocity). Không thêm ở đây để
 * tránh mở rộng phạm vi ngoài yêu cầu hiện tại.
 */

} // namespace ai5d::training
