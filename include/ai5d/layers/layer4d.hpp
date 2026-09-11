#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng đánh giá và lọc candidate từ Layer3D.
 *
 * Layer4D nhận nhiều candidate, đánh giá chất lượng,
 * xếp hạng và giữ lại Top-K candidate tốt nhất.
 */
class Layer4D {
public:
    Layer4D() = default;
    ~Layer4D() = default;

    /**
     * @brief Đưa một Tensor qua Layer4D.
     *
     * @param input Tensor đầu vào.
     * @return Tensor sau khi xử lý.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Xếp hạng các candidate.
     *
     * @param candidates Danh sách candidate.
     * @return Các candidate theo thứ tự chất lượng giảm dần.
     */
    std::vector<Tensor> rank(
        const std::vector<Tensor>& candidates
    ) const;

    /**
     * @brief Chọn Top-K candidate.
     *
     * @param candidates Danh sách candidate.
     * @param k Số lượng candidate cần giữ lại.
     * @return Top-K candidate.
     */
    std::vector<Tensor> top_k(
        const std::vector<Tensor>& candidates,
        std::size_t k
    ) const;

    /**
     * @brief Đánh giá chất lượng một candidate.
     *
     * @param input Candidate cần đánh giá.
     * @return Điểm chất lượng.
     */
    float evaluate(const Tensor& input) const;

    /**
     * @brief Lấy số lượng candidate tối đa được giữ lại.
     */
    std::size_t top_k_count() const;

    /**
     * @brief Đặt số lượng candidate tối đa được giữ lại.
     *
     * @param count Số lượng candidate.
     */
    void set_top_k_count(std::size_t count);

private:
    std::size_t top_k_count_ = 3;
};

} // namespace ai5d::layers
