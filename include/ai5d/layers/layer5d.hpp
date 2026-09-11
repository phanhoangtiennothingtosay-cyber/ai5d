#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng quyết định cuối cùng của AI5D.
 *
 * Layer5D nhận các candidate đã được Layer4D lọc,
 * đánh giá chúng theo tiêu chí cuối cùng và chọn
 * candidate phù hợp nhất.
 *
 * Layer5D cũng có thể từ chối toàn bộ candidate nếu
 * không candidate nào đạt ngưỡng chấp nhận.
 */
class Layer5D {
public:
    Layer5D() = default;
    ~Layer5D() = default;

    /**
     * @brief Đưa một Tensor qua Layer5D.
     *
     * @param input Tensor đầu vào.
     * @return Tensor sau quyết định cuối.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Chọn candidate tốt nhất.
     *
     * @param candidates Danh sách candidate từ Layer4D.
     * @return Candidate được chọn.
     *
     * Nếu không có candidate phù hợp, trả về Tensor rỗng.
     */
    Tensor select(
        const std::vector<Tensor>& candidates
    ) const;

    /**
     * @brief Đánh giá một candidate.
     *
     * @param input Candidate cần đánh giá.
     * @return Điểm đánh giá cuối.
     */
    float evaluate(const Tensor& input) const;

    /**
     * @brief Kiểm tra candidate có đủ điều kiện được chấp nhận không.
     *
     * @param score Điểm của candidate.
     * @return true nếu được chấp nhận.
     */
    bool accept(float score) const;

    /**
     * @brief Lấy ngưỡng chấp nhận.
     */
    float threshold() const;

    /**
     * @brief Đặt ngưỡng chấp nhận.
     *
     * @param value Ngưỡng mới.
     */
    void set_threshold(float value);

private:
    float threshold_ = 0.0f;
};

} // namespace ai5d::layers
