#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng 2D của AI5D.
 *
 * Layer2D nhận dữ liệu từ các worker 1D, tổ chức chúng
 * thành các nhóm/matrix và thực hiện xử lý ở cấp 2D.
 *
 * WARNING:
 * Đây là phiên bản BETA / EXPERIMENTAL.
 * API và thuật toán có thể thay đổi.
 */
class Layer2D {
public:
    Layer2D() = default;
    ~Layer2D() = default;

    /**
     * @brief Xử lý Tensor đầu vào từ tầng 1D.
     *
     * @param input Dữ liệu đầu vào.
     * @return Tensor kết quả 2D.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Gom nhiều kết quả 1D thành một Tensor 2D.
     *
     * @param inputs Các kết quả từ worker 1D.
     * @return Tensor 2D đã được tổ chức.
     */
    Tensor combine(
        const std::vector<Tensor>& inputs
    ) const;

    /**
     * @brief Chấm điểm một candidate.
     *
     * Score được sử dụng ở các tầng phía trên để
     * đánh giá và lựa chọn kết quả.
     *
     * @param candidate Candidate cần đánh giá.
     * @return Điểm số.
     */
    float score(const Tensor& candidate) const;

    /**
     * @brief Số worker 1D thuộc Layer2D.
     */
    std::size_t worker_count() const;

    /**
     * @brief Đặt số worker 1D.
     */
    void set_worker_count(std::size_t count);

    /**
     * @brief Kích thước ma trận 2D.
     */
    std::size_t rows() const;
    std::size_t cols() const;

    /**
     * @brief Đặt kích thước ma trận 2D.
     */
    void set_shape(
        std::size_t rows,
        std::size_t cols
    );

private:
    std::size_t worker_count_ = 0;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

} // namespace ai5d::layers
