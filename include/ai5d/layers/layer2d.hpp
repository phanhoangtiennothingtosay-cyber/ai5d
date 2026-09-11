#pragma once

#include <cstddef>
#include <vector>
#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng tổ chức và đánh giá các đơn vị 1D.
 *
 * Layer2D nhận dữ liệu từ Layer1D và tổ chức chúng
 * thành cấu trúc 2 chiều để các worker có thể xử lý
 * và đánh giá các kết quả.
 */
class Layer2D {
public:
    Layer2D() = default;
    ~Layer2D() = default;

    /**
     * @brief Đưa Tensor qua Layer2D.
     *
     * @param input Tensor đầu vào.
     * @return Tensor sau khi xử lý.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Kết hợp nhiều candidate thành một kết quả.
     *
     * @param candidates Danh sách các Tensor ứng viên.
     * @return Candidate được lựa chọn.
     */
    Tensor combine(const std::vector<Tensor>& candidates) const;

    /**
     * @brief Đánh giá một Tensor.
     *
     * @param input Tensor cần đánh giá.
     * @return Điểm số của Tensor.
     */
    float score(const Tensor& input) const;

    /**
     * @brief Lấy số worker của Layer2D.
     */
    std::size_t worker_count() const;

    /**
     * @brief Đặt số worker của Layer2D.
     *
     * @param count Số worker.
     */
    void set_worker_count(std::size_t count);

    /**
     * @brief Lấy số hàng của cấu trúc 2D.
     */
    std::size_t rows() const;

    /**
     * @brief Lấy số cột của cấu trúc 2D.
     */
    std::size_t cols() const;

    /**
     * @brief Đặt kích thước cấu trúc 2D.
     *
     * @param rows Số hàng.
     * @param cols Số cột.
     */
    void set_shape(
        std::size_t rows,
        std::size_t cols
    );

private:
    std::size_t worker_count_ = 1;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

} // namespace ai5d::layers
