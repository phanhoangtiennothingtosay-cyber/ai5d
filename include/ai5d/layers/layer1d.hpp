#pragma once

#include <cstddef>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng 1D của AI5D.
 *
 * Layer1D là tầng xử lý cơ sở, làm việc trực tiếp với
 * dữ liệu vector trước khi chuyển lên tầng 2D.
 *
 * WARNING:
 * Đây là phiên bản BETA / EXPERIMENTAL.
 * API và thuật toán có thể thay đổi.
 */
class Layer1D {
public:
    Layer1D() = default;
    ~Layer1D() = default;

    /**
     * @brief Xử lý một Tensor đầu vào.
     *
     * @param input Dữ liệu 1D đầu vào.
     * @return Tensor kết quả.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Xử lý một phần tử dữ liệu.
     *
     * Hàm này là đơn vị xử lý nhỏ nhất của Layer1D.
     *
     * @param value Giá trị đầu vào.
     * @return Giá trị sau xử lý.
     */
    float process(float value) const;

    /**
     * @brief Số phần tử đầu vào mà layer có thể xử lý.
     */
    std::size_t input_size() const;

    /**
     * @brief Đặt kích thước đầu vào.
     */
    void set_input_size(std::size_t size);

private:
    std::size_t input_size_ = 0;
};

} // namespace ai5d::layers
