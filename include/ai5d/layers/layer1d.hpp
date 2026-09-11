#pragma once

#include <cstddef>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng xử lý 1D cơ bản của AI5D.
 *
 * Layer1D là đơn vị xử lý thấp nhất.
 *
 * Nhiệm vụ:
 * - Nhận dữ liệu Tensor.
 * - Xử lý từng phần tử thông qua process().
 * - Trả về Tensor đã được xử lý.
 *
 * Layer1D chưa thực hiện grouping hay ranking.
 * Những nhiệm vụ đó thuộc các tầng cao hơn.
 */
class Layer1D {
public:
    Layer1D() = default;
    ~Layer1D() = default;

    /**
     * @brief Xử lý toàn bộ Tensor đầu vào.
     *
     * @param input Tensor đầu vào.
     * @return Tensor sau khi xử lý.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Xử lý một phần tử.
     *
     * Đây là đơn vị tính toán cơ bản của Layer1D.
     *
     * @param value Giá trị đầu vào.
     * @return Giá trị sau xử lý.
     */
    float process(float value) const;

    /**
     * @brief Lấy kích thước đầu vào đã cấu hình.
     *
     * @return Số phần tử đầu vào.
     *
     * Giá trị 0 nghĩa là chưa cấu hình kích thước cố định.
     */
    std::size_t input_size() const;

    /**
     * @brief Đặt kích thước đầu vào.
     *
     * @param size Số phần tử mà Layer1D dự kiến xử lý.
     */
    void set_input_size(std::size_t size);

private:
    std::size_t input_size_ = 0;
};

} // namespace ai5d::layers
