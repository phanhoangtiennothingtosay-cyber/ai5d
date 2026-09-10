#pragma once

#include <cstddef>
#include <vector>
#include <initializer_list>

namespace ai5d {

/**
 * @brief Tensor dữ liệu cơ bản của AI5D.
 *
 * Tensor sử dụng bộ nhớ liên tục (contiguous memory).
 * Đây là lớp dữ liệu nền cho các tầng 1D, 2D, 3D, 4D và 5D.
 *
 * WARNING:
 * Đây là phiên bản BETA / EXPERIMENTAL.
 * API và cấu trúc nội bộ có thể thay đổi.
 */
class Tensor {
public:
    Tensor();

    /**
     * @brief Tạo Tensor với kích thước cho trước.
     *
     * Ví dụ:
     *     Tensor t({2, 3});
     */
    explicit Tensor(const std::vector<std::size_t>& shape);

    /**
     * @brief Tạo Tensor từ dữ liệu 1D.
     */
    Tensor(std::initializer_list<float> data);

    ~Tensor();

    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------

    /**
     * @brief Truy cập dữ liệu thô.
     */
    float* data();

    /**
     * @brief Truy cập dữ liệu chỉ đọc.
     */
    const float* data() const;

    /**
     * @brief Tổng số phần tử.
     */
    std::size_t size() const;

    /**
     * @brief Số chiều của Tensor.
     */
    std::size_t ndim() const;

    /**
     * @brief Kích thước từng chiều.
     */
    const std::vector<std::size_t>& shape() const;

    // ---------------------------------------------------------------------
    // Element access
    // ---------------------------------------------------------------------

    /**
     * @brief Truy cập phần tử theo chỉ số phẳng.
     *
     * Dữ liệu được lưu liên tục trong memory.
     */
    float& operator[](std::size_t index);

    /**
     * @brief Truy cập phần tử chỉ đọc.
     */
    const float& operator[](std::size_t index) const;

    // ---------------------------------------------------------------------
    // Shape
    // ---------------------------------------------------------------------

    /**
     * @brief Thay đổi shape mà không thay đổi dữ liệu.
     *
     * Tensor mới phải có cùng số phần tử.
     */
    void reshape(const std::vector<std::size_t>& new_shape);

    /**
     * @brief Kiểm tra Tensor có rỗng hay không.
     */
    bool empty() const;

    /**
     * @brief Xóa toàn bộ dữ liệu.
     */
    void clear();

    // ---------------------------------------------------------------------
    // Raw vector access
    // ---------------------------------------------------------------------

    /**
     * @brief Lấy vector dữ liệu.
     */
    std::vector<float>& vector();

    /**
     * @brief Lấy vector dữ liệu chỉ đọc.
     */
    const std::vector<float>& vector() const;

private:
    // Dữ liệu được lưu liên tục để thuận tiện cho CPU cache/SIMD.
    std::vector<float> data_;

    // Shape của Tensor.
    std::vector<std::size_t> shape_;

    /**
     * @brief Tính tổng số phần tử từ shape.
     */
    static std::size_t calculate_size(
        const std::vector<std::size_t>& shape
    );
};

} // namespace ai5d
