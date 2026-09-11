#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ai5d {

/**
 * @brief Core Tensor container của AI5D.
 *
 * Contract v0.1:
 * - Scalar type: float32.
 * - Storage: contiguous.
 * - Layout: row-major.
 * - Copy: deep copy.
 * - Move: noexcept.
 *
 * Shape:
 *   [D]    -> vector 1D
 *   [R, C] -> matrix 2D
 *   [D0, ...] -> Tensor nhiều chiều
 *
 * Tensor mặc định (Tensor{}) đại diện cho empty Tensor.
 */
class Tensor {
public:
    Tensor() = default;

    /**
     * @brief Tạo Tensor từ shape.
     *
     * @deprecated
     * Dùng Tensor::from_shape() cho code mới để tránh
     * ambiguity giữa shape và raw data.
     */
    [[deprecated(
        "Use Tensor::from_shape() for shape construction."
    )]]
    explicit Tensor(
        const std::vector<std::size_t>& shape
    );

    /**
     * @brief Tạo Tensor từ dữ liệu thô.
     *
     * Ví dụ:
     * Tensor values{1.0f, 2.0f, 3.0f};
     *
     * Shape sẽ là [3].
     */
    Tensor(std::initializer_list<float> data);

    ~Tensor() = default;

    // Explicit copy semantics.
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // Explicit move semantics.
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    /**
     * @brief Tạo Tensor từ shape một cách rõ ràng.
     *
     * Ví dụ:
     * Tensor matrix = Tensor::from_shape({2, 3});
     */
    static Tensor from_shape(
        const std::vector<std::size_t>& shape
    );

    /**
     * @brief Pointer tới vùng dữ liệu contiguous.
     *
     * Pointer chỉ hợp lệ trong lifetime của Tensor.
     * Pointer có thể mất hiệu lực sau khi Tensor được
     * reshape hoặc thay đổi storage.
     */
    float* data();
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
     * @brief Shape của Tensor.
     */
    const std::vector<std::size_t>& shape() const;

    /**
     * @brief Truy cập phần tử bằng flat index.
     *
     * Tensor sử dụng contiguous row-major storage.
     *
     * @throws std::out_of_range nếu index >= size().
     */
    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    /**
     * @brief Thay đổi cách diễn giải shape.
     *
     * Không được thay đổi tổng số phần tử.
     *
     * @throws std::invalid_argument nếu shape mới
     *         không có cùng số phần tử.
     */
    void reshape(
        const std::vector<std::size_t>& new_shape
    );

    /**
     * @brief Kiểm tra Tensor có rỗng hay không.
     */
    bool empty() const;

    /**
     * @brief Xóa dữ liệu và shape.
     */
    void clear();

    /**
     * @brief Truy cập vector storage.
     */
    std::vector<float>& vector();
    const std::vector<float>& vector() const;

    /**
     * @brief Kiểm tra Tensor có chứa NaN.
     *
     * Không được tự động gọi trong hot path.
     */
    bool has_nan() const;

    /**
     * @brief Kiểm tra Tensor có chứa Inf.
     *
     * Không được tự động gọi trong hot path.
     */
    bool has_inf() const;

private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;

    /**
     * @brief Tính tổng số phần tử từ shape.
     *
     * Kiểm tra:
     * - dimension == 0
     * - size_t multiplication overflow
     */
    static std::size_t calculate_size(
        const std::vector<std::size_t>& shape
    );
};

} // namespace ai5d
