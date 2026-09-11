#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ai5d {

/**
 * @brief Tensor dữ liệu cơ bản của AI5D.
 *
 * Tensor sử dụng:
 * - float32 làm scalar mặc định.
 * - contiguous storage.
 * - row-major layout cho Tensor từ 2D trở lên.
 * - deep copy.
 *
 * Shape:
 *   [D]       -> vector 1D
 *   [R, C]    -> matrix 2D
 *   [D0,...]  -> Tensor nhiều chiều
 */
class Tensor {
public:
    Tensor();
    explicit Tensor(const std::vector<std::size_t>& shape);

    /**
     * @brief Tạo Tensor từ dữ liệu thô.
     *
     * Ví dụ:
     * Tensor{1.0f, 2.0f, 3.0f}
     *
     * sẽ có shape [3].
     */
    Tensor(std::initializer_list<float> data);

    ~Tensor();

    // Explicit copy semantics.
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // Explicit move semantics.
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    /**
     * @brief Tạo Tensor chỉ từ shape.
     *
     * Đây là API rõ ràng để phân biệt shape
     * với dữ liệu thô.
     *
     * Ví dụ:
     * Tensor::from_shape({2, 3});
     */
    static Tensor from_shape(
        const std::vector<std::size_t>& shape
    );

    /**
     * @brief Truy cập vùng dữ liệu liên tục.
     *
     * Pointer chỉ hợp lệ trong lifetime của Tensor
     * và có thể mất hiệu lực sau reshape/resize.
     */
    float* data();
    const float* data() const;

    /**
     * @brief Tổng số phần tử.
     */
    std::size_t size() const;

    /**
     * @brief Số chiều.
     */
    std::size_t ndim() const;

    /**
     * @brief Shape của Tensor.
     */
    const std::vector<std::size_t>& shape() const;

    /**
     * @brief Truy cập phần tử theo flat index.
     *
     * @throws std::out_of_range nếu index không hợp lệ.
     */
    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    /**
     * @brief Thay đổi shape nhưng không thay đổi dữ liệu.
     *
     * Tổng số phần tử trước và sau phải giống nhau.
     */
    void reshape(
        const std::vector<std::size_t>& new_shape
    );

    /**
     * @brief Kiểm tra Tensor có rỗng hay không.
     */
    bool empty() const;

    /**
     * @brief Xóa toàn bộ dữ liệu và shape.
     */
    void clear();

    /**
     * @brief Truy cập vector dữ liệu thô.
     */
    std::vector<float>& vector();
    const std::vector<float>& vector() const;

    /**
     * @brief Kiểm tra Tensor có chứa NaN hay không.
     *
     * Không tự động gọi trong các hot path.
     */
    bool has_nan() const;

    /**
     * @brief Kiểm tra Tensor có chứa Inf hay không.
     *
     * Không tự động gọi trong các hot path.
     */
    bool has_inf() const;

private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;

    /**
     * @brief Tính tổng số phần tử từ shape.
     *
     * Có kiểm tra:
     * - dimension bằng 0
     * - overflow của std::size_t
     */
    static std::size_t calculate_size(
        const std::vector<std::size_t>& shape
    );
};

} // namespace ai5d
