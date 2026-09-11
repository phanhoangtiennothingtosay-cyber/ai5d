#pragma once

#include <cstddef>
#include <vector>

#include "ai5d/tensor.hpp"

namespace ai5d::layers {

/**
 * @brief Tầng tổ chức và giao tiếp giữa các nhóm 2D.
 *
 * Layer3D nhận kết quả từ Layer2D, tạo nhiều candidate
 * và tổng hợp chúng thành kết quả tốt hơn.
 */
class Layer3D {
public:
    Layer3D() = default;
    ~Layer3D() = default;

    /**
     * @brief Đưa Tensor qua Layer3D.
     *
     * @param input Tensor đầu vào.
     * @return Tensor sau khi xử lý.
     */
    Tensor forward(const Tensor& input) const;

    /**
     * @brief Tổng hợp nhiều candidate.
     *
     * @param candidates Danh sách candidate từ các nhóm 2D.
     * @return Candidate được lựa chọn/tổng hợp.
     */
    Tensor aggregate(
        const std::vector<Tensor>& candidates
    ) const;

    /**
     * @brief Lấy số lượng nhóm 2D bên trong Layer3D.
     */
    std::size_t group_count() const;

    /**
     * @brief Đặt số lượng nhóm 2D.
     *
     * @param count Số nhóm.
     */
    void set_group_count(std::size_t count);

private:
    std::size_t group_count_ = 1;
};

} // namespace ai5d::layers
