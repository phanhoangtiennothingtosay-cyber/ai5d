#pragma once

#include <cstddef>
#include <vector>

namespace ai5d {

/**
 * @brief Bộ não điều phối toàn bộ kiến trúc AI5D.
 *
 * Brain là lớp cấp cao nhất của thư viện.
 * Người dùng chỉ cần làm việc với Brain thay vì trực tiếp
 * quản lý từng Layer 1D, 2D, 3D, 4D và 5D.
 */
class Brain {
public:
    /**
     * @brief Khởi tạo Brain với cấu hình mặc định.
     */
    Brain();

    /**
     * @brief Hủy Brain.
     */
    ~Brain();

    /**
     * @brief Nạp dữ liệu đầu vào dạng 1D.
     *
     * Ví dụ:
     *     ai5d::Brain brain;
     *     brain.in_1d({1.0f, 2.0f, 3.0f});
     *
     * @param data Vector dữ liệu đầu vào.
     * @return Brain& để hỗ trợ method chaining.
     */
    Brain& in_1d(const std::vector<float>& data);

    /**
     * @brief Chuyển dữ liệu từ tầng 1D sang 2D.
     *
     * @return Brain& để tiếp tục chuỗi xử lý.
     */
    Brain& to_2d();

    /**
     * @brief Chạy tầng xử lý 3D.
     *
     * Tầng này chịu trách nhiệm xử lý nhiều nhóm 2D
     * và cơ chế giao tiếp giữa các node 3D.
     *
     * @return Brain&.
     */
    Brain& talk_3d();

    /**
     * @brief Chạy tầng 4D.
     *
     * 4D tiếp nhận các kết quả từ hệ thống 3D,
     * xử lý bằng các worker nội bộ và chọn Top-K.
     *
     * @return Brain&.
     */
    Brain& filter_4d();

    /**
     * @brief Chạy tầng 5D.
     *
     * 5D nhận các ứng viên tốt nhất từ 4D
     * và đưa ra quyết định cuối cùng.
     *
     * @return Brain&.
     */
    Brain& get_5d();

    /**
     * @brief Lấy kết quả cuối cùng.
     *
     * @return Vector kết quả.
     */
    const std::vector<float>& result() const;

    /**
     * @brief Xóa trạng thái xử lý hiện tại.
     *
     * Cho phép Brain được tái sử dụng cho một input mới.
     */
    void reset();

private:
    // Dữ liệu đầu vào ban đầu.
    std::vector<float> input_;

    // Kết quả hiện tại trong pipeline.
    std::vector<float> result_;

    // Trạng thái pipeline.
    bool has_input_;
    bool processed_1d_;
    bool processed_2d_;
    bool processed_3d_;
    bool processed_4d_;
    bool processed_5d_;
};

} // namespace ai5d
