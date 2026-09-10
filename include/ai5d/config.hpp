#pragma once

#include <cstddef>

namespace ai5d {

/**
 * @brief Chế độ tính toán của AI5D.
 *
 * FAST   : ưu tiên tốc độ, dùng ít tài nguyên.
 * NORMAL : chế độ mặc định.
 * DEEP   : tăng lượng xử lý.
 * FULL   : sử dụng toàn bộ compute budget.
 */
enum class ComputeMode {
    FAST,
    NORMAL,
    DEEP,
    FULL
};

/**
 * @brief Cấu hình chung cho Brain và các layer.
 *
 * WARNING:
 * Đây là phiên bản BETA / EXPERIMENTAL.
 * Cấu hình và API có thể thay đổi trong các phiên bản sau.
 */
struct Config {

    // ---------------------------------------------------------------------
    // Compute
    // ---------------------------------------------------------------------

    ComputeMode compute_mode = ComputeMode::NORMAL;

    /**
     * @brief Compute budget tương đối.
 *
     * 0.10 = khoảng 10%
     * 0.25 = khoảng 25%
     * 0.50 = khoảng 50%
     * 1.00 = toàn bộ
     *
     * Đây là ngân sách tính toán, không nhất thiết là
     * số phần trăm node cố định.
     */
    float compute_budget = 0.50f;

    // ---------------------------------------------------------------------
    // 3D
    // ---------------------------------------------------------------------

    /**
     * @brief Số candidate tối đa được giữ lại trong quá trình
     *        giảm cấp ở tầng 3D.
     */
    std::size_t max_3d_candidates = 8;

    // ---------------------------------------------------------------------
    // 4D
    // ---------------------------------------------------------------------

    /**
     * @brief Số kết quả tốt nhất 4D gửi cho 5D.
     *
     * Kiến trúc hiện tại mặc định sử dụng Top-3.
     */
    std::size_t top_k_4d = 3;

    // ---------------------------------------------------------------------
    // 5D
    // ---------------------------------------------------------------------

    /**
     * @brief Cho phép 5D từ chối toàn bộ candidate
     *        nếu không có kết quả đủ tốt.
     */
    bool allow_5d_reject = true;

    // ---------------------------------------------------------------------
    // Numerical
    // ---------------------------------------------------------------------

    /**
     * @brief Ngưỡng số học nhỏ để tránh các phép tính
     *        gây vấn đề khi giá trị quá gần 0.
     */
    float epsilon = 1e-6f;

    // ---------------------------------------------------------------------
    // Runtime
    // ---------------------------------------------------------------------

    /**
     * @brief Bật/tắt logging.
     */
    bool enable_logging = true;

    /**
     * @brief Số worker tối đa được phép chạy song song.
     *
     * 0 = để runtime tự quyết định.
     */
    std::size_t max_threads = 0;

    // ---------------------------------------------------------------------
    // Factory
    // ---------------------------------------------------------------------

    /**
     * @brief Tạo cấu hình mặc định.
     */
    static Config defaults();
};

} // namespace ai5d
