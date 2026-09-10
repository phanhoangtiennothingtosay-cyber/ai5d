#pragma once

#include <string>
#include <string_view>

namespace ai5d::logging {

/**
 * @brief Mức độ của log message.
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

/**
 * @brief Logger trung tâm của AI5D.
 *
 * Logger chịu trách nhiệm ghi lại hoạt động của hệ thống:
 * layer nào đang chạy, lỗi nào xảy ra, thời gian xử lý,
 * thông tin debug, v.v.
 *
 * WARNING:
 * Đây là phiên bản BETA / EXPERIMENTAL.
 * API có thể thay đổi.
 */
class Logger {
public:
    /**
     * @brief Lấy Logger dùng chung của toàn hệ thống.
     */
    static Logger& instance();

    /**
     * @brief Ghi một log message.
     *
     * @param level Mức độ log.
     * @param message Nội dung cần ghi.
     */
    void write(
        LogLevel level,
        std::string_view message
    );

    /**
     * @brief Bật hoặc tắt logging.
     */
    void set_enabled(bool enabled);

    /**
     * @brief Kiểm tra logging có đang bật hay không.
     */
    bool enabled() const;

private:
    Logger() = default;

    // Không cho copy Logger.
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    bool enabled_ = true;
};

/**
 * @brief Ghi log mức DEBUG.
 */
void debug(std::string_view message);

/**
 * @brief Ghi log mức INFO.
 */
void info(std::string_view message);

/**
 * @brief Ghi log mức WARNING.
 */
void warning(std::string_view message);

/**
 * @brief Ghi log mức ERROR.
 */
void error(std::string_view message);

} // namespace ai5d::logging

namespace ai5d {

/**
 * @brief Shortcut ghi log thông tin.
 */
inline void log(std::string_view message) {
    logging::info(message);
}

} // namespace ai5d
