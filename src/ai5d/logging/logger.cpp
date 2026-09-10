#include "ai5d/logging/logger.hpp"

#include <iostream>
#include <string_view>

namespace ai5d::logging {

namespace {

const char* level_name(LogLevel level)
{
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::WARNING:
        return "WARNING";
    case LogLevel::ERROR:
        return "ERROR";
    }

    return "UNKNOWN";
}

} // namespace

Logger& Logger::instance()
{
    static Logger logger;
    return logger;
}

void Logger::write(LogLevel level, std::string_view message)
{
    if (!enabled_) {
        return;
    }

    std::ostream& output =
        (level == LogLevel::ERROR) ? std::cerr : std::cout;

    output << "[AI5D][" << level_name(level) << "] "
           << message << '\n';
}

void Logger::set_enabled(bool enabled)
{
    enabled_ = enabled;
}

bool Logger::enabled() const
{
    return enabled_;
}

void debug(std::string_view message)
{
    Logger::instance().write(LogLevel::DEBUG, message);
}

void info(std::string_view message)
{
    Logger::instance().write(LogLevel::INFO, message);
}

void warning(std::string_view message)
{
    Logger::instance().write(LogLevel::WARNING, message);
}

void error(std::string_view message)
{
    Logger::instance().write(LogLevel::ERROR, message);
}

} // namespace ai5d::logging
