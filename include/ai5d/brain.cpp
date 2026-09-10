#include "ai5d/brain.hpp"

#include <stdexcept>
#include <utility>

namespace ai5d {

Brain::Brain() = default;

Brain::~Brain() = default;

Brain& Brain::in_1d(const std::vector<float>& data)
{
    input_ = data;
    result_.clear();

    has_input_ = true;
    processed_1d_ = false;
    processed_2d_ = false;
    processed_3d_ = false;
    processed_4d_ = false;
    processed_5d_ = false;

    return *this;
}

Brain& Brain::to_2d()
{
    if (!has_input_) {
        throw std::runtime_error("AI5D: input is required before to_2d().");
    }

    processed_1d_ = true;
    processed_2d_ = true;

    result_ = input_;

    return *this;
}

Brain& Brain::talk_3d()
{
    if (!processed_2d_) {
        throw std::runtime_error("AI5D: to_2d() must be called before talk_3d().");
    }

    processed_3d_ = true;

    return *this;
}

Brain& Brain::filter_4d()
{
    if (!processed_3d_) {
        throw std::runtime_error("AI5D: talk_3d() must be called before filter_4d().");
    }

    processed_4d_ = true;

    return *this;
}

Brain& Brain::get_5d()
{
    if (!processed_4d_) {
        throw std::runtime_error("AI5D: filter_4d() must be called before get_5d().");
    }

    processed_5d_ = true;

    return *this;
}

const std::vector<float>& Brain::result() const
{
    return result_;
}

void Brain::reset()
{
    input_.clear();
    result_.clear();

    has_input_ = false;
    processed_1d_ = false;
    processed_2d_ = false;
    processed_3d_ = false;
    processed_4d_ = false;
    processed_5d_ = false;
}

} // namespace ai5d
