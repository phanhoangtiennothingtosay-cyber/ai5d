#include "ai5d/layers/layer3d.hpp"

#include <algorithm>
#include <limits>
#include <utility>
namespace ai5d::layers {

Tensor Layer3D::forward(const Tensor& input) const
{
    if (input.empty()) {
        return Tensor{};
    }

    const std::size_t groups =
        group_count_ == 0 ? 1 : group_count_;

    /*
     * Layer3D chia dữ liệu thành nhiều group.
     *
     * Mỗi group tạo một candidate riêng.
     * Ở tầng 3D, các group được xem như những "phòng ban"
     * độc lập cùng giải quyết một phần dữ liệu.
     */
    const std::size_t chunk =
        (input.size() + groups - 1) / groups;

    std::vector<Tensor> candidates;
    candidates.reserve(groups);

    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t begin = group * chunk;

        if (begin >= input.size()) {
            break;
        }

        const std::size_t end =
            std::min(begin + chunk, input.size());

        Tensor candidate(
            std::vector<std::size_t>{
                end - begin
            }
        );

        for (std::size_t i = begin; i < end; ++i) {
            candidate[i - begin] = input[i];
        }

        candidates.push_back(std::move(candidate));
    }

    return aggregate(candidates);
}

Tensor Layer3D::aggregate(
    const std::vector<Tensor>& candidates
) const
{
    if (candidates.empty()) {
        return Tensor{};
    }

    /*
     * Mỗi candidate bỏ phiếu bằng điểm trung bình.
     *
     * Đây là phiên bản nền. Sau này phần này là nơi
     * phù hợp để đưa P2P attention / candidate interaction
     * vào.
     */
    std::size_t best_index = 0;
    float best_score =
        -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].empty()) {
            continue;
        }

        float sum = 0.0f;

        for (std::size_t j = 0; j < candidates[i].size(); ++j) {
            sum += candidates[i][j];
        }

        const float score =
            sum / static_cast<float>(candidates[i].size());

        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }

    return candidates[best_index];
}

std::size_t Layer3D::group_count() const
{
    return group_count_;
}

void Layer3D::set_group_count(std::size_t count)
{
    group_count_ = count;
}

} // namespace ai5d::layers
