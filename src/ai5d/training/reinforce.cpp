#include "ai5d/training/reinforce.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ai5d::training {

Tensor to_probabilities(const Tensor& scores)
{
    if (scores.empty()) {
        return Tensor{};
    }

    if (scores.ndim() != 1) {
        throw std::invalid_argument(
            "AI5D: to_probabilities requires a 1D tensor."
        );
    }

    // Trừ giá trị lớn nhất trước khi exp để ổn định số học
    // (numerically stable softmax) — tránh tràn số khi score lớn.
    float max_score = scores[0];
    for (std::size_t i = 1; i < scores.size(); ++i) {
        max_score = std::max(max_score, scores[i]);
    }

    Tensor probabilities = Tensor::from_shape({scores.size()});

    float sum = 0.0f;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const float value = std::exp(scores[i] - max_score);
        probabilities[i] = value;
        sum += value;
    }

    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        probabilities[i] /= sum;
    }

    return probabilities;
}

std::size_t sample_index(const Tensor& probabilities)
{
    if (probabilities.empty()) {
        throw std::invalid_argument(
            "AI5D: sample_index requires a non-empty probability tensor."
        );
    }

    // Generator dùng chung cho toàn bộ tiến trình (thread_local để
    // an toàn nếu sau này có nhiều luồng train song song).
    static thread_local std::mt19937 generator{std::random_device{}()};
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const float roll = distribution(generator);

    float cumulative = 0.0f;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (roll <= cumulative) {
            return i;
        }
    }

    // Phòng trường hợp sai số dấu phẩy động khiến tổng cumulative
    // hụt một chút dưới 1.0 — trả về phần tử cuối cùng thay vì
    // để hàm không trả về gì.
    return probabilities.size() - 1;
}

Tensor policy_gradient(
    const Tensor& probabilities,
    std::size_t chosen_index,
    float reward
)
{
    if (probabilities.empty()) {
        throw std::invalid_argument(
            "AI5D: policy_gradient requires a non-empty probability tensor."
        );
    }

    if (chosen_index >= probabilities.size()) {
        throw std::invalid_argument(
            "AI5D: chosen_index is out of range for policy_gradient."
        );
    }

    Tensor gradient = Tensor::from_shape({probabilities.size()});

    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const float one_hot = (i == chosen_index) ? 1.0f : 0.0f;
        gradient[i] = reward * (probabilities[i] - one_hot);
    }

    return gradient;
}

float compute_baseline(const std::vector<ActionRecord>& records)
{
    if (records.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (const ActionRecord& record : records) {
        sum += record.reward;
    }

    return sum / static_cast<float>(records.size());
}

} // namespace ai5d::training
