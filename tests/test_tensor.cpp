#include "ai5d/tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

int main()
{
    // ------------------------------------------------------------
    // 1. Raw data constructor
    // ------------------------------------------------------------

    ai5d::Tensor values{
        1.0f,
        2.0f,
        3.0f
    };

    assert(values.size() == 3);
    assert(values.ndim() == 1);
    assert(values.shape()[0] == 3);

    assert(values[0] == 1.0f);
    assert(values[1] == 2.0f);
    assert(values[2] == 3.0f);

    // ------------------------------------------------------------
    // 2. from_shape()
    // ------------------------------------------------------------

    ai5d::Tensor matrix =
        ai5d::Tensor::from_shape({2, 3});

    assert(matrix.size() == 6);
    assert(matrix.ndim() == 2);
    assert(matrix.shape()[0] == 2);
    assert(matrix.shape()[1] == 3);

    // ------------------------------------------------------------
    // 3. Contiguous storage
    // ------------------------------------------------------------

    float* ptr = matrix.data();

    assert(ptr != nullptr);
    assert(&matrix[0] == ptr);
    assert(&matrix[1] == ptr + 1);

    // ------------------------------------------------------------
    // 4. Row-major flat storage
    // ------------------------------------------------------------

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<float>(i);
    }

    assert(matrix[0] == 0.0f);
    assert(matrix[1] == 1.0f);
    assert(matrix[2] == 2.0f);
    assert(matrix[3] == 3.0f);
    assert(matrix[4] == 4.0f);
    assert(matrix[5] == 5.0f);

    // ------------------------------------------------------------
    // 5. Copy must be deep copy
    // ------------------------------------------------------------

    ai5d::Tensor copy = matrix;

    copy[0] = 999.0f;

    assert(matrix[0] == 0.0f);
    assert(copy[0] == 999.0f);

    // ------------------------------------------------------------
    // 6. Move
    // ------------------------------------------------------------

    ai5d::Tensor moved =
        std::move(copy);

    assert(moved.size() == 6);
    assert(moved[0] == 999.0f);

    // ------------------------------------------------------------
    // 7. reshape()
    // ------------------------------------------------------------

    moved.reshape({3, 2});

    assert(moved.size() == 6);
    assert(moved.ndim() == 2);
    assert(moved.shape()[0] == 3);
    assert(moved.shape()[1] == 2);

    assert(moved[0] == 999.0f);
    assert(moved[5] == 5.0f);

    // ------------------------------------------------------------
    // 8. Invalid reshape
    // ------------------------------------------------------------

    bool reshape_failed = false;

    try {
        moved.reshape({4, 4});
    }
    catch (const std::invalid_argument&) {
        reshape_failed = true;
    }

    assert(reshape_failed);

    // ------------------------------------------------------------
    // 9. Out-of-bounds indexing
    // ------------------------------------------------------------

    bool index_failed = false;

    try {
        (void)moved[100];
    }
    catch (const std::out_of_range&) {
        index_failed = true;
    }

    assert(index_failed);

    // ------------------------------------------------------------
    // 10. Zero dimension must fail
    // ------------------------------------------------------------

    bool zero_dimension_failed = false;

    try {
        auto invalid =
            ai5d::Tensor::from_shape({0, 5});

        (void)invalid;
    }
    catch (const std::invalid_argument&) {
        zero_dimension_failed = true;
    }

    assert(zero_dimension_failed);

    // ------------------------------------------------------------
    // 11. Overflow must fail
    // ------------------------------------------------------------

    bool overflow_failed = false;

    try {
        const std::size_t max_size =
            std::numeric_limits<std::size_t>::max();

        auto invalid =
            ai5d::Tensor::from_shape({
                max_size,
                2
            });

        (void)invalid;
    }
    catch (const std::overflow_error&) {
        overflow_failed = true;
    }

    assert(overflow_failed);

    // ------------------------------------------------------------
    // 12. Empty Tensor
    // ------------------------------------------------------------

    ai5d::Tensor empty;

    assert(empty.empty());
    assert(empty.size() == 0);

    // ------------------------------------------------------------
    // 13. NaN / Inf detection
    // ------------------------------------------------------------

    ai5d::Tensor finite{
        1.0f,
        2.0f,
        3.0f
    };

    assert(!finite.has_nan());
    assert(!finite.has_inf());

    ai5d::Tensor nan_tensor{
        1.0f,
        std::numeric_limits<float>::quiet_NaN()
    };

    assert(nan_tensor.has_nan());
    assert(!nan_tensor.has_inf());

    ai5d::Tensor inf_tensor{
        1.0f,
        std::numeric_limits<float>::infinity()
    };

    assert(!inf_tensor.has_nan());
    assert(inf_tensor.has_inf());

    // ------------------------------------------------------------
    // 14. clear()
    // ------------------------------------------------------------

    finite.clear();

    assert(finite.empty());
    assert(finite.size() == 0);
    assert(finite.ndim() == 0);

    return 0;
}
