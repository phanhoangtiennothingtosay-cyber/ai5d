# ============================================================
# AI5D — CMakeLists.txt
# ============================================================

cmake_minimum_required(VERSION 3.15)

project(ai5d
    VERSION 0.1.0
    LANGUAGES CXX
)

# ------------------------------------------------------------
# 1. C++ standard
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# 2. AI5D C++ core
# ------------------------------------------------------------

add_library(ai5d_core STATIC

    src/ai5d/tensor.cpp
    src/ai5d/brain.cpp

    src/ai5d/layers/layer1d.cpp
    src/ai5d/layers/layer2d.cpp
    src/ai5d/layers/layer3d.cpp
    src/ai5d/layers/layer4d.cpp
    src/ai5d/layers/layer5d.cpp

    src/ai5d/math/activation.cpp
    src/ai5d/math/attention.cpp
    src/ai5d/math/entropy.cpp
    src/ai5d/math/linear.cpp
    src/ai5d/math/normalization.cpp
    src/ai5d/math/topk.cpp

    src/ai5d/routing/compute_router.cpp
    src/ai5d/logging/logger.cpp
)

target_include_directories(ai5d_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------
# 3. Python
# ------------------------------------------------------------

find_package(Python
    COMPONENTS
        Interpreter
        Development
    REQUIRED
)

# ------------------------------------------------------------
# 4. pybind11
# ------------------------------------------------------------

find_package(pybind11 QUIET CONFIG)

if(NOT pybind11_FOUND)

    include(FetchContent)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(pybind11)

endif()

# ------------------------------------------------------------
# 5. Python extension module
# ------------------------------------------------------------

pybind11_add_module(_ai5d
    python/bindings.cpp
)

# AI5D C++ core + Python runtime
target_link_libraries(_ai5d
    PRIVATE
        ai5d_core
        Python::Python
)

target_include_directories(_ai5d
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------
# 6. Python package output
# ------------------------------------------------------------

set_target_properties(_ai5d PROPERTIES

    LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/ai5d

    RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/ai5d

)

# ------------------------------------------------------------
# 7. Multi-config generators
# ------------------------------------------------------------

foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)

    string(TOUPPER ${CONFIG} CONFIG_UPPER)

    set_target_properties(_ai5d PROPERTIES

        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}
            ${CMAKE_CURRENT_SOURCE_DIR}/ai5d

        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}
            ${CMAKE_CURRENT_SOURCE_DIR}/ai5d

    )

endforeach()
