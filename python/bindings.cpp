#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ai5d/brain.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ai5d, m)
{
    m.doc() = "AI5D Python bindings";

    py::class_<ai5d::Brain>(m, "Brain")
        .def(py::init<>())

        .def(
            "in_1d",
            &ai5d::Brain::in_1d,
            py::return_value_policy::reference_internal
        )

        .def(
            "to_2d",
            &ai5d::Brain::to_2d,
            py::return_value_policy::reference_internal
        )

        .def(
            "talk_3d",
            &ai5d::Brain::talk_3d,
            py::return_value_policy::reference_internal
        )

        .def(
            "filter_4d",
            &ai5d::Brain::filter_4d,
            py::return_value_policy::reference_internal
        )

        .def(
            "get_5d",
            &ai5d::Brain::get_5d,
            py::return_value_policy::reference_internal
        )

        .def(
            "result",
            &ai5d::Brain::result,
            py::return_value_policy::reference_internal
        )

        .def(
            "reset",
            &ai5d::Brain::reset
        );
}