#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "SplineTrajectory.hpp"
#include "bind_common.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

// ---------------------------------------------------------------------------
// bind_boundary_conditions<DIM>
// Registers BoundaryConditionsND as "BoundaryConditionsXD" in module m.
// ---------------------------------------------------------------------------
template <int DIM>
void bind_boundary_conditions(py::module_& m, const std::string& name)
{
    using BC = BoundaryConditions<DIM>;
    using Vec = Eigen::Matrix<double, DIM, 1>;

    py::class_<BC, std::shared_ptr<BC>>(m, name.c_str())
        .def(py::init<>(), "All boundary derivatives default to zero.")
        .def(py::init([](py::array_t<double> sv, py::array_t<double> ev) {
            return BC(np_to_vec<DIM>(sv), np_to_vec<DIM>(ev));
        }), py::arg("start_velocity"), py::arg("end_velocity"),
            "Construct with start/end velocity only.")
        .def(py::init([](py::array_t<double> sv, py::array_t<double> sa,
                         py::array_t<double> ev, py::array_t<double> ea) {
            return BC(np_to_vec<DIM>(sv), np_to_vec<DIM>(sa),
                      np_to_vec<DIM>(ev), np_to_vec<DIM>(ea));
        }), py::arg("start_velocity"), py::arg("start_acceleration"),
            py::arg("end_velocity"), py::arg("end_acceleration"),
            "Construct with start/end velocity and acceleration.")
        .def(py::init([](py::array_t<double> sv, py::array_t<double> sa, py::array_t<double> sj,
                         py::array_t<double> ev, py::array_t<double> ea, py::array_t<double> ej) {
            return BC(np_to_vec<DIM>(sv), np_to_vec<DIM>(sa), np_to_vec<DIM>(sj),
                      np_to_vec<DIM>(ev), np_to_vec<DIM>(ea), np_to_vec<DIM>(ej));
        }), py::arg("start_velocity"), py::arg("start_acceleration"), py::arg("start_jerk"),
            py::arg("end_velocity"), py::arg("end_acceleration"), py::arg("end_jerk"),
            "Construct with start/end velocity, acceleration, and jerk.")
        // Properties that copy Eigen vectors to numpy arrays
        .def_property("start_velocity",
            [](const BC& bc) { return vec_to_np<DIM>(bc.start_velocity); },
            [](BC& bc, py::array_t<double> v) { bc.start_velocity = np_to_vec<DIM>(v); })
        .def_property("start_acceleration",
            [](const BC& bc) { return vec_to_np<DIM>(bc.start_acceleration); },
            [](BC& bc, py::array_t<double> v) { bc.start_acceleration = np_to_vec<DIM>(v); })
        .def_property("start_jerk",
            [](const BC& bc) { return vec_to_np<DIM>(bc.start_jerk); },
            [](BC& bc, py::array_t<double> v) { bc.start_jerk = np_to_vec<DIM>(v); })
        .def_property("end_velocity",
            [](const BC& bc) { return vec_to_np<DIM>(bc.end_velocity); },
            [](BC& bc, py::array_t<double> v) { bc.end_velocity = np_to_vec<DIM>(v); })
        .def_property("end_acceleration",
            [](const BC& bc) { return vec_to_np<DIM>(bc.end_acceleration); },
            [](BC& bc, py::array_t<double> v) { bc.end_acceleration = np_to_vec<DIM>(v); })
        .def_property("end_jerk",
            [](const BC& bc) { return vec_to_np<DIM>(bc.end_jerk); },
            [](BC& bc, py::array_t<double> v) { bc.end_jerk = np_to_vec<DIM>(v); })
        .def("__repr__", [name](const BC&) {
            return "<" + name + ">";
        });
}

// ---------------------------------------------------------------------------
// bind_spline<DIM, SplineClass>
// Registers SplineClassND as class_name in module m.
// ---------------------------------------------------------------------------
template <int DIM, template <int> class SplineClass>
void bind_spline(py::module_& m, const std::string& class_name)
{
    using Spline = SplineClass<DIM>;
    using BC = BoundaryConditions<DIM>;
    using MatrixType = typename Spline::MatrixType;  // RowMajor dynamic

    py::class_<Spline>(m, class_name.c_str())
        .def(py::init([](const std::vector<double>& time_points,
                         py::array_t<double, py::array::c_style | py::array::forcecast> wp,
                         const BC& bc) {
            auto info = wp.request();
            if (info.ndim != 2)
                throw std::invalid_argument("waypoints must be a 2-D array");
            if (info.shape[1] != DIM)
                throw std::invalid_argument(
                    "waypoints second dimension must be " + std::to_string(DIM) +
                    ", got " + std::to_string(info.shape[1]));
            if (static_cast<int>(time_points.size()) != info.shape[0])
                throw std::invalid_argument(
                    "len(time_points) must equal number of waypoint rows");
            MatrixType mat(info.shape[0], DIM);
            std::memcpy(mat.data(), info.ptr, info.shape[0] * DIM * sizeof(double));
            return Spline(time_points, mat, bc);
        }),
        py::arg("time_points"),
        py::arg("waypoints"),
        py::arg("bc") = BC{},
        "Construct from time_points (N,), waypoints (N, DIM), optional boundary conditions.")

        .def("evaluate",
            [](const Spline& s, double t, Deriv d) {
                if (!s.isInitialized())
                    throw std::runtime_error("Spline is not initialized");
                return vec_to_np<DIM>(s.getTrajectory().evaluate(t, d));
            },
            py::arg("t"), py::arg("deriv") = Deriv::Pos,
            "Evaluate the spline at time t. Returns ndarray of shape (DIM,).")

        .def("evaluate_batch",
            [](const Spline& s, const std::vector<double>& ts, Deriv d) {
                if (!s.isInitialized())
                    throw std::runtime_error("Spline is not initialized");
                auto results = s.getTrajectory().evaluate(ts, d);
                py::array_t<double> out({static_cast<py::ssize_t>(ts.size()),
                                         static_cast<py::ssize_t>(DIM)});
                auto ptr = out.mutable_data();
                for (size_t i = 0; i < results.size(); ++i)
                    std::memcpy(ptr + i * DIM, results[i].data(), DIM * sizeof(double));
                return out;
            },
            py::arg("times"), py::arg("deriv") = Deriv::Pos,
            "Evaluate the spline at multiple times. Returns ndarray of shape (N, DIM).")

        .def_property_readonly("start_time", &Spline::getStartTime)
        .def_property_readonly("end_time", &Spline::getEndTime)
        .def_property_readonly("duration", &Spline::getDuration)
        .def_property_readonly("num_segments", &Spline::getNumSegments)
        .def_property_readonly("energy", &Spline::getEnergy)
        .def_property_readonly("is_initialized", &Spline::isInitialized)

        .def("__repr__", [class_name](const Spline& s) {
            if (!s.isInitialized())
                return "<" + class_name + " (uninitialized)>";
            return "<" + class_name + " segments=" + std::to_string(s.getNumSegments()) +
                   " t=[" + std::to_string(s.getStartTime()) + "," +
                   std::to_string(s.getEndTime()) + "]>";
        });
}
