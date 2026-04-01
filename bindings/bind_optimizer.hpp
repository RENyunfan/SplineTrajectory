#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "SplineOptimizer.hpp"
#include "bind_common.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

// ---------------------------------------------------------------------------
// bind_optimizer_common
// Registers OptimizationMask and BoundaryDerivativeMask once.
// Defined in bind_optimizer_cubic.cpp; declared here for include convenience.
// ---------------------------------------------------------------------------
void bind_optimizer_common(py::module_& m);

inline void bind_optimizer_common_impl(py::module_& m)
{
    py::class_<BoundaryDerivativeMask>(m, "BoundaryDerivativeMask")
        .def(py::init<>())
        .def_readwrite("v", &BoundaryDerivativeMask::v,
                       "Optimize start/end velocity boundary condition")
        .def_readwrite("a", &BoundaryDerivativeMask::a,
                       "Optimize start/end acceleration boundary condition")
        .def_readwrite("j", &BoundaryDerivativeMask::j,
                       "Optimize start/end jerk boundary condition");

    py::class_<OptimizationMask>(m, "OptimizationMask")
        .def(py::init<>())
        .def_readwrite("time", &OptimizationMask::time,
                       "list[int]: 1 = optimize this segment duration, 0 = fix it")
        .def_readwrite("waypoints", &OptimizationMask::waypoints,
                       "list[int]: 1 = optimize this waypoint, 0 = fix it")
        .def_readwrite("start", &OptimizationMask::start,
                       "BoundaryDerivativeMask: which start BC derivatives to optimize")
        .def_readwrite("end", &OptimizationMask::end,
                       "BoundaryDerivativeMask: which end BC derivatives to optimize");
}

// ---------------------------------------------------------------------------
// bind_optimizer<DIM, SplineType>
// Registers SplineOptimizerND (with default maps) as class_name in module m.
// ---------------------------------------------------------------------------
template <int DIM, template <int> class SplineClass>
void bind_optimizer(py::module_& m, const std::string& class_name)
{
    using SplineType = SplineClass<DIM>;
    using Opt = SplineOptimizer<DIM, SplineType>;
    using BC = BoundaryConditions<DIM>;
    using WP = typename Opt::WaypointsType;
    using Problem = typename Opt::ProblemDefinition;
    using Context = typename Opt::OptimizationContext;
    using TCost = PyTimeCost;
    using ICost = PyIntegralCost<DIM>;

    // ---- ProblemDefinition ------------------------------------------------
    // Registered at module level; also accessible as optimizer.Problem
    std::string prob_name = class_name + "_Problem";
    auto prob_cls = py::class_<Problem, std::shared_ptr<Problem>>(m, prob_name.c_str())
        .def(py::init<>())
        .def_readwrite("time_segments", &Problem::time_segments,
                       "list[float]: initial duration for each segment")
        .def_property("waypoints",
            [](const Problem& p) {
                int rows = static_cast<int>(p.waypoints.rows());
                py::array_t<double> out({rows, DIM});
                std::memcpy(out.mutable_data(), p.waypoints.data(),
                            rows * DIM * sizeof(double));
                return out;
            },
            [](Problem& p, py::array_t<double, py::array::c_style | py::array::forcecast> wp) {
                auto info = wp.request();
                if (info.ndim != 2 || info.shape[1] != DIM)
                    throw std::invalid_argument(
                        "waypoints must be shape (N, " + std::to_string(DIM) + ")");
                p.waypoints.resize(info.shape[0], DIM);
                std::memcpy(p.waypoints.data(), info.ptr,
                            info.shape[0] * DIM * sizeof(double));
            })
        .def_readwrite("start_time", &Problem::start_time)
        .def_readwrite("bc", &Problem::bc)
        .def_property("mask",
            [](const Problem& p) -> py::object {
                if (!p.mask.has_value()) return py::none();
                return py::cast(p.mask.value());
            },
            [](Problem& p, py::object obj) {
                if (obj.is_none()) p.mask = std::nullopt;
                else p.mask = obj.cast<OptimizationMask>();
            });

    // ---- OptimizationContext ----------------------------------------------
    std::string ctx_name = class_name + "_Context";
    auto ctx_cls = py::class_<Context, std::shared_ptr<Context>>(m, ctx_name.c_str())
        .def_property_readonly("is_valid", &Context::isValid)
        .def_property_readonly("num_segments",
            [](const Context& c) { return c.prepared.num_segments; });

    // ---- SplineOptimizer --------------------------------------------------
    auto opt_cls = py::class_<Opt>(m, class_name.c_str())
        .def(py::init<>())

        .def("set_config",
            [](Opt& self, double rho_energy, int integral_num_steps) {
                typename Opt::OptimizerConfig cfg;
                cfg.rho_energy = rho_energy;
                cfg.integral_num_steps = integral_num_steps;
                auto status = self.setConfig(cfg);
                if (!status.ok)
                    throw std::runtime_error(status.message);
            },
            py::arg("rho_energy") = 0.0,
            py::arg("integral_num_steps") = 64,
            "Configure the optimizer.\n\n"
            "  rho_energy: weight for the integrated energy (jerk/snap) regularization term.\n"
            "  integral_num_steps: number of trapezoidal integration steps per segment.")

        // prepare_context(problem: ProblemDefinition) overload
        .def("prepare_context",
            [](const Opt& self, const Problem& prob) {
                auto ctx = std::make_shared<Context>();
                auto status = self.prepareContext(prob, *ctx);
                if (!status.ok)
                    throw std::runtime_error(status.message);
                return ctx;
            },
            py::arg("problem"),
            "Prepare an optimization context from a ProblemDefinition.\n"
            "Returns an OptimizationContext. Raises RuntimeError on invalid input.")

        // prepare_context(time_segments, waypoints, bc, mask) keyword overload
        .def("prepare_context",
            [](const Opt& self,
               std::vector<double> time_segments,
               py::array_t<double, py::array::c_style | py::array::forcecast> wp,
               const BC& bc,
               py::object mask_obj) {
                auto info = wp.request();
                if (info.ndim != 2 || info.shape[1] != DIM)
                    throw std::invalid_argument(
                        "waypoints must be shape (N, " + std::to_string(DIM) + ")");
                Problem prob;
                prob.time_segments = std::move(time_segments);
                prob.waypoints.resize(info.shape[0], DIM);
                std::memcpy(prob.waypoints.data(), info.ptr,
                            info.shape[0] * DIM * sizeof(double));
                prob.bc = bc;
                if (!mask_obj.is_none())
                    prob.mask = mask_obj.cast<OptimizationMask>();
                auto ctx = std::make_shared<Context>();
                auto status = self.prepareContext(prob, *ctx);
                if (!status.ok)
                    throw std::runtime_error(status.message);
                return ctx;
            },
            py::arg("time_segments"),
            py::arg("waypoints"),
            py::arg("bc") = BC{},
            py::arg("mask") = py::none(),
            "Prepare a context from keyword arguments (convenience overload).\n"
            "Returns an OptimizationContext.")

        .def("generate_initial_guess",
            [](const Opt& self, const std::shared_ptr<Context>& ctx) {
                return vxd_to_np(self.generateInitialGuess(*ctx));
            },
            py::arg("ctx"),
            "Generate a default initial decision-variable vector for the optimizer.\n"
            "Returns ndarray of shape (D,).")

        .def("evaluate",
            [](Opt& self, std::shared_ptr<Context> ctx,
               py::array_t<double> x_np,
               py::object time_cost_py,
               py::object integral_cost_py) -> py::tuple {
                Eigen::VectorXd x = np_to_vxd(x_np);
                Eigen::VectorXd grad(x.size());
                grad.setZero();

                TCost tc{time_cost_py};
                ICost ic{integral_cost_py};
                auto spec = Opt::makeEvaluateSpec(tc, ic);
                auto result = self.evaluate(*ctx, x, grad, spec);
                if (!result.ok)
                    throw std::runtime_error(result.message);
                return py::make_tuple(result.cost, vxd_to_np(grad));
            },
            py::arg("ctx"),
            py::arg("x"),
            py::arg("time_cost"),
            py::arg("integral_cost"),
            "Evaluate cost and gradient at decision variable x.\n\n"
            "  time_cost: callable(times: ndarray) -> (float, ndarray)\n"
            "  integral_cost: callable(t, t_global, seg, step, p,v,a,j,s)\n"
            "                 -> (cost, gp, gv, ga, gj, gs, gt)\n"
            "Returns (cost: float, grad: ndarray).")

        .def("get_working_spline",
            [](const Opt& self, const std::shared_ptr<Context>& ctx) {
                // Return a copy of the working spline
                return SplineType(self.getWorkingSpline(*ctx));
            },
            py::arg("ctx"),
            "Return the current spline from the optimization context (copy).")

        .def("__repr__", [class_name](const Opt&) {
            return "<" + class_name + ">";
        });

    // Make Problem and Context accessible as class attributes
    // e.g. QuinticOptimizer3D.Problem(), QuinticOptimizer3D.Context
    opt_cls.attr("Problem") = prob_cls;
    opt_cls.attr("Context") = ctx_cls;
}
