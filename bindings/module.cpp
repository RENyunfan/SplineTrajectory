#include <pybind11/pybind11.h>
#include "SplineTrajectory.hpp"
#include "bind_common.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

// Forward declarations from other translation units
void bind_cubic_splines(py::module_& m);
void bind_quintic_splines(py::module_& m);
void bind_septic_splines(py::module_& m);
void bind_cubic_optimizers(py::module_& m);
void bind_quintic_optimizers(py::module_& m);
void bind_septic_optimizers(py::module_& m);
void bind_optimizer_common(py::module_& m);

PYBIND11_MODULE(_spline_trajectory, m)
{
    m.doc() = R"(
spline_trajectory C++ extension module.

Provides efficient spline trajectory optimization based on MINCO-family splines:
  - CubicSplineND   (minimizes acceleration, S2-MINCO)
  - QuinticSplineND (minimizes jerk, S3-MINCO)
  - SepticSplineND  (minimizes snap, S4-MINCO)

Each spline type is available for dimensions 1 through 6 (e.g. CubicSpline3D).
Corresponding optimizers (CubicOptimizer3D, etc.) expose gradient-based optimization
against user-supplied Python cost callables, usable with scipy.optimize.minimize.
)";

    // Deriv enum
    py::enum_<Deriv>(m, "Deriv",
        "Derivative order for spline evaluation.")
        .value("Pos",    Deriv::Pos,    "Position (0th derivative)")
        .value("Vel",    Deriv::Vel,    "Velocity (1st derivative)")
        .value("Acc",    Deriv::Acc,    "Acceleration (2nd derivative)")
        .value("Jerk",   Deriv::Jerk,   "Jerk (3rd derivative)")
        .value("Snap",   Deriv::Snap,   "Snap (4th derivative)")
        .value("Crackle",Deriv::Crackle,"Crackle (5th derivative)")
        .value("Pop",    Deriv::Pop,    "Pop (6th derivative)")
        .export_values();

    // BoundaryConditions + CubicSplines (1-6D)
    bind_cubic_splines(m);

    // QuinticSplines (1-6D) — BoundaryConditions already registered
    bind_quintic_splines(m);

    // SepticSplines (1-6D)
    bind_septic_splines(m);

    // Common optimizer types (Mask, etc.)
    bind_optimizer_common(m);

    // Optimizers for all spline types × dims 1-6
    bind_cubic_optimizers(m);
    bind_quintic_optimizers(m);
    bind_septic_optimizers(m);
}
