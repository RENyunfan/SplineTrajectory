#include "bind_spline.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

void bind_cubic_splines(py::module_& m)
{
    // BoundaryConditions for all dims (shared across spline types; only define once here)
    bind_boundary_conditions<1>(m, "BoundaryConditions1D");
    bind_boundary_conditions<2>(m, "BoundaryConditions2D");
    bind_boundary_conditions<3>(m, "BoundaryConditions3D");
    bind_boundary_conditions<4>(m, "BoundaryConditions4D");
    bind_boundary_conditions<5>(m, "BoundaryConditions5D");
    bind_boundary_conditions<6>(m, "BoundaryConditions6D");

    // CubicSplineND for dims 1-6
    bind_spline<1, CubicSplineND>(m, "CubicSpline1D");
    bind_spline<2, CubicSplineND>(m, "CubicSpline2D");
    bind_spline<3, CubicSplineND>(m, "CubicSpline3D");
    bind_spline<4, CubicSplineND>(m, "CubicSpline4D");
    bind_spline<5, CubicSplineND>(m, "CubicSpline5D");
    bind_spline<6, CubicSplineND>(m, "CubicSpline6D");
}
