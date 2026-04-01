#include "bind_spline.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

void bind_quintic_splines(py::module_& m)
{
    bind_spline<1, QuinticSplineND>(m, "QuinticSpline1D");
    bind_spline<2, QuinticSplineND>(m, "QuinticSpline2D");
    bind_spline<3, QuinticSplineND>(m, "QuinticSpline3D");
    bind_spline<4, QuinticSplineND>(m, "QuinticSpline4D");
    bind_spline<5, QuinticSplineND>(m, "QuinticSpline5D");
    bind_spline<6, QuinticSplineND>(m, "QuinticSpline6D");
}
