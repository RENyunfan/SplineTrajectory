#include "bind_spline.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

void bind_septic_splines(py::module_& m)
{
    bind_spline<1, SepticSplineND>(m, "SepticSpline1D");
    bind_spline<2, SepticSplineND>(m, "SepticSpline2D");
    bind_spline<3, SepticSplineND>(m, "SepticSpline3D");
    bind_spline<4, SepticSplineND>(m, "SepticSpline4D");
    bind_spline<5, SepticSplineND>(m, "SepticSpline5D");
    bind_spline<6, SepticSplineND>(m, "SepticSpline6D");
}
