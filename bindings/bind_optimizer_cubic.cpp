#include "bind_optimizer.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

// Provide the single definition of bind_optimizer_common (declared in bind_optimizer.hpp)
void bind_optimizer_common(py::module_& m)
{
    bind_optimizer_common_impl(m);
}

void bind_cubic_optimizers(py::module_& m)
{
    bind_optimizer<1, CubicSplineND>(m, "CubicOptimizer1D");
    bind_optimizer<2, CubicSplineND>(m, "CubicOptimizer2D");
    bind_optimizer<3, CubicSplineND>(m, "CubicOptimizer3D");
    bind_optimizer<4, CubicSplineND>(m, "CubicOptimizer4D");
    bind_optimizer<5, CubicSplineND>(m, "CubicOptimizer5D");
    bind_optimizer<6, CubicSplineND>(m, "CubicOptimizer6D");
}
