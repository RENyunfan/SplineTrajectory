#include "bind_optimizer.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

void bind_quintic_optimizers(py::module_& m)
{
    bind_optimizer<1, QuinticSplineND>(m, "QuinticOptimizer1D");
    bind_optimizer<2, QuinticSplineND>(m, "QuinticOptimizer2D");
    bind_optimizer<3, QuinticSplineND>(m, "QuinticOptimizer3D");
    bind_optimizer<4, QuinticSplineND>(m, "QuinticOptimizer4D");
    bind_optimizer<5, QuinticSplineND>(m, "QuinticOptimizer5D");
    bind_optimizer<6, QuinticSplineND>(m, "QuinticOptimizer6D");
}
