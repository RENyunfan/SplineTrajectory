#include "bind_optimizer.hpp"

namespace py = pybind11;
using namespace SplineTrajectory;

void bind_septic_optimizers(py::module_& m)
{
    bind_optimizer<1, SepticSplineND>(m, "SepticOptimizer1D");
    bind_optimizer<2, SepticSplineND>(m, "SepticOptimizer2D");
    bind_optimizer<3, SepticSplineND>(m, "SepticOptimizer3D");
    bind_optimizer<4, SepticSplineND>(m, "SepticOptimizer4D");
    bind_optimizer<5, SepticSplineND>(m, "SepticOptimizer5D");
    bind_optimizer<6, SepticSplineND>(m, "SepticOptimizer6D");
}
