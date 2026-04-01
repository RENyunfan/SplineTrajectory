"""
spline_trajectory — Python bindings for SplineTrajectory C++ library.

Provides efficient, gradient-capable spline trajectory optimization based on
MINCO-family splines:

  - CubicSplineND   (minimizes integrated acceleration, S2-MINCO)
  - QuinticSplineND (minimizes integrated jerk, S3-MINCO)
  - SepticSplineND  (minimizes integrated snap, S4-MINCO)

Each type is available for spatial dimensions 1–6:
  CubicSpline1D, CubicSpline2D, ..., CubicSpline6D
  QuinticSpline1D, ..., QuinticSpline6D
  SepticSpline1D,  ..., SepticSpline6D

Corresponding optimizers allow gradient-based optimization of segment durations
and waypoint positions against custom Python cost functions:
  CubicOptimizer3D, QuinticOptimizer3D, SepticOptimizer3D, etc.

Quick start
-----------
>>> import numpy as np
>>> from spline_trajectory import CubicSpline3D, BoundaryConditions3D, Deriv
>>>
>>> waypoints = np.array([[0,0,0],[1,0,0],[2,1,0]], dtype=float)
>>> bc = BoundaryConditions3D(start_velocity=[0.5,0,0], end_velocity=[0,-0.5,0])
>>> spline = CubicSpline3D([0.0, 1.0, 2.0], waypoints, bc)
>>> spline.evaluate(0.5, Deriv.Pos)
>>> spline.evaluate_batch(np.linspace(0, 2, 50), Deriv.Vel)
"""

import numpy as np

from ._spline_trajectory import (
    # Derivative enum
    Deriv,

    # Boundary conditions (DIM 1-6)
    BoundaryConditions1D,
    BoundaryConditions2D,
    BoundaryConditions3D,
    BoundaryConditions4D,
    BoundaryConditions5D,
    BoundaryConditions6D,

    # Cubic splines (minimise acceleration)
    CubicSpline1D,
    CubicSpline2D,
    CubicSpline3D,
    CubicSpline4D,
    CubicSpline5D,
    CubicSpline6D,

    # Quintic splines (minimise jerk)
    QuinticSpline1D,
    QuinticSpline2D,
    QuinticSpline3D,
    QuinticSpline4D,
    QuinticSpline5D,
    QuinticSpline6D,

    # Septic splines (minimise snap)
    SepticSpline1D,
    SepticSpline2D,
    SepticSpline3D,
    SepticSpline4D,
    SepticSpline5D,
    SepticSpline6D,

    # Optimizers — Cubic
    CubicOptimizer1D,
    CubicOptimizer2D,
    CubicOptimizer3D,
    CubicOptimizer4D,
    CubicOptimizer5D,
    CubicOptimizer6D,

    # Optimizers — Quintic
    QuinticOptimizer1D,
    QuinticOptimizer2D,
    QuinticOptimizer3D,
    QuinticOptimizer4D,
    QuinticOptimizer5D,
    QuinticOptimizer6D,

    # Optimizers — Septic
    SepticOptimizer1D,
    SepticOptimizer2D,
    SepticOptimizer3D,
    SepticOptimizer4D,
    SepticOptimizer5D,
    SepticOptimizer6D,

    # Optimizer helpers
    OptimizationMask,
    BoundaryDerivativeMask,
)

# Convenient 3D aliases
CubicSpline = CubicSpline3D
QuinticSpline = QuinticSpline3D
SepticSpline = SepticSpline3D
BoundaryConditions = BoundaryConditions3D
CubicOptimizer = CubicOptimizer3D
QuinticOptimizer = QuinticOptimizer3D
SepticOptimizer = SepticOptimizer3D


def _zero_time_cost(times):
    return 0.0, np.zeros_like(times)


def _make_zero_integral_cost(dim):
    z = np.zeros(dim)

    def _cost(t, t_global, seg, step, p, v, a, j, s):
        return 0.0, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), 0.0

    return _cost


def optimize(optimizer, ctx, time_cost=None, integral_cost=None, *,
             max_iter=200, ftol=1e-9, gtol=1e-6):
    """Run L-BFGS-B optimization via scipy.optimize.minimize.

    Parameters
    ----------
    optimizer : CubicOptimizerND / QuinticOptimizerND / SepticOptimizerND
        An already-configured optimizer instance.
    ctx : OptimizationContext
        Context returned by ``optimizer.prepare_context(problem)``.
    time_cost : callable or None
        Signature: ``(times: ndarray) -> (float, ndarray)``
        where *times* has shape ``(N,)`` (decoded segment durations) and the
        returned tuple is ``(scalar cost, gradient of shape (N,))``.
        Pass ``None`` to use zero cost.
    integral_cost : callable or None
        Signature::

            (t, t_global, seg, step, p, v, a, j, s)
            -> (cost, gp, gv, ga, gj, gs, gt)

        where ``p, v, a, j, s`` each have shape ``(DIM,)`` and the returned
        7-tuple contains the scalar cost and gradients w.r.t. each quantity.
        Pass ``None`` to use zero cost.
    max_iter : int
        Maximum number of L-BFGS-B iterations.
    ftol : float
        Function value tolerance (scipy ``ftol``).
    gtol : float
        Gradient norm tolerance (scipy ``gtol``).

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Use ``optimizer.get_working_spline(ctx)`` to retrieve the optimized spline
        after the call.

    Notes
    -----
    Make sure to call ``optimizer.set_config(rho_energy=...)`` before
    ``optimizer.prepare_context(...)`` to enable energy regularization.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "scipy is required for optimize(). "
            "Install it with: pip install spline-trajectory[optimize]"
        ) from e

    if time_cost is None:
        time_cost = _zero_time_cost
    if integral_cost is None:
        # Infer DIM from context
        dim = ctx.num_segments  # not DIM — need another way
        # Use a generic zero integral cost (will be called with DIM-sized arrays)
        def integral_cost(t, t_global, seg, step, p, v, a, j, s):
            z = np.zeros_like(p)
            return 0.0, z, z, z, z, z, 0.0

    def _objective(x):
        cost, grad = optimizer.evaluate(ctx, x, time_cost, integral_cost)
        return float(cost), np.asarray(grad, dtype=np.float64)

    x0 = optimizer.generate_initial_guess(ctx)
    return minimize(
        _objective, x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )


__all__ = [
    # Enum
    "Deriv",
    # BoundaryConditions
    "BoundaryConditions",
    "BoundaryConditions1D", "BoundaryConditions2D", "BoundaryConditions3D",
    "BoundaryConditions4D", "BoundaryConditions5D", "BoundaryConditions6D",
    # Cubic splines
    "CubicSpline",
    "CubicSpline1D", "CubicSpline2D", "CubicSpline3D",
    "CubicSpline4D", "CubicSpline5D", "CubicSpline6D",
    # Quintic splines
    "QuinticSpline",
    "QuinticSpline1D", "QuinticSpline2D", "QuinticSpline3D",
    "QuinticSpline4D", "QuinticSpline5D", "QuinticSpline6D",
    # Septic splines
    "SepticSpline",
    "SepticSpline1D", "SepticSpline2D", "SepticSpline3D",
    "SepticSpline4D", "SepticSpline5D", "SepticSpline6D",
    # Optimizers
    "CubicOptimizer", "CubicOptimizer1D", "CubicOptimizer2D", "CubicOptimizer3D",
    "CubicOptimizer4D", "CubicOptimizer5D", "CubicOptimizer6D",
    "QuinticOptimizer", "QuinticOptimizer1D", "QuinticOptimizer2D", "QuinticOptimizer3D",
    "QuinticOptimizer4D", "QuinticOptimizer5D", "QuinticOptimizer6D",
    "SepticOptimizer", "SepticOptimizer1D", "SepticOptimizer2D", "SepticOptimizer3D",
    "SepticOptimizer4D", "SepticOptimizer5D", "SepticOptimizer6D",
    # Helpers
    "OptimizationMask",
    "BoundaryDerivativeMask",
    # Convenience function
    "optimize",
]
