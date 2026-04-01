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


def optimize_closed_loop(
    optimizer,
    waypoints,
    time_segments,
    *,
    closure_weight=1000.0,
    rho_energy=1.0,
    integral_num_steps=64,
    optimize_times=True,
    time_cost=None,
    integral_cost=None,
    max_iter=300,
    ftol=1e-9,
    gtol=1e-6,
):
    """Closed-loop trajectory optimization (e.g. drone racing).

    Enforces C² continuity at the loop junction by:
    - Setting waypoints[-1] = waypoints[0] (position closure).
    - Freeing start/end boundary derivatives as decision variables.
    - Adding a soft penalty  ``closure_weight * ||start_BC - end_BC||²``
      so that velocity and acceleration match at the junction.

    The penalty weight ``closure_weight`` should be large enough to drive
    the BC mismatch close to zero (typically 100–10000).  After optimization,
    the working spline passes through all waypoints with smooth, periodic
    velocity and acceleration.

    Parameters
    ----------
    optimizer : CubicOptimizerND / QuinticOptimizerND / SepticOptimizerND
        Optimizer instance (``set_config`` is called internally).
    waypoints : array-like, shape (N, DIM)
        Waypoints the trajectory must pass through.  The last waypoint is
        overwritten with the first to close the loop.
    time_segments : list[float]
        Initial duration for each segment (length N-1 for N waypoints).
    closure_weight : float
        Penalty weight for the BC closure constraint.
    rho_energy : float
        Weight for integrated energy (jerk/snap) regularization.
    integral_num_steps : int
        Trapezoidal integration steps per segment.
    optimize_times : bool
        If True (default) segment durations are also decision variables.
    time_cost : callable or None
        ``(times: ndarray) -> (float, ndarray)``  — custom time cost.
        ``None`` uses a zero cost (only energy regularization).
    integral_cost : callable or None
        Integral cost callable — see ``optimize()`` for signature.
    max_iter, ftol, gtol : optimization termination criteria.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
    ctx    : OptimizationContext  (use with ``optimizer.get_working_spline(ctx)``)

    Example
    -------
    >>> import numpy as np
    >>> from spline_trajectory import QuinticOptimizer3D, optimize_closed_loop, Deriv
    >>> gates = np.array([[0,0,1],[3,1,2],[5,4,1],[2,5,2]], dtype=float)
    >>> time_segs = [1.0, 1.5, 1.2, 1.0]   # N-1 segments for N gates
    >>> opt = QuinticOptimizer3D()
    >>> result, ctx = optimize_closed_loop(opt, gates, time_segs)
    >>> spline = opt.get_working_spline(ctx)
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "scipy is required for optimize_closed_loop(). "
            "Install it with: pip install spline-trajectory[optimize]"
        ) from e

    wp = np.array(waypoints, dtype=float)
    if wp.ndim != 2:
        raise ValueError("waypoints must be 2-D array of shape (N, DIM)")
    n_wp, dim = wp.shape
    n_seg = len(time_segments)
    if n_seg != n_wp - 1:
        raise ValueError(
            f"len(time_segments)={n_seg} must equal len(waypoints)-1={n_wp-1}"
        )

    # Close the position loop
    wp[-1] = wp[0]

    # Build zero BC (actual values will be optimized)
    _bc_classes = {
        1: BoundaryConditions1D, 2: BoundaryConditions2D,
        3: BoundaryConditions3D, 4: BoundaryConditions4D,
        5: BoundaryConditions5D, 6: BoundaryConditions6D,
    }
    if dim not in _bc_classes:
        raise ValueError(f"DIM={dim} is not supported (must be 1–6)")
    bc = _bc_classes[dim]()

    optimizer.set_config(rho_energy=rho_energy, integral_num_steps=integral_num_steps)

    # Probe which BC slots this spline order supports by trying progressively
    # fewer derivative orders (jerk → acc → vel only).
    # ORDER≥7 (Septic)  : v + a + j
    # ORDER≥5 (Quintic) : v + a
    # ORDER=3 (Cubic)   : v only
    def _make_mask(enable_a, enable_j):
        m = OptimizationMask()
        m.waypoints = [0] * n_wp
        m.time = [1 if optimize_times else 0] * n_seg
        m.start.v = True
        m.start.a = enable_a
        m.start.j = enable_j
        m.end.v = True
        m.end.a = enable_a
        m.end.j = enable_j
        return m

    ctx = None
    for try_a, try_j in [(True, True), (True, False), (False, False)]:
        try:
            ctx = optimizer.prepare_context(
                time_segments=list(time_segments),
                waypoints=wp,
                bc=bc,
                mask=_make_mask(try_a, try_j),
            )
            break
        except RuntimeError:
            continue
    if ctx is None:
        raise RuntimeError(
            "optimize_closed_loop: could not prepare a valid context. "
            "Check waypoints, time_segments, and optimizer type."
        )

    # Layout: bc_offset tells us where in x the BC variables live.
    # Order (from forEachOptimizedBoundaryDerivativeSlot):
    #   [start_v (dim), start_a (dim), start_j (dim),   <- start blocks
    #    end_v   (dim), end_a   (dim), end_j   (dim)]   <- end blocks
    # (only slots actually enabled for this spline order appear)
    bc_offset = ctx.bc_offset
    n_bc_total = ctx.n_bc_vars          # total BC vars in x
    n_bc_per_end = n_bc_total // 2      # symmetric: same slots for start and end

    if time_cost is None:
        time_cost = _zero_time_cost

    if integral_cost is None:
        def integral_cost(t, tg, seg, step, p, v, a, j, s):
            z = np.zeros_like(p)
            return 0.0, z, z, z, z, z, 0.0

    def _objective(x):
        traj_cost, grad = optimizer.evaluate(ctx, x, time_cost, integral_cost)
        grad = np.asarray(grad, dtype=np.float64)

        # Closure penalty: ||start_BC - end_BC||²
        if n_bc_per_end > 0:
            s_bc = x[bc_offset : bc_offset + n_bc_per_end]
            e_bc = x[bc_offset + n_bc_per_end : bc_offset + 2 * n_bc_per_end]
            diff = s_bc - e_bc
            penalty = closure_weight * float(np.dot(diff, diff))
            grad[bc_offset : bc_offset + n_bc_per_end] += 2.0 * closure_weight * diff
            grad[bc_offset + n_bc_per_end : bc_offset + 2 * n_bc_per_end] -= (
                2.0 * closure_weight * diff
            )
        else:
            penalty = 0.0

        return float(traj_cost) + penalty, grad

    x0 = optimizer.generate_initial_guess(ctx)
    result = minimize(
        _objective, x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return result, ctx


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
    # Convenience functions
    "optimize",
    "optimize_closed_loop",
]
