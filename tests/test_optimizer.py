"""Tests for the SplineOptimizer Python bindings."""
import numpy as np
import pytest
from spline_trajectory import (
    QuinticOptimizer3D,
    CubicOptimizer3D,
    BoundaryConditions3D,
    OptimizationMask,
    Deriv,
    optimize,
)

WAYPOINTS = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
TIME_SEGS = [1.0, 1.0]
BC = BoundaryConditions3D(start_velocity=[0.5, 0, 0], end_velocity=[0, -0.5, 0])


def make_context(rho_energy=1.0, steps=64):
    opt = QuinticOptimizer3D()
    opt.set_config(rho_energy=rho_energy, integral_num_steps=steps)
    ctx = opt.prepare_context(
        time_segments=TIME_SEGS,
        waypoints=WAYPOINTS,
        bc=BC,
    )
    return opt, ctx


class TestProblemDefinition:
    def test_problem_class_accessible(self):
        # Problem is available as optimizer.Problem
        prob = QuinticOptimizer3D.Problem()
        prob.time_segments = [1.0, 1.2]
        prob.waypoints = WAYPOINTS
        assert prob.time_segments == pytest.approx([1.0, 1.2])
        assert prob.waypoints.shape == (3, 3)

    def test_prepare_context_from_problem(self):
        opt = QuinticOptimizer3D()
        opt.set_config(rho_energy=0.5)
        prob = QuinticOptimizer3D.Problem()
        prob.time_segments = TIME_SEGS
        prob.waypoints = WAYPOINTS
        ctx = opt.prepare_context(prob)
        assert ctx.is_valid
        assert ctx.num_segments == 2

    def test_prepare_context_kwargs(self):
        opt = QuinticOptimizer3D()
        opt.set_config(rho_energy=0.5)
        ctx = opt.prepare_context(
            time_segments=TIME_SEGS,
            waypoints=WAYPOINTS,
            bc=BC,
        )
        assert ctx.is_valid
        assert ctx.num_segments == 2


class TestInitialGuess:
    def test_initial_guess_shape(self):
        opt, ctx = make_context()
        x0 = opt.generate_initial_guess(ctx)
        assert x0.ndim == 1
        assert x0.size > 0

    def test_initial_guess_finite(self):
        opt, ctx = make_context()
        x0 = opt.generate_initial_guess(ctx)
        assert np.all(np.isfinite(x0))


class TestEvaluate:
    def _make_cost_fns(self):
        def time_cost(times):
            return float(np.sum(times)), np.ones_like(times)

        def integral_cost(t, t_global, seg, step, p, v, a, j, s):
            z = np.zeros_like(p)
            return 0.0, z, z, z, z, z, 0.0

        return time_cost, integral_cost

    def test_evaluate_returns_finite_cost_and_grad(self):
        opt, ctx = make_context()
        x0 = opt.generate_initial_guess(ctx)
        tc, ic = self._make_cost_fns()
        cost, grad = opt.evaluate(ctx, x0, tc, ic)
        assert np.isfinite(cost)
        assert grad.shape == x0.shape
        assert np.all(np.isfinite(grad))

    def test_energy_only(self):
        """With rho_energy > 0 and zero custom costs, cost should be non-negative."""
        opt, ctx = make_context(rho_energy=1.0)
        x0 = opt.generate_initial_guess(ctx)

        def zero_time(times): return 0.0, np.zeros_like(times)
        def zero_integral(t, tg, seg, step, p, v, a, j, s):
            z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

        cost, _ = opt.evaluate(ctx, x0, zero_time, zero_integral)
        assert cost >= 0.0

    def test_gradient_finite_difference(self):
        """Verify gradients via finite differences (central differences)."""
        opt, ctx = make_context(rho_energy=1.0, steps=16)
        x0 = opt.generate_initial_guess(ctx)

        def zero_time(times): return 0.0, np.zeros_like(times)
        def zero_integral(t, tg, seg, step, p, v, a, j, s):
            z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

        _, grad = opt.evaluate(ctx, x0, zero_time, zero_integral)

        eps = 1e-5
        fd_grad = np.zeros_like(x0)
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps; xm[i] -= eps
            cp, _ = opt.evaluate(ctx, xp, zero_time, zero_integral)
            cm, _ = opt.evaluate(ctx, xm, zero_time, zero_integral)
            fd_grad[i] = (cp - cm) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-3, atol=1e-6)


class TestGetWorkingSpline:
    def test_get_working_spline_after_evaluate(self):
        opt, ctx = make_context()
        x0 = opt.generate_initial_guess(ctx)

        def zero_time(times): return 0.0, np.zeros_like(times)
        def zero_integral(t, tg, s, step, p, v, a, j, sn):
            z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

        opt.evaluate(ctx, x0, zero_time, zero_integral)
        spline = opt.get_working_spline(ctx)
        assert spline.is_initialized
        assert spline.num_segments == 2


class TestOptimizationMask:
    def test_fix_times_optimize_waypoints(self):
        opt = QuinticOptimizer3D()
        opt.set_config(rho_energy=1.0)
        mask = OptimizationMask()
        mask.time = [0, 0]      # fix all time segments
        mask.waypoints = [0, 1, 0]  # optimize only the inner waypoint
        ctx = opt.prepare_context(
            time_segments=TIME_SEGS,
            waypoints=WAYPOINTS,
            bc=BC,
            mask=mask,
        )
        assert ctx.is_valid


class TestConvenienceOptimize:
    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["find_spec"]).find_spec("scipy") is None,
        reason="scipy not installed",
    )
    def test_optimize_runs(self):
        opt, ctx = make_context(rho_energy=2.0)
        result = optimize(opt, ctx, max_iter=10)
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert np.isfinite(result.fun)
