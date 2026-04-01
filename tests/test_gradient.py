"""Tests for gradient correctness of PyTimeCost and PyIntegralCost wrappers."""
import numpy as np
import pytest
from spline_trajectory import QuinticOptimizer3D, BoundaryConditions3D

WAYPOINTS = np.array([[0, 0, 0], [1, 0.5, 0], [2, 1, 0.5]], dtype=float)
BC = BoundaryConditions3D(start_velocity=[0.1, 0, 0], end_velocity=[0, -0.1, 0])


def make_context(rho_energy=0.0, integral_steps=32):
    opt = QuinticOptimizer3D()
    opt.set_config(rho_energy=rho_energy, integral_num_steps=integral_steps)
    ctx = opt.prepare_context(
        time_segments=[1.0, 1.5],
        waypoints=WAYPOINTS,
        bc=BC,
    )
    return opt, ctx


def finite_diff_grad(opt, ctx, x, time_cost, integral_cost, eps=1e-5):
    fd = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        cp, _ = opt.evaluate(ctx, xp, time_cost, integral_cost)
        cm, _ = opt.evaluate(ctx, xm, time_cost, integral_cost)
        fd[i] = (cp - cm) / (2 * eps)
    return fd


class TestTimeCostGradient:
    def test_linear_time_cost_grad(self):
        """Linear time cost: sum(times). Gradient should match FD."""
        opt, ctx = make_context(rho_energy=0.0)
        x0 = opt.generate_initial_guess(ctx)

        def time_cost(times):
            return float(np.sum(times)), np.ones_like(times)

        def zero_integral(t, tg, seg, step, p, v, a, j, s):
            z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

        _, analytic_grad = opt.evaluate(ctx, x0, time_cost, zero_integral)
        fd_grad = finite_diff_grad(opt, ctx, x0, time_cost, zero_integral)
        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-3, atol=1e-5)

    def test_quadratic_time_cost_grad(self):
        """Quadratic time cost: sum(T^2). Gradient = 2T."""
        opt, ctx = make_context(rho_energy=0.0)
        x0 = opt.generate_initial_guess(ctx)

        def time_cost(times):
            return float(np.sum(times ** 2)), 2.0 * times

        def zero_integral(t, tg, seg, step, p, v, a, j, s):
            z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

        _, analytic_grad = opt.evaluate(ctx, x0, time_cost, zero_integral)
        fd_grad = finite_diff_grad(opt, ctx, x0, time_cost, zero_integral)
        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-3, atol=1e-5)


class TestIntegralCostGradient:
    def test_velocity_penalty_grad(self):
        """Integral cost: penalise squared velocity norm. Gradient w.r.t. v = 2v."""
        opt, ctx = make_context(rho_energy=0.0, integral_steps=32)
        x0 = opt.generate_initial_guess(ctx)

        def zero_time(times): return 0.0, np.zeros_like(times)

        def vel_cost(t, tg, seg, step, p, v, a, j, s):
            cost = float(np.dot(v, v))
            gv = 2.0 * v
            z = np.zeros_like(p)
            return cost, z, gv, z, z, z, 0.0

        _, analytic_grad = opt.evaluate(ctx, x0, zero_time, vel_cost)
        fd_grad = finite_diff_grad(opt, ctx, x0, zero_time, vel_cost)
        # Larger tolerance for integral cost (quadrature approximation)
        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=5e-3, atol=1e-4)

    def test_combined_cost_grad(self):
        """Energy regularization + time cost — combined gradient test."""
        opt, ctx = make_context(rho_energy=1.0, integral_steps=32)
        x0 = opt.generate_initial_guess(ctx)

        def time_cost(times): return float(np.sum(times)), np.ones_like(times)

        def zero_integral(t, tg, seg, step, p, v, a, j, s):
            z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

        _, analytic_grad = opt.evaluate(ctx, x0, time_cost, zero_integral)
        fd_grad = finite_diff_grad(opt, ctx, x0, time_cost, zero_integral)
        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-3, atol=1e-5)
