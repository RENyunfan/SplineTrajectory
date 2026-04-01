"""Tests for spline construction and evaluation."""
import numpy as np
import pytest
from spline_trajectory import (
    Deriv,
    BoundaryConditions1D, BoundaryConditions3D, BoundaryConditions6D,
    CubicSpline1D, CubicSpline2D, CubicSpline3D, CubicSpline6D,
    QuinticSpline3D,
    SepticSpline3D,
)


def make_simple_3d_spline():
    waypoints = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
    time_points = [0.0, 1.0, 2.0]
    bc = BoundaryConditions3D(start_velocity=[0.5, 0, 0], end_velocity=[0, -0.5, 0])
    return CubicSpline3D(time_points, waypoints, bc), time_points, waypoints


class TestSplineConstruction:
    def test_basic_construction(self):
        spline, time_points, waypoints = make_simple_3d_spline()
        assert spline.is_initialized
        assert spline.num_segments == 2
        assert spline.start_time == pytest.approx(0.0)
        assert spline.end_time == pytest.approx(2.0)
        assert spline.duration == pytest.approx(2.0)

    def test_default_bc(self):
        waypoints = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
        spline = CubicSpline3D([0.0, 1.0], waypoints)
        assert spline.is_initialized
        assert spline.num_segments == 1

    def test_wrong_waypoint_dim(self):
        waypoints = np.array([[0, 0], [1, 0]], dtype=float)  # 2D, not 3D
        with pytest.raises(Exception):
            CubicSpline3D([0.0, 1.0], waypoints)

    def test_1d_spline(self):
        bc = BoundaryConditions1D(start_velocity=[0.0], end_velocity=[0.0])
        spline = CubicSpline1D([0.0, 1.0, 2.0],
                               np.array([[0.], [1.], [0.]]), bc)
        assert spline.is_initialized
        assert spline.num_segments == 2

    def test_6d_spline(self):
        waypoints = np.zeros((4, 6))
        waypoints[1, :] = 1.0
        waypoints[2, :] = 2.0
        waypoints[3, :] = 1.5
        spline = CubicSpline6D([0.0, 1.0, 2.0, 3.0], waypoints)
        assert spline.is_initialized


class TestSplineEvaluation:
    def test_evaluate_returns_correct_shape(self):
        spline, _, _ = make_simple_3d_spline()
        pos = spline.evaluate(0.5, Deriv.Pos)
        assert pos.shape == (3,)

    def test_interpolates_through_waypoints(self):
        waypoints = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
        spline = CubicSpline3D([0.0, 1.0, 2.0], waypoints)
        # Should pass exactly through start and end waypoints
        pos_start = spline.evaluate(0.0, Deriv.Pos)
        pos_end = spline.evaluate(2.0, Deriv.Pos)
        np.testing.assert_allclose(pos_start, waypoints[0], atol=1e-10)
        np.testing.assert_allclose(pos_end, waypoints[-1], atol=1e-10)

    def test_boundary_velocity(self):
        waypoints = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        sv = np.array([0.5, 0.1, 0.0])
        ev = np.array([0.0, -0.3, 0.0])
        bc = BoundaryConditions3D(start_velocity=sv, end_velocity=ev)
        spline = CubicSpline3D([0.0, 1.0], waypoints, bc)
        vel_start = spline.evaluate(0.0, Deriv.Vel)
        vel_end = spline.evaluate(1.0, Deriv.Vel)
        np.testing.assert_allclose(vel_start, sv, atol=1e-8)
        np.testing.assert_allclose(vel_end, ev, atol=1e-8)

    def test_evaluate_batch_shape(self):
        spline, _, _ = make_simple_3d_spline()
        times = np.linspace(0.0, 2.0, 30).tolist()
        positions = spline.evaluate_batch(times, Deriv.Pos)
        assert positions.shape == (30, 3)

    def test_evaluate_all_derivs(self):
        spline, _, _ = make_simple_3d_spline()
        t = 1.0
        for d in [Deriv.Pos, Deriv.Vel, Deriv.Acc, Deriv.Jerk]:
            val = spline.evaluate(t, d)
            assert val.shape == (3,)

    def test_default_deriv_is_pos(self):
        spline, _, _ = make_simple_3d_spline()
        assert np.allclose(spline.evaluate(1.0), spline.evaluate(1.0, Deriv.Pos))

    def test_batch_matches_scalar(self):
        spline, _, _ = make_simple_3d_spline()
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        batch = spline.evaluate_batch(times, Deriv.Vel)
        for i, t in enumerate(times):
            np.testing.assert_allclose(batch[i], spline.evaluate(t, Deriv.Vel), atol=1e-12)

    def test_energy_positive(self):
        spline, _, _ = make_simple_3d_spline()
        assert spline.energy >= 0.0


class TestAllSplineTypes:
    """Smoke test each spline type and a few dim variants."""

    def _make_spline(self, SplineCls, dim):
        n = 3
        wp = np.random.default_rng(42).uniform(0, 1, (n, dim))
        times = [0.0] + list(np.cumsum(np.ones(n - 1)))
        return SplineCls(times, wp)

    def test_cubic_3d(self):
        s = self._make_spline(CubicSpline3D, 3)
        assert s.is_initialized

    def test_quintic_3d(self):
        s = self._make_spline(QuinticSpline3D, 3)
        assert s.is_initialized

    def test_septic_3d(self):
        s = self._make_spline(SepticSpline3D, 3)
        assert s.is_initialized

    def test_cubic_2d(self):
        s = self._make_spline(CubicSpline2D, 2)
        pos = s.evaluate(0.5, Deriv.Pos)
        assert pos.shape == (2,)


class TestBoundaryConditions:
    def test_default_bc_is_zero(self):
        bc = BoundaryConditions3D()
        np.testing.assert_allclose(bc.start_velocity, np.zeros(3))
        np.testing.assert_allclose(bc.end_velocity, np.zeros(3))

    def test_set_properties(self):
        bc = BoundaryConditions3D()
        bc.start_velocity = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(bc.start_velocity, [1.0, 2.0, 3.0])

    def test_full_constructor(self):
        sv = np.ones(3)
        sa = np.ones(3) * 2
        sj = np.ones(3) * 3
        ev = -np.ones(3)
        ea = -np.ones(3) * 2
        ej = -np.ones(3) * 3
        bc = BoundaryConditions3D(sv, sa, sj, ev, ea, ej)
        np.testing.assert_allclose(bc.start_jerk, sj)
        np.testing.assert_allclose(bc.end_jerk, ej)
