"""
Quadrotor Trajectory with Thrust & Body-Rate Constraints
=========================================================
Constraint model: differential-flatness (zero drag, yaw=0, mass=1 kg)
  Thrust:     f   = ‖a + g·ẑ‖          (N/kg)
  Body rate:  ω   from exact flatness map (Zhepei Wang, GCOPTER)

Spline order: SepticSplineND (7th order, S4-MINCO)
  → exposes jerk as boundary-condition variable
  → start/end jerk = 0  ⟺  body rate = 0  (hard hover constraint)

Comparison:
  Before (row 1):  fixed short segment times, no penalty   → shows violations
  After  (row 2):  time optimized, flatness penalty active  → feasible, longer

Forward/backward ported from flatness::FlatnessMap (SUPER, HKU-MARS).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from spline_trajectory import (
    SepticOptimizer3D,
    BoundaryConditions3D,
    OptimizationMask,
    Deriv,
    optimize,
)

_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docs/images")
os.makedirs(_IMAGES_DIR, exist_ok=True)

# ── physical limits ────────────────────────────────────────────────────────────
GRAV = 9.81
F_MIN = 4.0  # m/s²
F_MAX = 20.0  # m/s²
OMEGA_MAX = 2.5  # rad/s
W_THRUST = 300.0
W_OMEGA = 200.0
W_TIME = 100.0

# ── differential flatness (zero drag, yaw=0, mass=1) ──────────────────────────


class FlatnessMap:
    """Exact forward/backward for thrust and body-rate.

    Ported from flatness::FlatnessMap (Zhepei Wang, GCOPTER;
    adapted in SUPER by Yunfan Ren, HKU-MARS Lab).
    Simplifications: cp = dh = dv = 0,  psi = dpsi = 0,  mass = 1.
    """

    def forward(self, vel: np.ndarray, acc: np.ndarray, jerk: np.ndarray) -> tuple:
        """Returns (thrust: float, omega: ndarray(3,))."""
        a0, a1, a2 = acc
        j0, j1, j2 = jerk

        zu0, zu1, zu2 = a0, a1, a2 + GRAV
        zu_sqr0 = zu0 * zu0
        zu_sqr1 = zu1 * zu1
        zu_sqr2 = zu2 * zu2
        zu01 = zu0 * zu1
        zu12 = zu1 * zu2
        zu02 = zu0 * zu2
        zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2
        zu_norm = np.sqrt(zu_sqr_norm)

        z0 = zu0 / zu_norm
        z1 = zu1 / zu_norm
        z2 = zu2 / zu_norm

        ng_den = zu_sqr_norm * zu_norm
        ng00 = (zu_sqr1 + zu_sqr2) / ng_den
        ng01 = -zu01 / ng_den
        ng02 = -zu02 / ng_den
        ng11 = (zu_sqr0 + zu_sqr2) / ng_den
        ng12 = -zu12 / ng_den
        ng22 = (zu_sqr0 + zu_sqr1) / ng_den

        dz0 = ng00 * j0 + ng01 * j1 + ng02 * j2
        dz1 = ng01 * j0 + ng11 * j1 + ng12 * j2
        dz2 = ng02 * j0 + ng12 * j1 + ng22 * j2

        thr = z0 * a0 + z1 * a1 + z2 * (a2 + GRAV)  # == zu_norm

        omg_den = z2 + 1.0
        omg_term = dz2 / omg_den
        omg = np.array(
            [
                -dz1 + z1 * omg_term,
                dz0 - z0 * omg_term,
                (z1 * dz0 - z0 * dz1) / omg_den,
            ]
        )

        self._c = dict(
            a0=a0,
            a1=a1,
            a2=a2,
            z0=z0,
            z1=z1,
            z2=z2,
            zu0=zu0,
            zu1=zu1,
            zu2=zu2,
            zu_sqr0=zu_sqr0,
            zu_sqr1=zu_sqr1,
            zu_sqr2=zu_sqr2,
            zu01=zu01,
            zu12=zu12,
            zu02=zu02,
            zu_sqr_norm=zu_sqr_norm,
            zu_norm=zu_norm,
            ng_den=ng_den,
            ng00=ng00,
            ng01=ng01,
            ng02=ng02,
            ng11=ng11,
            ng12=ng12,
            ng22=ng22,
            j0=j0,
            j1=j1,
            j2=j2,
            dz0=dz0,
            dz1=dz1,
            dz2=dz2,
            f_term0=a0,
            f_term1=a1,
            f_term2=a2 + GRAV,
            omg_den=omg_den,
            omg_term=omg_term,
        )
        return float(thr), omg

    def backward(self, thr_grad: float, omg_grad: np.ndarray) -> tuple:
        """Returns (gv, ga, gj) — shapes (3,)."""
        c = self._c
        g0, g1, g2 = omg_grad
        tf = thr_grad

        # ── adjoint through omega (psi=0, quat_grad=0) ────────────────────────
        omg_termb = -c["z0"] * g1 + c["z1"] * g0

        tempb = g2 / c["omg_den"]
        z1b = c["dz0"] * tempb
        dz0b = c["z1"] * tempb + g1
        z0b = -c["dz1"] * tempb
        dz1b = -c["z0"] * tempb - g0
        omg_denb = -(c["z1"] * c["dz0"] - c["z0"] * c["dz1"]) * tempb / c[
            "omg_den"
        ] - c["dz2"] * omg_termb / (c["omg_den"] ** 2)

        tempb = -(c["omg_term"] * g1)
        z0b += tempb  # omg[1] contribution

        tempb = -(c["omg_term"] * g0)
        z1b += -tempb  # omg[0] contribution

        # ── thrust: ∂thr/∂z  (thr = z · f_term, ∂thr/∂z_i = f_term_i) ───────
        z0b += c["f_term0"] * tf
        z1b += c["f_term1"] * tf

        dz2b = omg_termb / c["omg_den"]
        z2b = omg_denb + c["f_term2"] * tf

        # ── adjoint through dz = ng @ jerk ────────────────────────────────────
        ng02b = c["j0"] * dz2b + c["j2"] * dz0b
        dz_term0b = c["ng02"] * dz2b + c["ng01"] * dz1b + c["ng00"] * dz0b
        ng12b = c["j1"] * dz2b + c["j2"] * dz1b
        dz_term1b = c["ng12"] * dz2b + c["ng11"] * dz1b + c["ng01"] * dz0b
        ng22b = c["j2"] * dz2b
        dz_term2b = c["ng22"] * dz2b + c["ng12"] * dz1b + c["ng02"] * dz0b
        ng01b = c["j0"] * dz1b + c["j1"] * dz0b
        ng11b = c["j1"] * dz1b
        ng00b = c["j0"] * dz0b

        # ── adjoint through ng ────────────────────────────────────────────────
        ng_den = c["ng_den"]

        tempb = ng22b / ng_den
        zu_sqr0b = tempb
        zu_sqr1b = tempb
        ng_denb = -(c["zu_sqr0"] + c["zu_sqr1"]) * tempb / ng_den

        zu12b = -ng12b / ng_den
        tempb = ng11b / ng_den
        ng_denb += (
            c["zu12"] * ng12b / (ng_den**2)
            - (c["zu_sqr0"] + c["zu_sqr2"]) * tempb / ng_den
        )
        zu_sqr0b += tempb
        zu_sqr2b = tempb

        zu02b = -ng02b / ng_den
        zu01b = -ng01b / ng_den
        tempb = ng00b / ng_den
        ng_denb += (
            c["zu02"] * ng02b / (ng_den**2)
            + c["zu01"] * ng01b / (ng_den**2)
            - (c["zu_sqr1"] + c["zu_sqr2"]) * tempb / ng_den
        )

        # ── adjoint through z = zu / zu_norm ──────────────────────────────────
        zu_norm = c["zu_norm"]
        zu_sqr_norm = c["zu_sqr_norm"]

        zu_normb = (
            zu_sqr_norm * ng_denb
            - (c["zu2"] * z2b + c["zu1"] * z1b + c["zu0"] * z0b) / zu_sqr_norm
        )
        zu_sqr_normb = zu_norm * ng_denb + zu_normb / (2.0 * zu_norm)

        tempb += zu_sqr_normb
        zu_sqr1b += tempb
        zu_sqr2b += tempb

        zu2b = (
            z2b / zu_norm
            + c["zu0"] * zu02b
            + c["zu1"] * zu12b
            + 2 * c["zu2"] * zu_sqr2b
        )
        zu1b = (
            z1b / zu_norm
            + c["zu2"] * zu12b
            + c["zu0"] * zu01b
            + 2 * c["zu1"] * zu_sqr1b
        )
        zu_sqr0b += zu_sqr_normb
        zu0b = (
            z0b / zu_norm
            + c["zu2"] * zu02b
            + c["zu1"] * zu01b
            + 2 * c["zu0"] * zu_sqr0b
        )

        # zu = a + [0,0,g]  →  ∂zu/∂a = I
        # thr = z · f_term, f_term = [a0, a1, a2+g]
        #   direct contribution: ∂thr/∂a_i via f_term_i  (C++ acc_total_grad += mass*f_term_ib)
        gv = np.zeros(3)
        ga = np.array([zu0b, zu1b, zu2b])
        ga += np.array([c["z0"], c["z1"], c["z2"]]) * tf  # direct f_term path
        gj = np.array([dz_term0b, dz_term1b, dz_term2b])
        return gv, ga, gj


# ── trajectory ────────────────────────────────────────────────────────────────
# Sharp turns + altitude changes; start/end are hover (v=a=j=0)
waypoints = np.array(
    [
        [0.0, 0.0, 1.0],
        [7.0, 0.0, 1.0],  # fast straight
        [7.0, 7.0, 5.0],  # sharp 90° right turn + climb
        [0.0, 7.0, 1.0],  # sharp 90° left turn  + drop
        [0.0, 0.0, 4.0],  # return + climb
    ],
    dtype=float,
)
N_WP = len(waypoints)
N_SEG = N_WP - 1

T_SHORT = 0.8  # forces fast flight that violates constraints

# Hover BC: vel=acc=jerk=0  (jerk=0  ⟺  ω=0 at endpoints)
BC_HOVER = BoundaryConditions3D(
    start_velocity=[0, 0, 0],
    start_acceleration=[0, 0, 0],
    start_jerk=[0, 0, 0],
    end_velocity=[0, 0, 0],
    end_acceleration=[0, 0, 0],
    end_jerk=[0, 0, 0],
)


def make_mask(optimize_time: bool) -> OptimizationMask:
    m = OptimizationMask()
    m.waypoints = [0] * N_WP  # all positions fixed (hard constraints)
    m.time = [int(optimize_time)] * N_SEG
    return m


# ── cost functions ─────────────────────────────────────────────────────────────
_fm = FlatnessMap()


def time_cost(times):
    return W_TIME * float(np.sum(times)), W_TIME * np.ones_like(times)


def flatness_penalty(t, t_global, seg, step, p, v, a, j, s):
    thr, omg = _fm.forward(v, a, j)
    omega = float(np.linalg.norm(omg))

    cost = 0.0
    tf_grad = 0.0
    og_vec = np.zeros(3)

    lo = max(F_MIN - thr, 0.0)
    if lo > 0.0:
        cost += W_THRUST * 0.5 * lo**2
        tf_grad -= W_THRUST * lo

    hi = max(thr - F_MAX, 0.0)
    if hi > 0.0:
        cost += W_THRUST * 0.5 * hi**2
        tf_grad += W_THRUST * hi

    ew = max(omega - OMEGA_MAX, 0.0)
    if ew > 0.0 and omega > 1e-9:
        cost += W_OMEGA * 0.5 * ew**2
        og_vec += W_OMEGA * ew * omg / omega

    gv, ga, gj = _fm.backward(tf_grad, og_vec)
    z = np.zeros(3)
    return cost, z, gv, ga, gj, z, 0.0


def zero_integral(t, tg, seg, step, p, v, a, j, s):
    z = np.zeros(3)
    return 0.0, z, z, z, z, z, 0.0


def zero_time(times):
    return 0.0, np.zeros_like(times)


# ── physical state from spline ────────────────────────────────────────────────
def compute_pvwr(spline, n=600):
    ts = np.linspace(spline.start_time, spline.end_time, n)
    tl = ts.tolist()
    pos = spline.evaluate_batch(tl, Deriv.Pos)
    vel = spline.evaluate_batch(tl, Deriv.Vel)
    acc = spline.evaluate_batch(tl, Deriv.Acc)
    jerk = spline.evaluate_batch(tl, Deriv.Jerk)

    thrust = np.zeros(n)
    omega = np.zeros(n)
    fm = FlatnessMap()
    for i in range(n):
        thr, omg = fm.forward(vel[i], acc[i], jerk[i])
        thrust[i] = thr
        omega[i] = float(np.linalg.norm(omg))
    speed = np.linalg.norm(vel, axis=1)
    return ts - ts[0], pos, speed, thrust, omega


# ── 1. unconstrained: fixed short times ───────────────────────────────────────
print("Building unconstrained trajectory (fixed time, no penalty)...")
opt0 = SepticOptimizer3D()
opt0.set_config(rho_energy=0.02, integral_num_steps=80)
ctx0 = opt0.prepare_context(
    time_segments=[T_SHORT] * N_SEG,
    waypoints=waypoints,
    bc=BC_HOVER,
    mask=make_mask(optimize_time=False),
)
x0 = opt0.generate_initial_guess(ctx0)
opt0.evaluate(ctx0, x0, zero_time, zero_integral)
sp0 = opt0.get_working_spline(ctx0)
t0, p0, v0, w0, r0 = compute_pvwr(sp0)

# ── 2. constrained: time free + flatness penalty ───────────────────────────────
print("Running constrained optimization (time free + flatness penalty)...")
opt1 = SepticOptimizer3D()
opt1.set_config(rho_energy=0.02, integral_num_steps=80)
ctx1 = opt1.prepare_context(
    time_segments=[T_SHORT] * N_SEG,
    waypoints=waypoints,
    bc=BC_HOVER,
    mask=make_mask(optimize_time=True),
)
result = optimize(
    opt1, ctx1, time_cost=time_cost, integral_cost=flatness_penalty, max_iter=500
)
sp1 = opt1.get_working_spline(ctx1)
t1, p1, v1, w1, r1 = compute_pvwr(sp1)

n_s = len(w0)
f_viol = (w0 < F_MIN).sum() + (w0 > F_MAX).sum()
w_viol = (r0 > OMEGA_MAX).sum()
print(f"\nOptimization: {result.message}")
print(
    f"Before  T={sp0.duration:.2f}s  f=[{w0.min():.2f},{w0.max():.2f}]  ω_max={r0.max():.2f}"
)
print(
    f"After   T={sp1.duration:.2f}s  f=[{w1.min():.2f},{w1.max():.2f}]  ω_max={r1.max():.2f}"
)
print(f"  Thrust violations (before): {f_viol}/{n_s}")
print(f"  ω     violations (before): {w_viol}/{n_s}")

# ── plot ───────────────────────────────────────────────────────────────────────
COLS_XYZ = ["#e74c3c", "#2ecc71", "#3498db"]
fig = plt.figure(figsize=(20, 8))
fig.suptitle(
    "Quadrotor Trajectory — Thrust & Body-Rate Constraints "
    "(SepticSpline, Differential Flatness, Hover BC: jerk=0)\n"
    f"F ∈ [{F_MIN}, {F_MAX}] m/s²   ‖ω‖ ≤ {OMEGA_MAX} rad/s   |   "
    "Row 1: before (fixed short times)   Row 2: after (time optimized)",
    fontsize=9,
    fontweight="bold",
)
gs = GridSpec(2, 5, figure=fig, hspace=0.6, wspace=0.42, width_ratios=[1.7, 1, 1, 1, 1])

ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax3d.plot(
    *p0.T,
    "--",
    color="silver",
    lw=1.5,
    alpha=0.9,
    label=f"Before  T={sp0.duration:.1f}s",
)
ax3d.plot(*p1.T, "-", color="#e74c3c", lw=2.2, label=f"After   T={sp1.duration:.1f}s")
ax3d.scatter(*waypoints.T, s=90, color="#e67e22", zorder=6)
for i, wp in enumerate(waypoints):
    ax3d.text(
        wp[0] + 0.2, wp[1] + 0.2, wp[2] + 0.2, str(i), fontsize=9, fontweight="bold"
    )
ax3d.set_xlabel("X (m)", fontsize=8)
ax3d.set_ylabel("Y (m)", fontsize=8)
ax3d.set_zlabel("Z (m)", fontsize=8)
ax3d.tick_params(labelsize=6)
ax3d.set_title("3D Trajectory", fontsize=9)
ax3d.legend(fontsize=7)


def plot_col(
    col, title, d_unc, d_con, t_unc, t_con, labels, colors, hlines=None, first_col=False
):
    for r, (t, d) in enumerate([(t_unc, d_unc), (t_con, d_con)]):
        ax = fig.add_subplot(gs[r, col])
        if d.ndim == 1:
            lc = "#bbbbbb" if r == 0 else colors[0]
            ls = "--" if r == 0 else "-"
            ax.fill_between(t, d, alpha=0.12 if r == 1 else 0, color=lc)
            ax.plot(t, d, color=lc, lw=1.8, ls=ls, label=labels[0])
        else:
            for i, (lbl, c) in enumerate(zip(labels, colors)):
                lc = "#cccccc" if r == 0 else c
                ls = "--" if r == 0 else "-"
                ax.plot(t, d[:, i], color=lc, lw=1.6, ls=ls, label=lbl)
        if hlines:
            for val, hc, hlbl in hlines:
                ax.axhline(val, color=hc, lw=1.5, ls="--", alpha=0.9, label=hlbl)
        if r == 0:
            ax.set_title(title, fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        if first_col:
            ax.set_ylabel(
                "Before" if r == 0 else "After", fontsize=9, fontweight="bold"
            )
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.grid(True, lw=0.4, alpha=0.6)
        ax.tick_params(labelsize=7)


plot_col(1, "Position (m)", p0, p1, t0, t1, ["x", "y", "z"], COLS_XYZ, first_col=True)
plot_col(2, "Speed ‖v‖ (m/s)", v0, v1, t0, t1, ["‖v‖"], ["#8e44ad"])
plot_col(
    3,
    "Thrust f (m/s²)",
    w0,
    w1,
    t0,
    t1,
    ["f"],
    ["#9b59b6"],
    hlines=[(F_MIN, "#c0392b", f"F_min={F_MIN}"), (F_MAX, "#c0392b", f"F_max={F_MAX}")],
)
plot_col(
    4,
    "Body rate ‖ω‖ (rad/s)",
    r0,
    r1,
    t0,
    t1,
    ["‖ω‖"],
    ["#16a085"],
    hlines=[(OMEGA_MAX, "#c0392b", f"ω_max={OMEGA_MAX}")],
)

plt.savefig(os.path.join(_IMAGES_DIR, "quadrotor_trajectory.png"), dpi=150, bbox_inches="tight")
print("\nPlot saved → docs/images/quadrotor_trajectory.png")
plt.show()
