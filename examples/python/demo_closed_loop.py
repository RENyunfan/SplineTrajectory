"""
Drone Racing — Closed-Loop Trajectory Optimization Demo

场景：一架无人机穿越 6 个随机分布的 "gate"，首尾衔接形成一个闭合赛道。
优化目标：最小化总飞行时间 + jerk 能量，同时保证首尾速度/加速度连续（C² 闭合）。

画图：
  左上：3D 闭合赛道
  左下：X-Y 俯视
  右侧：pos / vel / acc / jerk 时间曲线
        灰虚线 = 优化前（等时间）
        红实线 = 优化后（闭合，BC 连续）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from spline_trajectory import (
    QuinticOptimizer3D,
    BoundaryConditions3D,
    OptimizationMask,
    Deriv,
    optimize_closed_loop,
)

_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docs/images")
os.makedirs(_IMAGES_DIR, exist_ok=True)

# ── 1. 随机 gate 位置 ─────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
N_GATES = 6
gates = rng.uniform(low=[0, 0, 0.5], high=[8, 8, 3], size=(N_GATES, 3))
# 均匀分布在圆形赛道附近，加一点随机高度变化，模拟真实赛道
angles = np.linspace(0, 2 * np.pi, N_GATES, endpoint=False)
r = 3.5
gates[:, 0] = 4.0 + r * np.cos(angles) + rng.uniform(-0.8, 0.8, N_GATES)
gates[:, 1] = 4.0 + r * np.sin(angles) + rng.uniform(-0.8, 0.8, N_GATES)
gates[:, 2] = 1.5 + rng.uniform(-0.8, 0.8, N_GATES)

time_segs = [1.2] * N_GATES   # N segments for N gates (closed loop adds one seg)
print(f"Gates ({N_GATES} gates + 1 closing segment = {N_GATES} segments):")
for i, g in enumerate(gates):
    print(f"  Gate {i}: {np.round(g, 2)}")

# ── 2. 优化前：等时间，固定 BC = 0 ──────────────────────────────────────────
opt_before = QuinticOptimizer3D()
opt_before.set_config(rho_energy=0.0)

# For "before", close loop manually but with zero BC (discontinuous junction)
wp_closed = np.vstack([gates, gates[0]])   # N+1 waypoints: close the loop
ctx_before = opt_before.prepare_context(
    time_segments=time_segs,
    waypoints=wp_closed,
    bc=BoundaryConditions3D(),
)
x0 = opt_before.generate_initial_guess(ctx_before)

def zero_t(t): return 0.0, np.zeros_like(t)
def zero_i(t, tg, seg, step, p, v, a, j, s):
    z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

opt_before.evaluate(ctx_before, x0, zero_t, zero_i)
spline_before = opt_before.get_working_spline(ctx_before)

# ── 3. 闭合优化 ───────────────────────────────────────────────────────────────
def time_cost(times):
    """最小化总飞行时间"""
    return float(np.sum(times)), np.ones_like(times)

opt_after = QuinticOptimizer3D()
result, ctx_after = optimize_closed_loop(
    opt_after,
    waypoints=wp_closed,    # last row will be overwritten = first row
    time_segments=time_segs,
    closure_weight=5000.0,  # strong penalty: force BC continuity
    rho_energy=0.5,
    integral_num_steps=64,
    time_cost=time_cost,
    max_iter=500,
)
spline_after = opt_after.get_working_spline(ctx_after)

print(f"\nOptimization: {result.message}")
print(f"  Iterations: {result.nit},  Final cost: {result.fun:.4f}")
print(f"\nBefore: T = {spline_before.duration:.3f} s,  energy = {spline_before.energy:.3f}")
print(f"After : T = {spline_after.duration:.3f} s,  energy = {spline_after.energy:.3f}")

# Verify BC closure
t0, tf = spline_after.start_time, spline_after.end_time
vel_start = spline_after.evaluate(t0, Deriv.Vel)
vel_end   = spline_after.evaluate(tf, Deriv.Vel)
acc_start = spline_after.evaluate(t0, Deriv.Acc)
acc_end   = spline_after.evaluate(tf, Deriv.Acc)
print(f"\nBC closure check (should be ~0):")
print(f"  ||vel_end - vel_start|| = {np.linalg.norm(vel_end - vel_start):.4e}")
print(f"  ||acc_end - acc_start|| = {np.linalg.norm(acc_end - acc_start):.4e}")

# ── 4. 采样 ──────────────────────────────────────────────────────────────────
def sample(spline, n=800):
    ts = np.linspace(spline.start_time, spline.end_time, n)
    tl = ts.tolist()
    return (ts - ts[0],
            spline.evaluate_batch(tl, Deriv.Pos),
            spline.evaluate_batch(tl, Deriv.Vel),
            spline.evaluate_batch(tl, Deriv.Acc),
            spline.evaluate_batch(tl, Deriv.Jerk))

t_b, pos_b, vel_b, acc_b, jrk_b = sample(spline_before)
t_a, pos_a, vel_a, acc_a, jrk_a = sample(spline_after)

# ── 5. 绘图 ──────────────────────────────────────────────────────────────────
LABELS = ["Position (m)", "Velocity (m/s)", "Accel. (m/s²)", "Jerk (m/s³)"]
XYZ    = ["x", "y", "z"]
COLS   = ["#e74c3c", "#2ecc71", "#3498db"]
GATE_COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, N_GATES))

fig = plt.figure(figsize=(20, 9))
fig.suptitle(
    f"Drone Racing  |  Closed-Loop Quintic Trajectory  |  {N_GATES} Gates\n"
    f"Before: T={spline_before.duration:.2f}s  →  "
    f"After: T={spline_after.duration:.2f}s  |  "
    f"Energy: {spline_before.energy:.1f} → {spline_after.energy:.3f}  |  "
    f"BC closure ‖Δv‖={np.linalg.norm(vel_end-vel_start):.1e}  "
    f"‖Δa‖={np.linalg.norm(acc_end-acc_start):.1e}",
    fontsize=10, fontweight="bold", y=1.01,
)

gs = GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.38,
              width_ratios=[1.4, 1, 1, 1, 1])

# ─── 3D 轨迹 ────────────────────────────────────────────────────────────────
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax3d.plot(*pos_b.T, "--", color="silver", lw=1.5, alpha=0.7,
          label=f"Before (T={spline_before.duration:.1f}s)")
ax3d.plot(*pos_a.T,  "-", color="#e74c3c", lw=2.2,
          label=f"After  (T={spline_after.duration:.1f}s)")
for i, (g, c) in enumerate(zip(gates, GATE_COLORS)):
    ax3d.scatter(*g, s=80, color=c, zorder=6)
    ax3d.text(g[0]+0.1, g[1]+0.1, g[2]+0.1, str(i), fontsize=7, color=c)
ax3d.set_xlabel("X", fontsize=8); ax3d.set_ylabel("Y", fontsize=8)
ax3d.set_zlabel("Z", fontsize=8); ax3d.tick_params(labelsize=6)
ax3d.set_title("3D Closed-Loop Trajectory", fontsize=9)
ax3d.legend(fontsize=6)

# ─── X-Y 俯视 ────────────────────────────────────────────────────────────────
ax2d = fig.add_subplot(gs[1, 0])
ax2d.plot(pos_b[:, 0], pos_b[:, 1], "--", color="silver", lw=1.5, alpha=0.7,
          label=f"Before")
ax2d.plot(pos_a[:, 0], pos_a[:, 1],  "-", color="#e74c3c", lw=2.2,
          label=f"After")
for i, (g, c) in enumerate(zip(gates, GATE_COLORS)):
    ax2d.scatter(g[0], g[1], s=80, color=c, zorder=6)
    ax2d.text(g[0]+0.1, g[1]+0.1, str(i), fontsize=8, color=c, fontweight="bold")
# Draw arrows to show direction
mid_idx = len(pos_a) // 4
ax2d.annotate("", xy=pos_a[mid_idx + 5, :2], xytext=pos_a[mid_idx, :2],
              arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
ax2d.set_xlabel("X (m)", fontsize=8); ax2d.set_ylabel("Y (m)", fontsize=8)
ax2d.set_title("Top View (X-Y)  →飞行方向", fontsize=9)
ax2d.legend(fontsize=6); ax2d.set_aspect("equal", "box")
ax2d.grid(True, lw=0.4); ax2d.tick_params(labelsize=7)

# ─── 右侧 4 列：导数曲线 ───────────────────────────────────────────────────
data_b = [pos_b, vel_b, acc_b, jrk_b]
data_a = [pos_a, vel_a, acc_a, jrk_a]
row_labels = [
    f"Before\n(equal T, BC=0)",
    f"After\n(opt T, BC continuous)",
]

for col_idx, label in enumerate(LABELS):
    for row_idx, (t, d, rl) in enumerate(
        zip([t_b, t_a], [data_b[col_idx], data_a[col_idx]], row_labels)
    ):
        ax = fig.add_subplot(gs[row_idx, col_idx + 1])
        for i, (c, dim_lbl) in enumerate(zip(COLS, XYZ)):
            ax.plot(t, d[:, i], color=c, lw=1.6, label=dim_lbl)

        # Mark the loop junction
        T = t[-1]
        ax.axvline(T, color="gray", lw=0.8, ls="--", alpha=0.6)

        ax.set_xlabel("t (s)", fontsize=7)
        if col_idx == 0:
            ax.set_ylabel(rl, fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=3)
        ax.grid(True, lw=0.4, alpha=0.7)
        ax.tick_params(labelsize=7)

plt.savefig(os.path.join(_IMAGES_DIR, "drone_racing_closed_loop.png"), dpi=150, bbox_inches="tight")
print("\nPlot saved → docs/images/drone_racing_closed_loop.png")
plt.show()
