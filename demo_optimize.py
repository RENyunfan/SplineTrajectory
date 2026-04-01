"""
随机 waypoint 轨迹优化演示
- 随机生成 6 个 3D waypoints（轨迹必须全部经过）
- 用 QuinticOptimizer3D 优化每段时间分配（固定 waypoints，只优化 time segments）
- Before：等时间分配（每段 1s）
- After ：优化后的时间分配（最小化总时间 + jerk 能量）
- 画出：
    最左列：3D 轨迹 + X-Y 俯视图
    右侧 4 列：pos / vel / acc / jerk 随时间曲线（上=优化前，下=优化后）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from spline_trajectory import (
    QuinticOptimizer3D,
    BoundaryConditions3D,
    OptimizationMask,
    Deriv,
    optimize,
)

# ── 1. 随机 waypoints ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
N_WP = 6
N_SEG = N_WP - 1

waypoints = rng.uniform(0, 5, size=(N_WP, 3))
waypoints[0]  = [0.0, 0.0, 0.0]
waypoints[-1] = [5.0, 5.0, 2.0]
print("Waypoints:\n", np.round(waypoints, 3))

bc = BoundaryConditions3D()   # 起止速度/加速度均为零

# ── 2. mask：固定所有 waypoints，只优化时间段 ──────────────────────────────
mask = OptimizationMask()
mask.waypoints = [0] * N_WP   # 全部固定 → 轨迹必须经过所有给定 waypoints
mask.time      = [1] * N_SEG  # 所有时间段自由优化

# ── 3. Before：等时间初始轨迹 ────────────────────────────────────────────
opt_before = QuinticOptimizer3D()
opt_before.set_config(rho_energy=0.0)   # 纯等时间，不优化
ctx_before = opt_before.prepare_context(
    time_segments=[1.0] * N_SEG,
    waypoints=waypoints,
    bc=bc,
    mask=mask,
)
x0 = opt_before.generate_initial_guess(ctx_before)

def zero_time(t): return 0.0, np.zeros_like(t)
def zero_int(t, tg, seg, step, p, v, a, j, s):
    z = np.zeros_like(p); return 0.0, z, z, z, z, z, 0.0

opt_before.evaluate(ctx_before, x0, zero_time, zero_int)
spline_before = opt_before.get_working_spline(ctx_before)

# ── 4. After：优化时间分配，最小化总时间 + jerk 能量 ──────────────────────
opt_after = QuinticOptimizer3D()
opt_after.set_config(rho_energy=1.0, integral_num_steps=64)
ctx_after = opt_after.prepare_context(
    time_segments=[1.0] * N_SEG,
    waypoints=waypoints,
    bc=bc,
    mask=mask,
)

def time_cost(times):
    """最小化总时间"""
    return float(np.sum(times)), np.ones_like(times)

result = optimize(opt_after, ctx_after, time_cost=time_cost, integral_cost=None, max_iter=300)
print(f"\nOptimization: {result.message}")
print(f"  Iterations: {result.nit},  Final cost: {result.fun:.4f}")

spline_after = opt_after.get_working_spline(ctx_after)

print(f"\nBefore: T = {spline_before.duration:.3f} s,  energy = {spline_before.energy:.3f}")
print(f"After : T = {spline_after.duration:.3f} s,  energy = {spline_after.energy:.3f}")

# ── 5. 采样 ──────────────────────────────────────────────────────────────────
def sample(spline, n=600):
    ts = np.linspace(spline.start_time, spline.end_time, n)
    tl = ts.tolist()
    return (ts - ts[0],
            spline.evaluate_batch(tl, Deriv.Pos),
            spline.evaluate_batch(tl, Deriv.Vel),
            spline.evaluate_batch(tl, Deriv.Acc),
            spline.evaluate_batch(tl, Deriv.Jerk))

t_b, pos_b, vel_b, acc_b, jrk_b = sample(spline_before)
t_a, pos_a, vel_a, acc_a, jrk_a = sample(spline_after)

# ── 6. 绘图 ──────────────────────────────────────────────────────────────────
LABELS = ["Position (m)", "Velocity (m/s)", "Accel. (m/s²)", "Jerk (m/s³)"]
XYZ    = ["x", "y", "z"]
COLS   = ["#e74c3c", "#2ecc71", "#3498db"]

fig = plt.figure(figsize=(20, 8))
fig.suptitle(
    "QuinticSpline3D  |  Fixed Waypoints, Optimized Time Allocation\n"
    f"Before: equal T={spline_before.duration:.2f}s (each 1s)  →  "
    f"After: optimized T={spline_after.duration:.2f}s  |  "
    f"Energy: {spline_before.energy:.1f} → {spline_after.energy:.3f}",
    fontsize=11, fontweight="bold", y=1.02
)

gs = GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.38,
              width_ratios=[1.3, 1, 1, 1, 1])

# ─── 左上：3D ────────────────────────────────────────────────────────────────
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax3d.plot(*pos_b.T, "--", color="silver", lw=1.5, label=f"Before (T={spline_before.duration:.1f}s)")
ax3d.plot(*pos_a.T,  "-", color="#e74c3c", lw=2.0, label=f"After  (T={spline_after.duration:.1f}s)")
ax3d.scatter(*waypoints.T, s=60, c="k", zorder=5, label="Waypoints")
ax3d.scatter(*waypoints[0],  s=100, c="#2ecc71", zorder=6, marker="^")
ax3d.scatter(*waypoints[-1], s=100, c="#e74c3c", zorder=6, marker="*")
ax3d.set_xlabel("X", fontsize=8); ax3d.set_ylabel("Y", fontsize=8)
ax3d.set_zlabel("Z", fontsize=8); ax3d.tick_params(labelsize=6)
ax3d.set_title("3D Trajectory", fontsize=9)
ax3d.legend(fontsize=6)

# ─── 左下：X-Y 俯视 ──────────────────────────────────────────────────────────
ax2d = fig.add_subplot(gs[1, 0])
ax2d.plot(pos_b[:, 0], pos_b[:, 1], "--", color="silver", lw=1.5,
          label=f"Before (T={spline_before.duration:.1f}s)")
ax2d.plot(pos_a[:, 0], pos_a[:, 1],  "-", color="#e74c3c", lw=2.0,
          label=f"After  (T={spline_after.duration:.1f}s)")
ax2d.scatter(waypoints[:, 0], waypoints[:, 1], s=60, c="k", zorder=5)
ax2d.scatter(*waypoints[0, :2],  s=100, c="#2ecc71", zorder=6, marker="^")
ax2d.scatter(*waypoints[-1, :2], s=100, c="#e74c3c", zorder=6, marker="*")
ax2d.set_xlabel("X (m)", fontsize=8); ax2d.set_ylabel("Y (m)", fontsize=8)
ax2d.set_title("Top View (X-Y)", fontsize=9)
ax2d.legend(fontsize=6); ax2d.set_aspect("equal", "box")
ax2d.grid(True, lw=0.4); ax2d.tick_params(labelsize=7)

# ─── 右侧 4 列：每阶导数，上行=Before，下行=After ──────────────────────────
data_b = [pos_b, vel_b, acc_b, jrk_b]
data_a = [pos_a, vel_a, acc_a, jrk_a]
row_labels = [
    f"Before\n(equal T, {spline_before.duration:.1f}s)",
    f"After\n(opt T, {spline_after.duration:.1f}s)",
]

for col_idx, label in enumerate(LABELS):
    for row_idx, (t, d, rl) in enumerate(zip([t_b, t_a], [data_b[col_idx], data_a[col_idx]], row_labels)):
        ax = fig.add_subplot(gs[row_idx, col_idx + 1])
        for i, (c, dim) in enumerate(zip(COLS, XYZ)):
            ax.plot(t, d[:, i], color=c, lw=1.6, label=dim)
        ax.set_xlabel("t (s)", fontsize=7)
        if col_idx == 0:
            ax.set_ylabel(rl, fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=3)
        ax.grid(True, lw=0.4, alpha=0.7)
        ax.tick_params(labelsize=7)

plt.savefig("trajectory_optimization.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → trajectory_optimization.png")
plt.show()
