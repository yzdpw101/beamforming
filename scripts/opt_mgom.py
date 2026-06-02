"""MGOM 稀布面阵 — 线阵公式加速 (phi=0用x, phi=90用y)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, time
from antopt.mapping import MGOM
from antopt.antenna import Pattern
from antopt.analysis import get_psll
from antopt.opt.solver import minimize

# ── 阵列 ──
Ne, Lx, Ly, dmin = 108, 9.5, 4.5, 0.5
mapper = MGOM(Ne=Ne, Lx=Lx, Ly=Ly, dmin=dmin, is_symmetric=True)
print(f"Nq={mapper.Nq}  grid={mapper.max_rows}x{mapper.max_cols}  vars={mapper.n_vars}")

# ── 线阵 Pattern (theta=[0,90]) ──
pat = Pattern(theta_deg_start=0, theta_deg_end=90, theta_deg_step=0.1)
theta = pat.theta_deg

def _psll_of(positions):
    """线阵 PSLL"""
    # af = pat.linear_af(positions)
    af = pat.linear_af_symmetric(positions)
    af_db = Pattern.to_dB(af)
    psll, _ = get_psll(af_db, theta)
    return psll if not np.isinf(psll) else 0.0

def fitness(x):
    # fx, fy = mapper.synthesize(x, expand_to_4q=True)
    fx, fy = mapper.synthesize(x, expand_to_4q=False)

    # phi=0: 只看 x 坐标 (线阵公式)
    psll0 = _psll_of(fx)
    # phi=90: 只看 y 坐标 (线阵公式)
    psll90 = _psll_of(fy)

    # 主瓣指向惩罚
    af_x = pat.linear_af(fx)
    peak_idx = np.argmax(np.abs(af_x))
    peak_t = theta[peak_idx]
    pointing = (peak_t - 0.0) ** 2 * 10

    return float(psll0 + psll90 + pointing)

# ── 优化 ──
lb = np.zeros(mapper.n_vars); ub = np.ones(mapper.n_vars)
print(f"\nCMA-ES 优化中...")
t0 = time.perf_counter()
result = minimize(
    fitness, mapper.n_vars,
    method="cma", bounds=(lb, ub),
    sigma=0.5, pop_size=200, max_iter=1000,
    n_jobs=-1, verbose=True, seed=42,
)
elapsed = time.perf_counter() - t0

# ── 结果 ──
fx, fy = mapper.synthesize(result["x"], expand_to_4q=True)
psll0 = _psll_of(fx)
psll90 = _psll_of(fy)

dist = np.sqrt((fx[:,None]-fx[None,:])**2 + (fy[:,None]-fy[None,:])**2)
np.fill_diagonal(dist, np.inf)

print(f"\n耗时: {elapsed:.1f}s  fitness: {result['f']:.2f}")
print(f"  phi=0  PSLL={psll0:.2f}dB")
print(f"  phi=90 PSLL={psll90:.2f}dB")
print(f"  和: {psll0+psll90:.2f}dB")
print(f"  最小间距: {dist.min():.4f}λ  (dmin={dmin})  {'OK' if dist.min()>=dmin-1e-10 else 'FAIL'}")

# ── 画图 ──
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 位置图
ax = axes[0]; x0, x1 = mapper.x0, mapper.x1; y0, y1 = mapper.y0, mapper.y1
for xi in np.arange(x0, x1+dmin/2, dmin):
    ax.axvline(xi, color='gray', lw=0.3, alpha=0.4)
for yi in np.arange(y0, y1+dmin/2, dmin):
    ax.axhline(yi, color='gray', lw=0.3, alpha=0.4)
ax.add_patch(plt.Rectangle((x0-0.5*dmin, y0-0.5*dmin), x1-x0+dmin, y1-y0+dmin,
                            fill=False, edgecolor='black', lw=1.2))
gx = np.arange(x0, x1+dmin/2, dmin); gy = np.arange(y0, y1+dmin/2, dmin)
GX, GY = np.meshgrid(gx, gy)
ax.scatter(GX.ravel(), GY.ravel(), s=8, c='lightgray', marker='.', zorder=1)
qx, qy = mapper.synthesize(result["x"], expand_to_4q=False)
r = dmin/2
for xi, yi in zip(qx, qy):
    ax.add_patch(plt.Circle((xi, yi), r, color='royalblue', alpha=0.8, zorder=2))
ax.set_xlim(x0-0.7*dmin, x1+0.7*dmin); ax.set_ylim(y0-0.7*dmin, y1+0.7*dmin)
ax.set_xlabel("x (λ)"); ax.set_ylabel("y (λ)")
ax.set_title(f"第一象限 {mapper.Nq}元  dmin={dmin}λ"); ax.set_aspect('equal')

# 方向图
for ax, pos, label in [(axes[1], fx, "phi=0 (x)"), (axes[2], fy, "phi=90 (y)")]:
    af_db = Pattern.to_dB(pat.linear_af(pos))
    psll, ang = get_psll(af_db, theta)
    ax.plot(theta, af_db)
    ax.axhline(psll, color='r', ls='--', lw=0.8, label=f'PSLL={psll:.1f}dB')
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("dB")
    ax.set_ylim(-60, 3); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(label)

fig.suptitle(f"MGOM稀布面阵 {Ne}元  Lx={Lx}λ Ly={Ly}λ  PSLL和={psll0+psll90:.1f}dB", fontsize=13)
fig.tight_layout()
fig.savefig("mgom_opt.png", dpi=150); plt.close(fig)
print("图片已保存: mgom_opt.png")
