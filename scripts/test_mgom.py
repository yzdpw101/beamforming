"""MGOM 面阵映射测试 — 108元对称阵列，φ=0° 和 φ=90° PSLL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from antopt.mapping import MGOM
from antopt.antenna import Pattern

# ── 配置 ──
Ne, Lx, Ly, dmin = 108, 9.5, 4.5, 0.5
mapper = MGOM(Ne=Ne, Lx=Lx, Ly=Ly, dmin=dmin, is_symmetric=True)
print(f"MGOM: {Ne}元  Nq={mapper.Nq}  grid={mapper.max_rows}x{mapper.max_cols}  n_vars={mapper.n_vars}")

# 随机变量 → 位置
rng = np.random.default_rng(42)
x0 = rng.uniform(0, 1, mapper.n_vars)
fx, fy = mapper.synthesize(x0, expand_to_4q=True)

# 验证 dmin
dist = np.sqrt((fx[:,None] - fx[None,:])**2 + (fy[:,None] - fy[None,:])**2)
np.fill_diagonal(dist, np.inf)
print(f"阵元间距: min={dist.min():.3f}λ (dmin={dmin})")

# ── 阵因子 ──
pat = Pattern(
    array_type="planar",
    theta_deg_step=0.1,
    phi_deg_start=0, phi_deg_end=180, phi_deg_step=1.0,
)
af = pat.planar_af(fx, fy)
af_db = Pattern.to_dB(af)
theta = pat.theta_deg; phi = pat.phi_deg

from antopt.analysis import get_psll

for p_label, p_deg in [("phi=0°", 0), ("phi=90°", 90)]:
    pi = int(np.argmin(np.abs(phi - p_deg)))
    psll, ang = get_psll(af_db[:, pi], theta)
    print(f"  {p_label}: PSLL={psll:.2f}dB @ theta={ang:.1f}°")

# ── 快速画图 ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 位置图 — 第一象限 + 网格
ax = axes[0]
x0, x1 = mapper.x0, mapper.x1
y0, y1 = mapper.y0, mapper.y1

for xi in np.arange(x0, x1 + dmin/2, dmin):
    ax.axvline(xi, color='gray', lw=0.3, alpha=0.4)
for yi in np.arange(y0, y1 + dmin/2, dmin):
    ax.axhline(yi, color='gray', lw=0.3, alpha=0.4)

ax.add_patch(plt.Rectangle((x0 - 0.5*dmin, y0 - 0.5*dmin),
                            x1 - x0 + dmin, y1 - y0 + dmin,
                            fill=False, edgecolor='black', lw=1.2))

gx = np.arange(x0, x1 + dmin/2, dmin)
gy = np.arange(y0, y1 + dmin/2, dmin)
GX, GY = np.meshgrid(gx, gy)
ax.scatter(GX.ravel(), GY.ravel(), s=8, c='lightgray', marker='.', zorder=1)

qx, qy = mapper.synthesize(rng.uniform(0, 1, mapper.n_vars), expand_to_4q=False)
r = dmin / 2
for xi, yi in zip(qx, qy):
    ax.add_patch(plt.Circle((xi, yi), r, color='royalblue', alpha=0.8, zorder=2))

ax.set_xlim(x0 - 0.7*dmin, x1 + 0.7*dmin)
ax.set_ylim(y0 - 0.7*dmin, y1 + 0.7*dmin)
ax.set_xlabel("x (λ)"); ax.set_ylabel("y (λ)")
ax.set_title(f"第一象限 {mapper.Nq}元  dmin={dmin}λ")
ax.set_aspect('equal')

for ax, p_deg, p_label in [(axes[1], 0, "phi=0"), (axes[2], 90, "phi=90")]:
    pi = int(np.argmin(np.abs(phi - p_deg)))
    ax.plot(theta, af_db[:, pi])
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("dB")
    ax.set_ylim(-60, 3); ax.grid(alpha=0.3)
    psll, ang = get_psll(af_db[:, pi], theta)
    ax.axhline(psll, color='r', ls='--', lw=0.8, label=f'PSLL={psll:.1f}dB')
    ax.legend()
    ax.set_title(f"{p_label}")

fig.suptitle(f"MGOM 面阵 {Ne}元  Lx={Lx}λ Ly={Ly}λ  dmin={dmin}λ", fontsize=13)
fig.tight_layout()
fig.savefig("mgom_test.png", dpi=150)
plt.close(fig)
print("\n图片已保存: mgom_test.png")
