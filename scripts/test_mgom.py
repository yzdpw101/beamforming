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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, p_deg, p_label in [(ax1, 0, "phi=0"), (ax2, 90, "phi=90")]:
    pi = int(np.argmin(np.abs(phi - p_deg)))
    ax.plot(theta, af_db[:, pi])
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("dB")
    ax.set_ylim(-60, 3); ax.grid(alpha=0.3)
    psll, ang = get_psll(af_db[:, pi], theta)
    ax.axhline(psll, color='r', ls='--', lw=0.8, label=f'PSLL={psll:.1f}dB')
    ax.legend()
    ax.set_title(f"{Ne}元 MGOM Lx={Lx}λ Ly={Ly}λ  {p_label}")

fig.suptitle(f"MGOM 面阵方向图 (dmin={dmin}λ)", fontsize=13)
fig.tight_layout()
fig.savefig("mgom_test.png", dpi=150)
plt.close(fig)
print("\n图片已保存: mgom_test.png")
