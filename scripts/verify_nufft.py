"""FINUFFT vs 直接求和 — 随机稀布面阵精度对比"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, time
from antopt.antenna import Pattern

# ── 随机稀布面阵 ──
rng = np.random.default_rng(42)
N = 1000  # 1000 元随机非均匀面阵
x = rng.uniform(-15, 15, N)  # 波长单位
y = rng.uniform(-15, 15, N)
amps = rng.uniform(0.1, 1.0, N)
phases = rng.uniform(0, 2 * np.pi, N)

# ── 直接求和 ──
pat = Pattern(array_type="planar", theta_deg_step=0.5,
              phi_deg_start=-90, phi_deg_end=90, phi_deg_step=2.0)
theta = pat.theta_deg; phi = pat.phi_deg
Nt, Np = len(theta), len(phi)
print(f"{N}元随机面阵  Nθ={Nt} Nφ={Np}  孔径: x±{np.max(np.abs(x)):.0f}λ y±{np.max(np.abs(y)):.0f}λ")

t0 = time.perf_counter()
af_dir = pat.planar_af(x, y, amplitudes=amps, phases=phases)
t_dir = time.perf_counter() - t0
db_dir = Pattern.to_dB(af_dir)

# ── FINUFFT ──
import finufft as fu

# 缩放到 FINUFFT 坐标 [-π, π]
L = max(np.max(np.abs(x)), np.max(np.abs(y)))
scale = np.pi / L
x_scaled = x * scale
y_scaled = y * scale
c = (amps * np.exp(1j * phases)).astype(np.complex128)

# Nyquist: du = d(sinθ) ≈ dθ in radians → Nu = 2L/du
dtheta_rad = np.deg2rad(np.abs(theta[1] - theta[0]))
Nu = int(np.ceil(2 * L * np.pi / dtheta_rad))
Nv = Nu
Nu = max(Nu, 512); Nv = max(Nv, 512)
print(f"  FINUFFT 网格: {Nu}x{Nv}")

t1 = time.perf_counter()
af_grid = fu.nufft2d1(x_scaled, y_scaled, c, (Nu, Nv), eps=1e-6, isign=1)
t_nufft = time.perf_counter() - t1

# 从均匀 (u,v) 插值到 (θ,φ)
from scipy.interpolate import RegularGridInterpolator
u_axis = np.linspace(-np.pi, np.pi, Nu, endpoint=False)
v_axis = np.linspace(-np.pi, np.pi, Nv, endpoint=False)
interp = RegularGridInterpolator((u_axis, v_axis), af_grid,
                                  bounds_error=False, fill_value=0.0)
TH, PH = np.meshgrid(theta, phi, indexing='ij')
u_tgt = np.sin(np.deg2rad(TH)) * np.cos(np.deg2rad(PH)) * scale
v_tgt = np.sin(np.deg2rad(TH)) * np.sin(np.deg2rad(PH)) * scale
t2 = time.perf_counter()
af_nufft = interp(np.stack([u_tgt.ravel(), v_tgt.ravel()], axis=1)).reshape(Nt, Np)
t_interp = time.perf_counter() - t2
db_nufft = Pattern.to_dB(af_nufft)

# ── 对比 ──
diff = db_dir - db_nufft
mask = db_dir > -50  # 有意义的区域
print(f"\n  耗时:  直接求和 {t_dir:.2f}s  FINUFFT {t_nufft:.2f}s + 插值 {t_interp:.2f}s")
print(f"  误差:  max={np.max(np.abs(diff)):.3f}dB  RMS={np.sqrt(np.mean(diff**2)):.3f}dB")
print(f"  >-50dB区域: max={np.max(np.abs(diff[mask])):.3f}dB  RMS={np.sqrt(np.mean(diff[mask]**2)):.3f}dB\n")

# ── phi=0° 切面对比 ──
phi0 = int(np.argmin(np.abs(phi - 0)))
from antopt.analysis import get_psll
psll_d, a1 = get_psll(db_dir[:, phi0], theta)
psll_f, a2 = get_psll(db_nufft[:, phi0], theta)
print(f"  phi=0 PSLL: 直接={psll_d:.2f}dB  FINUFFT={psll_f:.2f}dB  差={psll_d-psll_f:.3f}dB")
