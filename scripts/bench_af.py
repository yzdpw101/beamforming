"""阵因子计算全面基准测试 — 线阵/面阵/均匀/非均匀/多种群"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, time
import finufft as fu
from antopt.antenna import Pattern

rng = np.random.default_rng(42)


def bench_once(label, x, y, c, theta_step, phi_step, n_jobs=1):
    """单次计算：直接 vs NUFFT"""
    pat = Pattern(array_type="planar",
                  theta_deg_step=theta_step,
                  phi_deg_start=-90, phi_deg_end=90, phi_deg_step=phi_step)
    TH, PH = np.meshgrid(pat.theta_deg, pat.phi_deg, indexing="ij")
    u = np.sin(np.deg2rad(TH)) * np.cos(np.deg2rad(PH))
    v = np.sin(np.deg2rad(TH)) * np.sin(np.deg2rad(PH))

    # 直接求和
    t0 = time.perf_counter()
    af_dir = pat.planar_af(x, y, amplitudes=np.abs(c), phases=np.angle(c), n_jobs=n_jobs)
    td = time.perf_counter() - t0
    db_dir = Pattern.to_dB(af_dir)

    # NUFFT Type3
    L = max(np.max(np.abs(x)), np.max(np.abs(y))) if len(x) > 0 else 1.0
    xs, ys = x * (np.pi / L), y * (np.pi / L)
    ss, ts = (2 * L * u).ravel(), (2 * L * v).ravel()
    t1 = time.perf_counter()
    af_nu = fu.nufft2d3(xs, ys, c, ss, ts, eps=1e-6, isign=1).reshape(u.shape)
    tn = time.perf_counter() - t1
    db_nu = Pattern.to_dB(af_nu)

    # 误差
    diff = db_dir - db_nu
    mask = db_dir > -50
    err_max = float(np.max(np.abs(diff[mask]))) if np.any(mask) else 0.0

    return td, tn, err_max


def bench_population(label, x, y, c_template, pop_size, theta_step, phi_step):
    """多种群模拟：固定位置，只变激励"""
    pat = Pattern(array_type="planar",
                  theta_deg_step=theta_step,
                  phi_deg_start=-90, phi_deg_end=90, phi_deg_step=phi_step)
    TH, PH = np.meshgrid(pat.theta_deg, pat.phi_deg, indexing="ij")
    u = np.sin(np.deg2rad(TH)) * np.cos(np.deg2rad(PH))
    v = np.sin(np.deg2rad(TH)) * np.sin(np.deg2rad(PH))
    L = max(np.max(np.abs(x)), np.max(np.abs(y)))
    xs, ys = x * (np.pi / L), y * (np.pi / L)
    ss, ts = (2 * L * u).ravel(), (2 * L * v).ravel()

    # 生成 population 组随机激励
    amps = rng.uniform(0.1, 1.0, (pop_size, len(x)))
    phases = rng.uniform(0, 2 * np.pi, (pop_size, len(x)))
    cs = (amps * np.exp(1j * phases)).astype(np.complex128)

    # 直接求和 (全部种群)
    t0 = time.perf_counter()
    for i in range(pop_size):
        pat.planar_af(x, y, amplitudes=amps[i], phases=phases[i])
    td = time.perf_counter() - t0

    # NUFFT (全部种群)
    t1 = time.perf_counter()
    for i in range(pop_size):
        fu.nufft2d3(xs, ys, cs[i], ss, ts, eps=1e-6, isign=1)
    tn = time.perf_counter() - t1

    return td, tn


# ═══════════════════════════════════════════════════
#  测试 1: 单次计算 — 不同阵列规模
# ═══════════════════════════════════════════════════
print("=" * 72)
print(f"{'配置':<20} {'阵元':>6} {'直接':>7} {'NUFFT':>7} {'加速':>6} {'误差':>7}")
print("-" * 72)

configs = [
    ("线阵 均匀",    np.linspace(-5, 5, 500), np.zeros(500)),
    ("线阵 稀布",    rng.uniform(-5, 5, 500), np.zeros(500)),
    ("面阵 均匀 32x32", *(lambda N=32: (np.tile(np.arange(N)-(N-1)/2, N)*0.5, np.repeat(np.arange(N)-(N-1)/2, N)*0.5))()),
    ("面阵 稀布 1000", rng.uniform(-10, 10, 1000), rng.uniform(-10, 10, 1000)),
    ("面阵 稀布 5000", rng.uniform(-15, 15, 5000), rng.uniform(-15, 15, 5000)),
    ("面阵 稀布 10000", rng.uniform(-15, 15, 10000), rng.uniform(-15, 15, 10000)),
]

for name, x_arr, y_arr in configs:
    N = len(x_arr)
    c = (rng.uniform(0.1, 1.0, N) * np.exp(1j * rng.uniform(0, 2*np.pi, N))).astype(np.complex128)
    td, tn, err = bench_once(name, x_arr, y_arr, c, 1.0, 2.0)
    spd = f"{td/tn:.0f}x" if tn > 0.001 else "∞"
    print(f"{name:<20} {N:>6} {td:>6.2f}s {tn:>6.3f}s {spd:>6} {err:>6.3f}dB")

# ═══════════════════════════════════════════════════
#  测试 2: 多种群 — 模拟 CMA-ES 迭代
# ═══════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"{'种群模拟':<20} {'阵元':>6} {'pop':>5} {'直接':>8} {'NUFFT':>8} {'加速':>6}")
print("-" * 72)

pop_configs = [
    ("线阵 稀布 500元", rng.uniform(-5, 5, 500), np.zeros(500), 100),
    ("面阵 稀布 1000元", rng.uniform(-10, 10, 1000), rng.uniform(-10, 10, 1000), 100),
    ("面阵 稀布 5000元", rng.uniform(-12, 12, 5000), rng.uniform(-12, 12, 5000), 50),
]

for name, x_arr, y_arr, pop in pop_configs:
    N = len(x_arr)
    c_temp = (rng.uniform(0.1, 1.0, N) * np.exp(1j * rng.uniform(0, 2*np.pi, N))).astype(np.complex128)
    td, tn = bench_population(name, x_arr, y_arr, c_temp, pop, 1.0, 2.0)
    spd = f"{td/tn:.0f}x"
    print(f"{name:<20} {N:>6} {pop:>5} {td:>7.1f}s {tn:>7.1f}s {spd:>6}")

print("=" * 72)
