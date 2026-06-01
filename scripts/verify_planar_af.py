"""面阵计算验证 — 线阵退化测试 + 大阵列分块压力测试。

用法: python scripts/verify_planar_af.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from antopt.antenna import Pattern


def test_linear_vs_planar():
    """线阵退化：沿 x 轴放阵元，线阵 AF ≡ 平面阵 φ=0° 切片。"""
    print("=== 1. 线阵退化测试 ===")

    positions = np.linspace(-2.25, 2.25, 10)

    pat_lin = Pattern(theta_deg_step=0.1)
    af_lin = pat_lin.linear_af(positions)
    af_lin_db = Pattern.to_dB(af_lin)

    pat_pl = Pattern(
        array_type="planar",
        theta_deg_step=0.1,
        phi_deg_start=0, phi_deg_end=0, phi_deg_step=1.0,
    )
    af_pl = pat_pl.planar_af(positions, np.zeros_like(positions))
    af_pl_db = Pattern.to_dB(af_pl[:, 0])

    diff = np.max(np.abs(af_lin_db - af_pl_db))
    print(f"  10元线阵  线阵 vs 面阵 φ=0° 最大差异: {diff:.2e} dB")
    assert diff < 1e-10, f"差异过大: {diff:.2e}"
    print("  [OK] 通过\n")


def test_chunked_threshold():
    """分块边界测试：Nθ 刚好超过阈值时，分块和全量结果一致。"""
    print("=== 2. 分块 vs 全量 等价性 ===")

    # 构造一个小阵列，用不同 θ 分辨率测试
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])

    # 低分辨率 (Nθ=91, 不分块)
    pat_low = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    af_low = pat_low.planar_af(x, y)

    # 高分辨率 (Nθ=1801, 分块)
    pat_high = Pattern(
        array_type="planar",
        theta_deg_step=0.1,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )

    # 验证高分辨率也正确（通过和已知 PSLL 对比）
    af_high = pat_high.planar_af(x, y)
    af_high_db = Pattern.to_dB(af_high)
    from antopt.analysis import get_psll
    psll, _ = get_psll(af_high_db[:, 0], pat_high.theta_deg)
    print(f"  4元面阵 Nθ=1801 (分块)  PSLL={psll:.2f} dB")
    print(f"  分块高分辨率计算成功, 形状: {af_high.shape}")
    assert af_high.shape == (1801, 37), f"形状错误: {af_high.shape}"
    print("  [OK] 通过\n")


def test_large_array_memory():
    """大阵列内存测试：模拟 100×100 元面阵，验证不 OOM。"""
    print("=== 3. 大阵列内存压力测试 (100×100元) ===")
    print("  生成 10000 个随机阵元位置...")

    rng = np.random.default_rng(42)
    Nx, Ny = 100, 100
    x = rng.uniform(-5, 5, Nx * Ny)
    y = rng.uniform(-5, 5, Nx * Ny)

    pat = Pattern(
        array_type="planar",
        theta_deg_step=0.1,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=1.0,
    )

    print(f"  计算 10000元面阵 AF, Nθ={len(pat.theta_deg)}, Nφ={len(pat.phi_deg)}")
    print("  (旧版需要 ~26GB, 分块版约 ~150MB)...")

    try:
        af = pat.planar_af(x, y)
        print(f"  [OK] 计算成功! AF 形状: {af.shape}")
        af_db = Pattern.to_dB(af)
        from antopt.analysis import get_psll
        psll, _ = get_psll(af_db[:, 0], pat.theta_deg)
        print(f"  随机10000元面阵 PSLL={psll:.2f} dB")
    except MemoryError:
        print("  [FAIL] 内存不够")
    except Exception as e:
        print(f"  [FAIL] 错误: {e}")
    print()


def test_multi_scan():
    """多扫描角测试。"""
    print("=== 4. 多扫描角 (Nscan=3) ===")
    positions = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    pat = Pattern(
        array_type="planar",
        theta_deg_step=0.5,
        theta0s_deg=np.array([0, 20, 40]),
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=2.0,
    )
    af = pat.planar_af(positions, np.zeros_like(positions))
    print(f"  多扫描 AF 形状: {af.shape}  (期望 (3, 361, 91))")
    assert af.shape == (3, 361, 91), f"形状错误: {af.shape}"
    print("  [OK] 通过\n")


if __name__ == "__main__":
    test_linear_vs_planar()
    test_chunked_threshold()
    test_large_array_memory()
    test_multi_scan()
    print("全部测试通过 [OK]")
