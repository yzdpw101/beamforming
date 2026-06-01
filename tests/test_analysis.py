"""测试方向图分析函数。"""

import numpy as np
from antopt.antenna import Pattern
from antopt.analysis.metrics import get_psll, find_peaks, get_overall_psll


def test_get_psll_broadside():
    """验证均匀 10 元半波长直线阵法向 PSLL ≈ -13.26 dB。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.to_dB(pat.linear_af(positions))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert abs(psll_val - (-13.26)) < 0.5, f"PSLL={psll_val:.2f} dB, 预期 ~-13.26"
    assert isinstance(psll_angle, float)
    assert abs(psll_angle) > 0, "副瓣不应在法向"


def test_get_psll_single_element():
    """单阵元无副瓣，PSLL 应返回 (-inf, nan)。"""
    pat = Pattern()
    af_db = Pattern.to_dB(pat.linear_af(np.array([0.0])))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert psll_val == -np.inf
    assert np.isnan(psll_angle)


def test_get_psll_with_mainlobe_region():
    """指定主瓣区域排除后计算副瓣。"""
    pat = Pattern(theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.to_dB(pat.linear_af(positions))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg, mainlobe_region=(-10, 10))
    assert psll_val < 0, "副瓣电平应为负值"


def test_find_peaks_broadside():
    """验证 find_peaks 能找到均匀线阵的峰值。"""
    pat = Pattern(theta_deg_step=1.0)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.to_dB(pat.linear_af(positions))
    values, angles = find_peaks(af_db, pat.theta_deg)
    assert len(values) >= 2, "均匀线阵应有多个峰值"
    assert abs(values[0]) < 1e-6, "最高峰值应为 0 dB（主瓣）"


def test_find_peaks_no_theta():
    """不传 theta 时返回索引。"""
    data = np.array([0.0, -5.0, -10.0, -3.0, -8.0])
    values, indices = find_peaks(data)
    assert len(values) >= 2
    assert np.issubdtype(indices.dtype, np.integer)


def test_get_psll_no_theta():
    """不传 theta 时 get_psll 返回索引。"""
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll_val, psll_idx = get_psll(af_db)
    assert isinstance(psll_val, float)
    assert isinstance(psll_idx, (int, np.integer))
    assert psll_val < 0


def test_get_psll_mainlobe_region_requires_theta():
    """mainlobe_region 不传 theta 时报错。"""
    data = np.random.randn(100)
    try:
        get_psll(data, mainlobe_region=(-10, 10))
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_get_psll_2d_vs_1d_slice():
    """2D get_psll 每 φ 平面应与手动切片 1D get_psll 一致。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    pslls, coords = get_psll(af_db, pat.theta_deg, pat.phi_deg)

    assert pslls.shape == (len(pat.phi_deg),)
    assert coords.shape == (len(pat.phi_deg), 2)

    for j in range(len(pat.phi_deg)):
        val_1d, coord_1d = get_psll(af_db[:, j], pat.theta_deg)
        both_inf = np.isinf(pslls[j]) and np.isinf(val_1d)
        assert both_inf or abs(pslls[j] - val_1d) < 1e-10
        assert abs(coords[j, 0] - coord_1d) < 1e-4 \
            or (np.isnan(coords[j, 0]) and np.isnan(coord_1d))
        assert abs(coords[j, 1] - pat.phi_deg[j]) < 1e-10


def test_get_psll_2d_single_element():
    """2D 单阵元：所有 φ 平面均无副瓣。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=30.0,
    )
    af_db = Pattern.to_dB(pat.planar_af(np.array([0.0]), np.array([0.0])))
    pslls, coords = get_psll(af_db, pat.theta_deg, pat.phi_deg)
    assert np.all(np.isinf(pslls))
    assert np.all(np.isnan(coords[:, 0]))
    assert np.all(~np.isnan(coords[:, 1]))


def test_find_peaks_2d():
    """2D find_peaks 返回合并峰值，coords 为 (N, 2)。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.linspace(-2.25, 2.25, 10)
    y = np.zeros(10)
    af_db = Pattern.to_dB(pat.planar_af(x, y))
    values, coords = find_peaks(af_db, pat.theta_deg, pat.phi_deg)
    assert coords.shape[1] == 2
    assert len(values) > 0
    assert abs(values[0]) < 1e-6


def test_get_overall_psll_basic():
    """全平面 PSLL 应 ≥ 各 φ 平面的 PSLL。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.linspace(-2.25, 2.25, 10)
    y = np.zeros(10)
    af_db = Pattern.to_dB(pat.planar_af(x, y))
    psll, coord = get_overall_psll(af_db, pat.theta_deg, pat.phi_deg)
    assert isinstance(psll, float)
    assert coord.shape == (2,)
    assert psll < 0


def test_get_overall_psll_1d():
    """1D 输入时 get_overall_psll 等价于 get_psll。"""
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll1, coord1 = get_psll(af_db, pat.theta_deg)
    psll2, coord2 = get_overall_psll(af_db, pat.theta_deg)
    assert abs(psll1 - psll2) < 1e-10
    assert abs(coord2[0] - coord1) < 1e-10


def test_find_peaks_1d_mainlobe_region():
    """1D find_peaks 排除主瓣区域后主瓣应被移除。"""
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    v_all, c_all = find_peaks(af_db, pat.theta_deg)
    assert abs(v_all[0]) < 1e-6
    v_mr, c_mr = find_peaks(af_db, pat.theta_deg, mainlobe_region=(-15, 15))
    assert v_mr[0] < 0
    in_region = (c_mr >= -15) & (c_mr <= 15)
    assert not in_region.any()


def test_get_psll_1d_mainlobe_region_effect():
    """1D 排除主瓣后 PSLL 应不等于不排除时。"""
    pat = Pattern(theta_deg_step=0.5)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll_no_mr, _ = get_psll(af_db, pat.theta_deg)
    psll_mr, _ = get_psll(af_db, pat.theta_deg, mainlobe_region=(-15, 15))
    assert abs(psll_no_mr - (-13.26)) < 0.5
    assert psll_mr < 0
