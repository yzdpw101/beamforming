"""线阵波束显示 — 单张 2D 切面图，全标注叠加。

用法:
  修改下面参数 → python plot_linear_beam.py
  python plot_linear_beam.py <result_dir> --freqs 9.8 --theta0s 0 --ep <dir>
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from antopt.antenna import Pattern, ElementPattern
from antopt.analysis import PatternPlot, get_psll
from antopt.io import load_elements
from antopt import fmt_freq_angle

# ═══════════════════════════════════════════════════
#  默认参数 — 改这里直接跑: python plot_linear_beam.py
# ═══════════════════════════════════════════════════

RESULT_DIR  = r"E:\\Projects\\Opencode\\beamforming2\\results\\linear_sidelobe\\20260601_165559_9.8GHz_seed1671766257"            # elements.txt 所在目录
FREQS_GHZ   = [9.8]         # 频率 GHz
THETA0S_DEG = [0]           # 波束指向 °
THETA_RANGE = (-90, 90, 0.1)

EP_DIR      = "E:/Projects/Opencode/beamforming2/input/element_pattern/kty/phi0"  # 单元方向图目录 (空=纯阵因子)
EP_IN_DB    = True
EP_IS_GAIN  = True
EP_THETA_RNG = (-90, 90)  # EP CSV 的 theta 覆盖范围
AMP_IN_DB   = False

SAVE_DIR    = "default"
INTERACTIVE = False
DB_FLOOR    = -60


def _read_arr(result_dir):
    arr = load_elements(result_dir)
    amps = arr[:, 2]
    if AMP_IN_DB: amps = 10.0 ** (-amps / 20.0)
    return arr[:, 0], amps, arr[:, 3]


def plot_linear_beam(result_dir=RESULT_DIR,
                     freqs_ghz=None, theta0s_deg=None,
                     theta_range=None,
                     ep_dir=EP_DIR, ep_in_db=EP_IN_DB, ep_is_gain=EP_IS_GAIN,
                     ep_theta_range=EP_THETA_RNG,
                     save_dir=SAVE_DIR, interactive=INTERACTIVE):
    freqs   = np.atleast_1d(np.array(freqs_ghz or FREQS_GHZ, dtype=float))
    theta0s = np.atleast_1d(np.array(theta0s_deg or THETA0S_DEG, dtype=float))
    thr     = theta_range or THETA_RANGE

    result_dir = Path(result_dir)
    if result_dir.is_file(): result_dir = result_dir.parent

    pos_m, amps, phases_deg = _read_arr(result_dir)
    lam = 299792458.0 / (freqs[0] * 1e9)
    pos_wl = pos_m / lam
    Ne = len(pos_wl)

    print(f"{Ne}元  f={freqs}GHz  theta0={theta0s}°")
    print(f"  孔径: {np.ptp(pos_m)*1000:.1f}mm  λ={lam*1000:.2f}mm")

    pat = Pattern(theta_deg_start=thr[0], theta_deg_end=thr[1], theta_deg_step=thr[2],
                  theta0s_deg=theta0s, frequenciesGHz=freqs)
    theta = pat.theta_deg

    # EP
    ep = None
    if ep_dir and Path(str(ep_dir)).exists():
        ep = ElementPattern.from_hfss_multi_freq(
            str(ep_dir), freqs,
            (float(theta[0]), float(theta[-1]), abs(float(theta[1]-theta[0]))),
            is_gain=ep_is_gain, in_dB=ep_in_db, input_theta_range=ep_theta_range)
        print(f"  EP: {ep_dir}")

    af = pat.af(pos_wl, amplitudes=amps, phases=np.deg2rad(phases_deg))
    if ep: af = af * ep[0]
    af_db = Pattern.to_dB(af)

    # 保存
    sv = Path(save_dir) if save_dir and save_dir != "default" else result_dir / "figures"
    sv.mkdir(parents=True, exist_ok=True)

    af_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]
    for sub in range(af_flat.shape[0]):
        si, fi = sub % len(theta0s), sub // len(theta0s)
        f_ghz, t0 = freqs[fi], theta0s[si]
        db = af_flat[sub]
        psll_v, _ = get_psll(db, theta)

        title = f"{Ne}元 f={f_ghz}GHz $\\theta_0$={t0}°  PSLL={psll_v:.1f}dB"
        pp = PatternPlot(theta, db, title=title, db_floor=DB_FLOOR)
        pp.add_psll().add_pointing(float(t0)).add_hpbw()

        d = sv / fmt_freq_angle(f_ghz, t0); d.mkdir(parents=True, exist_ok=True)
        pp.save(str(d / "pattern_2d.png"))
        print(f"  {fmt_freq_angle(f_ghz, t0)} PSLL={psll_v:.1f}dB")

    print(f"  已保存 {sv}")
    if interactive: plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="线阵波束显示")
    p.add_argument("result_dir", nargs="?", default=RESULT_DIR)
    p.add_argument("--freqs", type=str, default=None)
    p.add_argument("--theta0s", type=str, default=None)
    p.add_argument("--ep", type=str, default=None, dest="ep_dir")
    p.add_argument("--save", type=str, default=SAVE_DIR, dest="save_dir")
    p.add_argument("--no-show", action="store_false", default=INTERACTIVE, dest="interactive")
    args = p.parse_args()

    kwargs = dict(
        result_dir=args.result_dir,
        ep_dir=args.ep_dir if args.ep_dir is not None else EP_DIR,
        save_dir=args.save_dir, interactive=args.interactive,
    )
    if args.freqs:   kwargs["freqs_ghz"]   = [float(x.strip()) for x in args.freqs.split(",")]
    if args.theta0s: kwargs["theta0s_deg"] = [float(x.strip()) for x in args.theta0s.split(",")]
    plot_linear_beam(**kwargs)
