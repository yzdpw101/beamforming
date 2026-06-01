"""平面阵波束显示 — 3D UV 方向图 + 每 φ 切面子图序列 + 位置图。

既可直接运行，也可作为模块调用:
  python plot_planar_beam.py
  python plot_planar_beam.py <result_dir> --phi-cuts 0,45,90

  from examples.plot_planar_beam import plot_planar_beam
  plot_planar_beam("results/some_run/", phi_cuts=[0, 45])
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
from antopt.analysis import get_psll, get_overall_psll, PatternPlot, plot_3d_uv, plot_3d_spherical, plot_positions_2d
from antopt.io import load_elements
from antopt import fmt_freq_angle

# ═══════════════════════════════════════════════════
#  默认参数
# ═══════════════════════════════════════════════════

RESULT_DIR    = ""  # elements.txt 所在目录
FREQS_GHZ     = [10.6, 11.4]
THETA0S_DEG   = [0, 10]
PHI0S_DEG     = [0, 10]
THETA_RANGE   = [-90.0, 90.0, 0.1]  # [start, end, step]
PHI_RANGE     = [0.0, 180.0, 1.0]    # [start, end, step]
PHI_CUTS      = [0]

NULL_ANGLES_DEG       = []
NULL_WINDOW_HALF_DEG  = []

SHOW_PSLL      = True
SHOW_NULL      = True
SHOW_DIFF_BEAM = False
SHOW_POINTING  = True
SHOW_HPBW      = True

EP_DIR         = None
EP_IN_DB       = True; EP_IS_GAIN = True; EP_THETA_RANGE = (-90, 90)
AMP_IN_DB      = False

SAVE_DIR       = "default"
INTERACTIVE    = True
DB_FLOOR       = -60

# ═══════════════════════════════════════════════════


def _resolve_save_dir(save_dir, result_dir):
    if save_dir is False or save_dir is None: return None
    if save_dir == "default": return Path(result_dir) / "figures"
    return Path(save_dir)


def _load_result(result_dir, amp_in_db=False):
    """从 elements.txt 加载结果。"""
    arr = load_elements(result_dir)
    pos_x, pos_y = arr[:, 0], arr[:, 1]
    amps = arr[:, 2]
    if amp_in_db: amps = 10.0 ** (-amps / 20.0)
    return pos_x, pos_y, amps, arr[:, 3]


def _null_windows(null_angles, window_half_list):
    if not null_angles: return []
    if isinstance(window_half_list, (int, float)):
        window_half_list = [window_half_list] * len(null_angles)
    while len(window_half_list) < len(null_angles):
        window_half_list.append(window_half_list[-1])
    return list(zip(null_angles, window_half_list[:len(null_angles)]))


def plot_planar_beam(result_dir=RESULT_DIR,
                     freqs_ghz=None, theta0s_deg=None, phi0s_deg=None,
                     theta_range=None, phi_range=None,
                     phi_cuts=None,
                     null_angles_deg=None, null_window_half_deg=None,
                     show_psll=SHOW_PSLL, show_null=SHOW_NULL,
                     show_diff_beam=SHOW_DIFF_BEAM, show_pointing=SHOW_POINTING,
                     show_hpbw=SHOW_HPBW,
                     ep_dir=EP_DIR, ep_in_db=EP_IN_DB, ep_is_gain=EP_IS_GAIN,
                     ep_theta_range=EP_THETA_RANGE,
                     amp_in_db=AMP_IN_DB,
                     save_dir=SAVE_DIR, interactive=INTERACTIVE):
    if freqs_ghz is None: freqs_ghz = FREQS_GHZ
    if theta0s_deg is None: theta0s_deg = THETA0S_DEG
    if phi0s_deg is None: phi0s_deg = PHI0S_DEG
    if theta_range is None: theta_range = THETA_RANGE
    if phi_range is None: phi_range = PHI_RANGE
    if phi_cuts is None: phi_cuts = PHI_CUTS
    if null_angles_deg is None: null_angles_deg = NULL_ANGLES_DEG
    if null_window_half_deg is None: null_window_half_deg = NULL_WINDOW_HALF_DEG
    theta_start, theta_end, theta_step = theta_range
    phi_start, phi_end, phi_step = phi_range
    null_windows = _null_windows(null_angles_deg, null_window_half_deg)

    result_dir = Path(result_dir)
    if result_dir.is_file():
        result_dir = result_dir.parent
    if not result_dir.exists():
        raise FileNotFoundError(f"结果目录不存在: {result_dir}")
    save_path = _resolve_save_dir(save_dir, result_dir)
    if save_path: save_path.mkdir(parents=True, exist_ok=True)

    pos_x_m, pos_y_m, amps, phases_deg = _load_result(result_dir, amp_in_db=amp_in_db)
    lam = 299792458.0 / (freqs_ghz[0] * 1e9)
    pos_x_wl = pos_x_m / lam; pos_y_wl = pos_y_m / lam
    Ne = len(pos_x_wl); n_freq = len(freqs_ghz); n_scan = len(theta0s_deg)

    print(f"加载 {Ne} 元平面阵, f={freqs_ghz}GHz")
    print(f"孔径: x={np.ptp(pos_x_m)*1000:.1f}mm  y={np.ptp(pos_y_m)*1000:.1f}mm")

    # ── Pattern + AF ──
    pat = Pattern(
        array_type="planar",
        theta_deg_start=theta_start, theta_deg_end=theta_end, theta_deg_step=theta_step,
        theta0s_deg=np.array(theta0s_deg, dtype=float),
        phi_deg_start=phi_start, phi_deg_end=phi_end, phi_deg_step=phi_step,
        phi0s_deg=np.array(phi0s_deg, dtype=float),
        frequenciesGHz=np.array(freqs_ghz, dtype=float),
    )
    theta = pat.theta_deg; phi = pat.phi_deg

    ep_data = None
    if ep_dir and Path(str(ep_dir)).exists():
        ep_data = ElementPattern.from_hfss_multi_freq(
            str(ep_dir), np.array(freqs_ghz, dtype=float),
            (float(theta[0]), float(theta[-1]), abs(float(theta[1]-theta[0]))),
            is_gain=ep_is_gain, in_dB=ep_in_db, input_theta_range=ep_theta_range)

    af_all = pat.planar_af(pos_x_wl, pos_y_wl, amplitudes=amps,
                           phases=np.deg2rad(phases_deg))
    af_abs = np.abs(af_all)
    if af_abs.ndim == 2: af_abs = af_abs[np.newaxis, np.newaxis, :, :]
    elif af_abs.ndim == 3: af_abs = af_abs.reshape(n_freq, n_scan, af_abs.shape[-2], af_abs.shape[-1])

    # ── 3D 图 ──
    for fi, f_ghz in enumerate(freqs_ghz):
        for si in range(n_scan):
            t0 = theta0s_deg[si] if si < len(theta0s_deg) else theta0s_deg[0]
            p0 = phi0s_deg[si] if si < len(phi0s_deg) else phi0s_deg[0]
            db3d = 20.0 * np.log10(np.maximum(af_abs[fi, si], 1e-30) / np.max(af_abs[fi, si]))
            psll_3d, coord_3d = get_overall_psll(db3d, theta, phi)
            print(f"  {fmt_freq_angle(f_ghz, t0, p0)} 全空间 PSLL: {psll_3d:.2f} dB @ "
                  f"θ={coord_3d[0]:.1f}° φ={coord_3d[1]:.1f}°")

            sub3d = save_path / fmt_freq_angle(f_ghz, t0, p0) if save_path else None
            if sub3d: sub3d.mkdir(parents=True, exist_ok=True)
            t3d = f"{Ne}元 f={f_ghz}GHz $\\theta_0$={t0}° $\\phi_0$={p0}° PSLL={psll_3d:.1f}dB"
            plot_3d_spherical(theta, phi, db3d, db_floor=DB_FLOOR, title=t3d,
                              save_path=str(sub3d / "3d_sph.png") if sub3d else None)
            plot_3d_uv(theta, phi, db3d, db_floor=DB_FLOOR, title=t3d,
                       save_path=str(sub3d / "3d_uv.png") if sub3d else None)

    # ── 面板清单 ──
    panel_specs = [("原始方向图", None)]
    if show_psll:      panel_specs.append(("PSLL 标记", "psll"))
    if show_null:      panel_specs.append(("零陷 标记", "null"))
    if show_diff_beam: panel_specs.append(("差波束 标记", "diff"))
    if show_pointing:  panel_specs.append(("指向 标记", "pointing"))
    if show_hpbw:      panel_specs.append(("HPBW 标记", "hpbw"))
    panel_specs.append(("全部叠加", "all"))
    n_panels = len(panel_specs)

    # ── φ 切面子图序列 ──
    if phi_cuts:
        n_phi_cuts = len(phi_cuts)
        for phi_cut in phi_cuts:
            phi_idx = int(np.argmin(np.abs(phi - phi_cut)))
            phi_actual = phi[phi_idx]
            if abs(phi_actual - phi_cut) > 0.5:
                print(f"  φ={phi_cut}° → 取最近 φ={phi_actual:.1f}°")

            for fi, f_ghz in enumerate(freqs_ghz):
                for si in range(n_scan):
                    t0 = theta0s_deg[si] if si < len(theta0s_deg) else theta0s_deg[0]
                    p0 = phi0s_deg[si] if si < len(phi0s_deg) else phi0s_deg[0]
                    db = 20.0 * np.log10(np.maximum(af_abs[fi, si, :, phi_idx], 1e-30)
                                         / np.max(af_abs[fi, si, :, phi_idx]))
                    psll_v = get_psll(db, theta)[0]

                    ncols = (n_panels + 1) // 2
                    nrows = 2 if n_panels > 1 else 1
                    fig, axes = plt.subplots(nrows, ncols,
                                             figsize=(5 * ncols, 4.5 * nrows),
                                             squeeze=False)

                    for pi, (title, ptype) in enumerate(panel_specs):
                        r, c = pi // ncols, pi % ncols
                        ax = axes[r, c]
                        pp = PatternPlot(theta, db, ax=ax, title=title, db_floor=DB_FLOOR)
                        if ptype in ("psll", "all")   and show_psll: pp.add_psll()
                        if ptype in ("null", "all")   and show_null:
                            pp.add_null(null_angles_deg, null_window_half_deg)
                        if ptype in ("diff", "all")   and show_diff_beam: pp.add_diff_beam(theta0_deg=t0)
                        if ptype in ("pointing", "all") and show_pointing: pp.add_pointing(theta0_deg=t0)
                        if ptype in ("hpbw", "all")   and show_hpbw: pp.add_hpbw()

                    for pi in range(n_panels, nrows * ncols):
                        axes[pi // ncols, pi % ncols].set_visible(False)
                    for c in range(ncols): axes[-1, c].set_xlabel(r"$\theta$ (deg)")
                    for r in range(nrows): axes[r, 0].set_ylabel("dB")

                    fig.suptitle(f"{Ne}元面阵 f={f_ghz}GHz $\\theta_0$={t0}° $\\phi_0$={p0}° "
                                 f"φ={phi_actual:.0f}° PSLL={psll_v:.1f}dB",
                                 fontsize=13, fontweight='bold')
                    fig.tight_layout()

                    sub = save_path / f"{fmt_freq_angle(f_ghz, t0, p0)}_phi{phi_actual:.0f}" if save_path else None
                    if sub:
                        sub.mkdir(parents=True, exist_ok=True)
                        fig.savefig(sub / "pattern_2d.png", dpi=200, bbox_inches="tight")
                    if not interactive: plt.close(fig)

    # ── 位置图 ──
    if save_path:
        plot_positions_2d(pos_x_m, pos_y_m, amps, title=f"{Ne}元面阵 位置-幅度",
                          color_label="Amplitude",
                          save_path=str(save_path / "positions_amp.png"))
        plot_positions_2d(pos_x_m, pos_y_m, phases_deg, title=f"{Ne}元面阵 位置-相位",
                          color_label="Phase (deg)",
                          save_path=str(save_path / "positions_phase.png"))
        print(f"共 {n_freq * n_scan + 1} 张图 → {save_path}")

    if interactive and n_panels > 0: plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="平面阵波束显示")
    p.add_argument("result_dir", nargs="?", default=RESULT_DIR)
    p.add_argument("--freqs", type=str, default=None)
    p.add_argument("--theta0s", type=str, default=None)
    p.add_argument("--phi0s", type=str, default=None)
    p.add_argument("--phi-cuts", type=str, default=None)
    p.add_argument("--nulls", type=str, default=None)
    p.add_argument("--null-windows", type=str, default=None)
    p.add_argument("--ep", type=str, default=None, dest="ep_dir")
    p.add_argument("--save", type=str, default=SAVE_DIR, dest="save_dir")
    p.add_argument("--no-show", action="store_false", default=INTERACTIVE, dest="interactive")
    args = p.parse_args()

    plot_planar_beam(
        result_dir=args.result_dir,
        freqs_ghz=[float(f.strip()) for f in args.freqs.split(",")] if args.freqs else None,
        theta0s_deg=[float(t.strip()) for t in args.theta0s.split(",")] if args.theta0s else None,
        phi0s_deg=[float(p.strip()) for p in args.phi0s.split(",")] if args.phi0s else None,
        phi_cuts=[float(p.strip()) for p in args.phi_cuts.split(",")] if args.phi_cuts else None,
        null_angles_deg=[float(n.strip()) for n in args.nulls.split(",")] if args.nulls else None,
        null_window_half_deg=[float(w.strip()) for w in args.null_windows.split(",")] if args.null_windows else None,
        ep_dir=args.ep_dir or EP_DIR, save_dir=args.save_dir, interactive=args.interactive,
    )
