"""ScenarioResult — 优化结果结构化存取。

输出格式:
  results/{example_name}/{YYYYMMDD_HHMMSS}_{tag}/
    ├── optResult.json       ← 精简：阵型 + 阵元数 + fitness + 组件指标
    ├── elements.txt          ← 合并：x y amplitudes phases
    ├── bestIndividual.txt    ← 优化器原始变量向量
    ├── figures/
    │   └── *.png
    └── patterns/
        └── *.csv
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np


# ═══════════════════════════════════════════════════
#  公共工具函数（画图脚本可独立 import）
# ═══════════════════════════════════════════════════

def load_elements(path) -> np.ndarray:
    """从 elements.txt 读取阵元数据 (N,4) [x_m, y_m, amplitude, phase_deg]。

    Args:
        path: elements.txt 路径，或包含该文件的目录

    Returns:
        np.ndarray, shape (N, 4), columns: x_m, y_m, amplitude, phase_deg
    """
    filepath = Path(path)
    if filepath.is_dir():
        filepath = filepath / "elements.txt"
    with open(filepath, "r") as f:
        lines = f.readlines()
    data = []
    for ln in lines:
        s = ln.strip()
        if s and s[0] in '-.0123456789':
            parts = s.split()
            if len(parts) >= 4:
                data.append([float(v) for v in parts])
    return np.array(data)


def print_component_summary(cr: dict):
    """打印组件指标到控制台（可独立调用，无需 ScenarioResult）。"""
    if "sidelobe" in cr:
        s = cr["sidelobe"]
        if s.get("psll_db") is not None:
            print(f"  PSLL: {s['psll_db']:.2f} dB @ theta={s.get('psll_angle_deg', '-')}°"
                  f"  (target {s.get('target_db', '-')} dB)")
    if "main_lobe_pointing" in cr:
        m = cr["main_lobe_pointing"]
        print(f"  主瓣指向: peak={m.get('peak_deg', '-')}°  target={m.get('target_deg', '-')}°")
    if "null_steering" in cr:
        for n in cr["null_steering"].get("nulls", []):
            print(f"  零陷 @ {n['angle_deg']}°: {n['depth_db']:.2f} dB  (target {n['target_db']} dB)")
    if "directivity" in cr:
        d = cr["directivity"]
        if d.get("dbi") is not None:
            print(f"  方向性: {d['dbi']:.2f} dBi  (target {d.get('target_dbi', '-')} dBi)")
    if "hpbw" in cr:
        h = cr["hpbw"]
        print(f"  HPBW: {h.get('width_deg', '-'):.1f}°  (target {h.get('target_deg', '-')}°)")
    if "difference_beam" in cr:
        db = cr["difference_beam"]
        print(f"  差波束: 零深={db.get('null_depth_db', '-'):.1f} dB  "
              f"对称度={db.get('symmetry_db', '-'):.2f} dB")


def save_result_figures(
    sr: "ScenarioResult",
    problem,
    decoded: dict,
    cfg,
    example_name: str,
    output_dir: Path,
    wavelength_m: float = 1.0,
):
    """生成全部图表（可独立调用，无需 sr 方法）。"""
    import csv
    from ..analysis.plotting import PatternPlot, plot_positions_1d
    from ..utils import fmt_freq_angle

    fig_root = Path(output_dir) / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    pos_wl = decoded["positions_wl"]
    amps = decoded["amplitudes"]
    phases_deg = np.rad2deg(decoded["phases_rad"])
    theta = problem.pattern.theta_deg
    freqs = problem.pattern.frequenciesGHz
    theta0s = problem.pattern.theta0s_deg
    Ne = len(pos_wl)
    tgt = cfg.target_components if cfg else {}
    cr = sr.component_results
    n_freq = len(freqs); n_scan = len(theta0s)

    pos_m = pos_wl * wavelength_m
    plot_positions_1d(pos_m, amps, title=f"{example_name} pos-amp",
                      color_label="Amplitude",
                      save_path=str(fig_root / "positions_amp.png"))
    plot_positions_1d(pos_m, phases_deg, title=f"{example_name} pos-phase",
                      color_label="Phase (deg)",
                      save_path=str(fig_root / "positions_phase.png"))

    af_db_all = decoded["af_db"]
    if af_db_all.ndim == 1:
        af_db_all = af_db_all.reshape(1, -1)

    for fi, f_ghz in enumerate(freqs):
        for si, t0_deg in enumerate(theta0s):
            if af_db_all.ndim == 3:
                db = af_db_all[fi, si]
            elif af_db_all.ndim == 2:
                db = af_db_all[fi * n_scan + si]
            else:
                db = af_db_all

            title_parts = [f"{example_name} {Ne}元 f={f_ghz}GHz $\\theta_0$={t0_deg}°"]
            if "sidelobe" in cr and cr["sidelobe"].get("psll_db") is not None:
                title_parts.append(f"PSLL={cr['sidelobe']['psll_db']:.1f}dB")
            if "directivity" in cr and cr["directivity"].get("dbi") is not None:
                title_parts.append(f"D={cr['directivity']['dbi']:.1f}dBi")
            if "hpbw" in cr and cr["hpbw"].get("width_deg") is not None:
                title_parts.append(f"HPBW={cr['hpbw']['width_deg']:.1f}°")
            title = " ".join(title_parts)

            sub = fig_root / fmt_freq_angle(f_ghz, t0_deg)
            sub.mkdir(parents=True, exist_ok=True)

            pp = PatternPlot(theta, db, title=title)
            if tgt:
                if "sidelobe" in tgt:
                    mlw = tgt["sidelobe"].get("mainlobeNullWidthDeg")
                    if mlw and mlw > 0:
                        pp.add_psll(mainlobe_region=(-float(mlw), float(mlw)))
                    else:
                        pp.add_psll()
                if "null_steering" in tgt:
                    nc = tgt["null_steering"]
                    pp.add_null(nc.get("anglesDeg", []), nc.get("windowHalfDeg"))
                if "main_lobe_pointing" in tgt:
                    pp.add_pointing(float(t0_deg))
                if "hpbw" in tgt:
                    pp.add_hpbw()
                if "difference_beam" in tgt:
                    pp.add_diff_beam(theta0_deg=float(t0_deg))
                if "directivity" in tgt:
                    pp.add_directivity(cr.get("directivity", {}).get("dbi"))
            else:
                pp.add_psll().add_hpbw()

            pp.save(str(sub / "pattern_2d.png"))

            csv_dir = output_dir / "patterns"; csv_dir.mkdir(parents=True, exist_ok=True)
            with open(csv_dir / f"{fmt_freq_angle(f_ghz, t0_deg)}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["theta_deg", "pattern_dB"])
                for j in range(len(theta)):
                    w.writerow([theta[j], db[j]])


class ScenarioResult:
    """单次优化运行的结构化结果。"""

    def __init__(
        self,
        positions_m: np.ndarray,
        amplitudes: np.ndarray,
        phases_deg: np.ndarray,
        best_individual_raw: np.ndarray,
        fitness: float,
        *,
        array_type: str = "linear",
        component_results: Optional[dict] = None,
        theta_step: float = 0.1,
        elapsed_seconds: float = 0.0,
        timestamp: str = "",
        tag: str = "",
        amp_in_db: bool = False,
    ):
        self.positions_m = np.asarray(positions_m, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.phases_deg = np.asarray(phases_deg, dtype=float)
        self.best_individual_raw = np.asarray(best_individual_raw, dtype=float)
        self.fitness = fitness
        self.array_type = array_type
        self.component_results = component_results or {}
        self.theta_step = theta_step
        self.elapsed_seconds = elapsed_seconds
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tag = tag
        self.amp_in_db = amp_in_db
        self._replay = {}  # 回放参数，from_optimization 设置

    # ── 便捷属性 ──

    @property
    def psll(self) -> Optional[float]:
        return self.component_results.get("sidelobe", {}).get("psll_db")

    @property
    def directivity_dbi(self) -> Optional[float]:
        return self.component_results.get("directivity", {}).get("dbi")

    @property
    def hpbw(self) -> Optional[float]:
        return self.component_results.get("hpbw", {}).get("width_deg")

    # ── 序列化 ──

    def save(self, output_dir: Path, wavelength_m: float = 1.0):
        subdir = Path(output_dir) / f"{self.timestamp}_{self.tag}" if self.tag else \
                 Path(output_dir) / self.timestamp
        subdir.mkdir(parents=True, exist_ok=True)

        # ── elements.txt (合并 x y amplitudes phases) ──
        Ne = len(self.positions_m)
        pos_x = np.asarray(self.positions_m, dtype=float).ravel()
        pos_y = np.zeros(Ne)
        amps = np.asarray(self.amplitudes, dtype=float)
        if self.amp_in_db:
            amps = -20.0 * np.log10(np.maximum(amps, 1e-30))
        phases = np.asarray(self.phases_deg, dtype=float)

        header = (
            f"arrayType: {self.array_type}\n"
            f"elementsNum: {Ne}\n"
            f"{'x':>14} {'y':>14} {'amplitudes':>14} {'phases':>14}"
        )
        fmt = "{:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f}"
        rows = [fmt.format(px, py, a, ph) for px, py, a, ph in zip(pos_x, pos_y, amps, phases)]
        with open(subdir / "elements.txt", "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("\n".join(rows) + "\n")

        # ── bestIndividual.txt ──
        raw = np.asarray(self.best_individual_raw, dtype=float).ravel()
        with open(subdir / "bestIndividual.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(
                f"{v:.6f}".rstrip('0').rstrip('.') if v == v and abs(v) < 1e99
                else f"{v:.6f}" for v in raw
            ) + "\n")

        # ── optResult.json (精简) ──
        result_data = {
            "arrayType": self.array_type,
            "elementsNum": Ne,
            "fitness": round(float(self.fitness), 10),
            "components": self.component_results,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "timestamp": self.timestamp,
            "tag": self.tag,
            "replay": getattr(self, '_replay', {}),
        }
        with open(subdir / "optResult.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return subdir

    # ── 输出（薄壳，委托给独立函数）──

    def print_summary(self):
        print_component_summary(self.component_results)

    def save_figures(self, problem, decoded, cfg, example_name, output_dir, wavelength_m=1.0):
        save_result_figures(self, problem, decoded, cfg, example_name, output_dir, wavelength_m)

    # ── 工厂方法 ──

    @classmethod
    def from_optimization(cls, problem, result_dict: dict, cfg=None,
                          wavelength_m: float = 1.0,
                          elapsed_seconds: float = 0.0,
                          tag: str = "") -> "ScenarioResult":
        """从优化结果构造，自动计算所有激活组件的指标。"""
        x_opt = result_dict["x"]
        decoded = problem.get_result(x_opt)

        positions_wl = decoded["positions_wl"]
        positions_m = positions_wl * wavelength_m
        phases_deg = np.rad2deg(decoded["phases_rad"])
        theta_step = cfg.theta_step if cfg else 0.1
        array_type = "linear"  # 当前仅线阵

        component_results = _compute_component_results(problem, decoded, cfg, theta_step)

        # 回放参数（画图脚本依赖）
        _replay = {
            "frequenciesGHz": problem.pattern.frequenciesGHz.tolist(),
            "theta0sDeg": problem.pattern.theta0s_deg.tolist(),
            "thetaDeg": [
                float(problem.pattern.theta_deg[0]),
                float(problem.pattern.theta_deg[-1]),
                round(abs(float(problem.pattern.theta_deg[1] - problem.pattern.theta_deg[0])), 6),
            ],
        }
        if cfg and cfg.ep_enabled:
            _replay["ep"] = {
                "csvDirectory": cfg.ep_csv_dir,
                "thetaRange": list(cfg.ep_theta_range),
                "isGain": cfg.ep_is_gain,
                "inDB": cfg.ep_in_db,
                "aepMode": cfg.ep_aep_mode,
            }

        sr = cls(
            positions_m=positions_m,
            amplitudes=decoded["amplitudes"],
            phases_deg=phases_deg,
            best_individual_raw=x_opt.copy(),
            fitness=float(result_dict["f"]),
            array_type=array_type,
            component_results=component_results,
            theta_step=theta_step,
            elapsed_seconds=elapsed_seconds,
            tag=tag,
            amp_in_db=getattr(cfg, 'amplitude_in_db', False),
        )
        sr._replay = _replay
        return sr

    @classmethod
    def load(cls, result_dir: Path) -> "ScenarioResult":
        result_dir = Path(result_dir)
        with open(result_dir / "optResult.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # elements.txt 格式: 前 3 行为 header, 之后为数据
        with open(result_dir / "elements.txt", "r") as f:
            raw = f.readlines()
        data_lines = [ln for ln in raw if ln.strip() and ln.strip()[0].isdigit() or ln.strip().startswith('-')]
        cols = np.array([[float(v) for v in line.split()] for line in data_lines])
        positions_m = cols[:, 0]
        amplitudes = cols[:, 2]
        phases_deg = cols[:, 3]

        best = np.loadtxt(result_dir / "bestIndividual.txt")
        cr = meta.get("components", {})
        replay = meta.get("replay", {})

        sr = cls(
            positions_m=positions_m,
            amplitudes=amplitudes,
            phases_deg=phases_deg,
            best_individual_raw=best,
            fitness=meta["fitness"],
            array_type=meta.get("arrayType", "linear"),
            component_results=cr,
            theta_step=meta.get("theta_step", 0.1),
            elapsed_seconds=meta.get("elapsed_seconds", 0.0),
            timestamp=meta.get("timestamp", ""),
            tag=meta.get("tag", ""),
        )
        sr._replay = replay
        return sr


# ═══════════════════════════════════════════════════
#  组件指标计算（角度统一四舍五入到 theta_step）
# ═══════════════════════════════════════════════════

def _snap_angle(deg: float, step: float) -> float:
    """角度四舍五入到采样步长精度，并 round 到干净小数。"""
    val = round(deg / step) * step
    val = 0.0 if abs(val) < step * 0.5 else val
    ndigits = max(0, int(np.ceil(-np.log10(step)))) if step < 1 else 0
    return round(val, ndigits) if ndigits > 0 else float(val)


def _fmt_deg_str(deg: float, step: float) -> str:
    ndigits = max(0, -int(np.floor(np.log10(step))) if step < 1 else 0)
    return f"{_snap_angle(deg, step):.{ndigits}f}"


def _compute_component_results(problem, decoded: dict, cfg, theta_step: float) -> dict:
    """从解码结果计算所有激活组件的指标。"""
    from ..analysis.metrics import get_psll, get_overall_psll, compute_directivity

    af_db = decoded["af_db"]
    af = decoded["af"]
    theta = problem.pattern.theta_deg
    tgt = cfg.target_components if cfg else {}

    # 展平
    af_db_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]
    af_flat = np.abs(af).reshape(-1, *np.abs(af).shape[-1:]) if np.abs(af).ndim > 1 else np.abs(af)[None, :]

    results = {}

    # ── sidelobe ──
    if "sidelobe" in tgt:
        sc = tgt["sidelobe"]
        mlw = sc.get("mainlobeNullWidthDeg")
        ml_region = None
        if mlw and mlw > 0:
            t0 = cfg.theta0s[0] if cfg.theta0s else 0.0
            ml_region = (t0 - mlw / 2, t0 + mlw / 2)

        worst_psll = -np.inf
        worst_angle = None
        for sub in range(af_db_flat.shape[0]):
            v, a = get_overall_psll(af_db_flat[sub], theta, mainlobe_region=ml_region)
            if not np.isinf(v) and v > worst_psll:
                worst_psll = float(v)
                worst_angle = float(a[0]) if hasattr(a, '__len__') else float(a)
        if not np.isinf(worst_psll) and worst_angle is not None:
            worst_angle = _snap_angle(worst_angle, theta_step)
        results["sidelobe"] = {
            "psll_db": round(worst_psll, 2) if not np.isinf(worst_psll) else None,
            "psll_angle_deg": worst_angle,
            "target_db": sc.get("targetDb"),
        }

    # ── main_lobe_pointing ──
    if "main_lobe_pointing" in tgt:
        t0 = cfg.theta0s[0] if cfg.theta0s else 0.0
        peak = float(theta[np.argmax(af_db_flat[0])])
        peak = _snap_angle(peak, theta_step)
        results["main_lobe_pointing"] = {
            "target_deg": t0,
            "peak_deg": peak,
        }

    # ── null_steering ──
    if "null_steering" in tgt:
        nc = tgt["null_steering"]
        angles = nc.get("anglesDeg", [])
        wh = nc.get("windowHalfDeg")
        if wh is None:
            wh = [3.0] * len(angles)
        if isinstance(wh, (int, float)):
            wh = [float(wh)] * len(angles)
        target_null_db = nc.get("targetDb", -80.0)
        nulls = []
        for i, na in enumerate(angles):
            if isinstance(na, (tuple, list)):
                na = na[0]
            half = float(wh[i]) if i < len(wh) else float(wh[-1])
            mask = (theta >= na - half) & (theta <= na + half)
            p_max = float(np.max(af_db_flat[0][mask])) if mask.any() else 0.0
            nulls.append({
                "angle_deg": float(na),
                "depth_db": round(p_max, 2),
                "target_db": target_null_db,
            })
        results["null_steering"] = {"nulls": nulls}

    # ── directivity ──
    if "directivity" in tgt:
        dc = tgt["directivity"]
        try:
            t_range = (theta[0], theta[-1], abs(theta[1] - theta[0]))
            D_dBi, _ = compute_directivity(af_flat[0], t_range, (0, 0, 1), field_type='field')
            results["directivity"] = {
                "dbi": round(float(D_dBi), 2),
                "target_dbi": float(dc.get("targetDbi", 10.0)),
            }
        except Exception:
            results["directivity"] = {"dbi": None, "target_dbi": float(dc.get("targetDbi", 10.0))}

    # ── hpbw ──
    if "hpbw" in tgt:
        hc = tgt["hpbw"]
        slice_db = af_db_flat[0]
        if slice_db.ndim == 2:
            slice_db = slice_db[:, 0]
        bw = _calc_hpbw(slice_db, theta)
        results["hpbw"] = {
            "width_deg": round(bw, 2) if bw else None,
            "target_deg": float(hc.get("targetDeg", 10.0)),
        }

    # ── difference_beam ──
    if "difference_beam" in tgt:
        dc = tgt["difference_beam"]
        t0 = cfg.theta0s[0] if cfg.theta0s else 0.0
        idx0 = int(np.argmin(np.abs(theta - t0)))
        _flat0 = af_db_flat[0]
        c = float(_flat0[idx0]) if _flat0.ndim == 1 else float(np.max(_flat0[idx0, :]))
        mid = len(theta) // 2
        l = float(np.max(_flat0[:mid] if _flat0.ndim == 1 else _flat0[:mid, :]))
        r = float(np.max(_flat0[mid:] if _flat0.ndim == 1 else _flat0[mid:, :]))
        results["difference_beam"] = {
            "null_depth_db": round(c, 2),
            "null_target_db": float(dc.get("nullTargetDb", -60.0)),
            "left_peak_db": round(l, 2),
            "right_peak_db": round(r, 2),
            "symmetry_db": round(abs(l - r), 2),
        }

    return results


def _calc_hpbw(slice_db, theta_deg) -> Optional[float]:
    peak_idx = int(np.argmax(slice_db))
    left_idx = peak_idx
    while left_idx > 0 and slice_db[left_idx] > -3.0:
        left_idx -= 1
    if left_idx < peak_idx and slice_db[left_idx] <= -3.0:
        t_left = theta_deg[left_idx] + (theta_deg[left_idx + 1] - theta_deg[left_idx]) * \
                 (-3.0 - slice_db[left_idx]) / max(slice_db[left_idx + 1] - slice_db[left_idx], 1e-10)
    else:
        t_left = theta_deg[left_idx]
    right_idx = peak_idx
    while right_idx < len(slice_db) - 1 and slice_db[right_idx] > -3.0:
        right_idx += 1
    if right_idx > 0 and slice_db[right_idx] <= -3.0:
        t_right = theta_deg[right_idx - 1] + (theta_deg[right_idx] - theta_deg[right_idx - 1]) * \
                  (-3.0 - slice_db[right_idx - 1]) / max(slice_db[right_idx] - slice_db[right_idx - 1], 1e-10)
    else:
        t_right = theta_deg[right_idx]
    return float(t_right - t_left)
