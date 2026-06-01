"""波束成形问题 — 组件化加权代价模型。

六个独立代价组件，每个返回 ≥ 0 的标量代价值：
  main_lobe_pointing — 主瓣指向偏差惩罚
  sidelobe          — 最大副瓣电平约束
  null_steering     — 多零陷深度约束
  directivity       — 方向性系数下限约束
  hpbw              — 半功率波束宽度上限约束
  difference_beam   — 差波束对称性约束

总适应度 = Σ w_i × C_i，优化器最小化该标量。
"""

import warnings
from typing import Optional, Callable
import numpy as np

from ..mapping import LMMapper
from ..antenna.pattern import Pattern, TWO_PI
from ..analysis.metrics import get_psll, get_overall_psll, find_peaks, compute_directivity


# ═══════════════════════════════════════════════
#  代价组件 (callable)
# ═══════════════════════════════════════════════

def _find_main_peak_idx(af_db, theta_deg):
    """找到方向图最高峰值对应的 theta 索引。"""
    if af_db.ndim == 2:
        idx = np.unravel_index(np.argmax(af_db), af_db.shape)
        return theta_deg[idx[0]]
    return theta_deg[np.argmax(af_db)]


def main_lobe_pointing(af_db, theta_deg, theta0_deg, **_kw) -> float:
    """主瓣指向偏差惩罚 C = (θ_peak − θ₀)²。"""
    theta_peak = _find_main_peak_idx(af_db, theta_deg)
    return float((theta_peak - theta0_deg) ** 2)


def sidelobe(af_db, theta_deg, target_psll=-30.0,
             mainlobe_region=None, **_kw) -> float:
    """副瓣约束 C = max(0, PSLL_current − target_psll)。"""
    psll, _ = get_overall_psll(af_db, theta_deg,
                                mainlobe_region=mainlobe_region)
    if np.isinf(psll):
        return 0.0
    return max(0.0, psll - target_psll)


def null_steering(af_db, theta_deg, null_angles_deg, null_target_db=-80.0,
                  null_window_half_deg=None, **_kw) -> float:
    """零陷约束 C = Σ max(0, P_max_in_window − null_target)。

    Args:
        null_angles_deg: [(θ1,), (θ2,), ...] 零陷中心角列表
        null_window_half_deg: 单值(所有零陷共用) 或 列表(与角度一一对应)
    """
    total = 0.0
    angles = null_angles_deg or []
    if null_window_half_deg is None:
        null_window_half_deg = [3.0] * len(angles)
    if isinstance(null_window_half_deg, (int, float)):
        null_window_half_deg = [float(null_window_half_deg)] * len(angles)
    while len(null_window_half_deg) < len(angles):
        null_window_half_deg.append(null_window_half_deg[-1])

    for i, null_angle in enumerate(angles):
        if isinstance(null_angle, (tuple, list)):
            theta_null = null_angle[0]
        else:
            theta_null = null_angle
        half = float(null_window_half_deg[i])
        window = (theta_null - half, theta_null + half)
        mask = (theta_deg >= window[0]) & (theta_deg <= window[1])
        if af_db.ndim == 2:
            window_max = np.max(af_db[mask, :])
        else:
            window_max = np.max(af_db[mask])
        total += max(0.0, window_max - null_target_db)
    return total


def directivity(af_linear, theta_range, phi_range=(0, 0, 1),
                target_dbi=10.0, **_kw) -> float:
    """方向性约束 C = max(0, targetDbi − D_current)²。"""
    try:
        D_dBi, _ = compute_directivity(
            np.abs(af_linear), theta_range, phi_range, field_type='field')
    except (ValueError, IndexError):
        return 0.0
    return max(0.0, target_dbi - D_dBi) ** 2


def hpbw(af_db, theta_deg, target_hpbw=10.0, **_kw) -> float:
    """HPBW 约束 C = max(0, HPBW_current − target_hpbw)。"""
    if af_db.ndim == 2:
        slice_db = af_db[:, 0]
    else:
        slice_db = af_db
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

    hpbw_current = t_right - t_left
    return max(0.0, hpbw_current - target_hpbw)


def difference_beam(af_db, theta_deg, theta0_deg=0.0,
                    diff_null_target=-60.0, **_kw) -> float:
    """差波束约束 C = max(0, P(θ₀) − null_target) + |P_left − P_right|。"""
    idx0 = int(np.argmin(np.abs(theta_deg - theta0_deg)))
    if af_db.ndim == 2:
        center_level = np.max(af_db[idx0, :])
    else:
        center_level = af_db[idx0]

    null_penalty = max(0.0, center_level - diff_null_target)

    mid = len(theta_deg) // 2
    left_peak = np.max(af_db[:mid] if af_db.ndim == 1 else af_db[:mid, :])
    right_peak = np.max(af_db[mid:] if af_db.ndim == 1 else af_db[mid:, :])
    symmetry_penalty = abs(left_peak - right_peak)

    return null_penalty + symmetry_penalty


# ═══════════════════════════════════════════════
#  组件注册表
# ═══════════════════════════════════════════════

_COMPONENT_REGISTRY = {
    "main_lobe_pointing": main_lobe_pointing,
    "sidelobe": sidelobe,
    "null_steering": null_steering,
    "directivity": directivity,
    "hpbw": hpbw,
    "difference_beam": difference_beam,
}


# ═══════════════════════════════════════════════
#  BeamformingProblem
# ═══════════════════════════════════════════════

class BeamformingProblem:
    """波束成形优化问题 — 组件化代价模型。

    封装变量解码、AF 计算、多组件代价评估。
    """

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        *,
        components: dict,
        position_source: str = "optimize",
        amplitude_source: str = "default",
        amplitude_bounds: tuple = (0.0, 1.0),
        amplitude_in_db: bool = False,
        amplitude_step: Optional[float] = None,
        phase_source: str = "default",
        phase_step_deg: Optional[float] = None,
        optimize_half_amp: bool = False,
        optimize_half_phase: bool = False,
        init_positions: Optional[np.ndarray] = None,
        init_phases_deg: Optional[np.ndarray] = None,
        init_amplitudes: Optional[np.ndarray] = None,
        element_patterns: Optional[list] = None,
        is_default_excitation: bool = False,
        theta0_default: float = 0.0,
        af_method: str = "auto",
    ):
        self.mapper = mapper
        self.pattern = pattern
        self.element_patterns = element_patterns
        self.components = components  # {name: (fn, weight, params)}
        self.is_default_excitation = is_default_excitation
        self.af_method = af_method
        self.theta0_default = theta0_default

        self.amplitude_lower, self.amplitude_upper = amplitude_bounds
        self.amplitude_in_db = amplitude_in_db
        self._amp_step = amplitude_step
        self._phs_step_rad = np.deg2rad(phase_step_deg) if phase_step_deg else None

        # ── 解析 position_source ──
        if position_source in ("optimize", True):
            self._p_mode = 1
            self.init_positions = None
        elif position_source == "uniform":
            self._p_mode = 0
            self.init_positions = init_positions
        elif isinstance(position_source, str):
            self._p_mode = 2
            self.init_positions = init_positions
        elif isinstance(position_source, (np.ndarray, list)):
            self._p_mode = 2
            self.init_positions = np.asarray(position_source, dtype=float)
        else:
            raise ValueError(f"position_source 无效: {position_source!r}")

        # ── 解析 amplitude_source ──
        if amplitude_source in ("optimize", True):
            self._a_mode = 1
            self.init_amplitudes = None
        elif amplitude_source == "default":
            self._a_mode = 0
            self.init_amplitudes = None
        elif isinstance(amplitude_source, str):
            self._a_mode = 2
            self.init_amplitudes = init_amplitudes
        else:
            raise ValueError(f"amplitude_source 无效: {amplitude_source!r}")

        # ── 解析 phase_source ──
        if phase_source in ("optimize", True):
            self._h_mode = 1
            self.init_phases_deg = None
        elif phase_source == "default":
            self._h_mode = 0
            self.init_phases_deg = None
        elif isinstance(phase_source, str):
            self._h_mode = 2
            self.init_phases_deg = init_phases_deg
        else:
            raise ValueError(f"phase_source 无效: {phase_source!r}")

        self.optimize_half_amp = optimize_half_amp and mapper.is_symmetric and self._a_mode == 1
        self.optimize_half_phase = optimize_half_phase and mapper.is_symmetric and self._h_mode == 1
        self._use_symmetric_af = (self.optimize_half_amp and self.optimize_half_phase)

        # ── 变量维度 ──
        self.n_pos = mapper.n_vars if self._p_mode == 1 else 0
        self.n_phase = (
            self._half_n() if self._h_mode == 1 and self.optimize_half_phase
            else (mapper.Ne if self._h_mode == 1 else 0)
        )
        self.n_amp = (
            self._half_n() if self._a_mode == 1 and self.optimize_half_amp
            else (mapper.Ne if self._a_mode == 1 else 0)
        )
        self.n_vars = self.n_pos + self.n_phase + self.n_amp

    def _half_n(self) -> int:
        return self.mapper._halfNe + (1 if self.mapper.has_center else 0)

    def _expand_half(self, half_vals):
        hN = self.mapper._halfNe
        has_c = self.mapper.has_center
        full = np.empty(self.mapper.Ne)
        if has_c:
            full[hN] = half_vals[0]
            right = half_vals[1:]
        else:
            right = half_vals
        full[hN + (1 if has_c else 0):] = right
        full[:hN] = right[::-1]
        return full

    def _decode(self, x: np.ndarray):
        """解码变量向量 → (pos, phases_rad, amplitudes)。"""
        x = np.asarray(x, dtype=float)
        Ne = self.mapper.Ne

        if self._p_mode == 1:
            pos = self.mapper.synthesize(x[:self.n_pos])
        else:
            pos = self.init_positions

        if self._h_mode == 1:
            raw = x[self.n_pos:self.n_pos + self.n_phase]
            if self._phs_step_rad is not None:
                raw = np.round(raw / self._phs_step_rad) * self._phs_step_rad
            phases = self._expand_half(raw) if self.optimize_half_phase else raw
        elif self._h_mode == 2:
            phases = np.deg2rad(self.init_phases_deg)
        else:
            phases = np.zeros(Ne)

        if self._a_mode == 1:
            raw = x[-self.n_amp:]
            raw = self.amplitude_lower + raw * (self.amplitude_upper - self.amplitude_lower)
            if self._amp_step is not None and self._amp_step > 0:
                raw = np.round(raw / self._amp_step) * self._amp_step
            amps = self._expand_half(raw) if self.optimize_half_amp else raw
            if self.amplitude_in_db:
                amps = np.power(10.0, -amps / 20.0)
        elif self._a_mode == 2:
            amps = self.init_amplitudes
        else:
            amps = np.ones(Ne)

        return pos, phases, amps

    def _compute_af(self, pos, amps, phases):
        """计算 AF，返回复数 AF。"""
        if self.pattern.is_planar:
            af = self.pattern.planar_af(pos, np.zeros_like(pos), amps, phases)
        elif self._use_symmetric_af:
            halfNe = self.mapper._halfNe
            offset = 1 if self.mapper.has_center else 0
            kwargs = dict(has_center=self.mapper.has_center,
                          amplitudes=amps[halfNe + offset:],
                          phases=phases[halfNe + offset:])
            if self.mapper.has_center:
                kwargs["center_amplitude"] = amps[halfNe]
                kwargs["center_phase"] = phases[halfNe]
            af = self.pattern.linear_af_symmetric(pos[halfNe + offset:], **kwargs)
        else:
            af = self.pattern.af(pos, amplitudes=amps, phases=phases,
                                 is_default_excitation=self.is_default_excitation,
                                 method=self.af_method)

        # EP 合成
        if self.element_patterns is not None:
            from ..antenna.element_pattern import ElementPattern
            ep = self.element_patterns
            _is_aep = isinstance(ep, list) and len(ep) > 0 and isinstance(ep[0], list)

            if _is_aep:
                pos_arr = np.asarray(pos, dtype=float)
                amps_arr = np.asarray(amps, dtype=float)
                phases_arr = np.asarray(phases, dtype=float)
                Ne = len(pos_arr)
                af_new = np.zeros_like(af, dtype=complex)
                for n in range(Ne):
                    exc_n = amps_arr[n] * np.exp(1j * phases_arr[n])
                    phase_n = TWO_PI * pos_arr[n] * self.pattern._delta_sin
                    af_new += ep[0][n] * exc_n * np.exp(1j * phase_n)
                af = af_new
            else:
                af = af * ep[0]  # IEP: 共用单元方向图

        return af

    def fitness(self, x: np.ndarray) -> float:
        """总适应度 = Σ (w_norm_i × C_i)。多频/多角度取最差组合。"""
        pos, phases, amps = self._decode(x)
        af = self._compute_af(pos, amps, phases)
        af_db = Pattern.to_dB(af)
        af_abs = np.abs(af)

        af_db_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]
        af_abs_flat = af_abs.reshape(-1, af_abs.shape[-1]) if af_abs.ndim > 1 else af_abs[None, :]
        n_combos = af_db_flat.shape[0]

        total_w = sum(w for _, (_, w, _) in self.components.items())
        if total_w <= 0:
            total_w = 1.0

        cost = 0.0
        for name, (fn, w, params) in self.components.items():
            w_norm = w / total_w

            if name == "directivity":
                c = directivity(
                    af_abs_flat[0],
                    theta_range=(self.pattern.theta_deg[0],
                                 self.pattern.theta_deg[-1],
                                 abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0])),
                    phi_range=(0, 0, 1),
                    **params,
                )
            else:
                worst = 0.0
                for sub in range(n_combos):
                    c_sub = fn(af_db_flat[sub], theta_deg=self.pattern.theta_deg, **params)
                    worst = max(worst, c_sub)
                c = worst

            cost += w_norm * c

        return cost

    def get_result(self, x_opt: np.ndarray) -> dict:
        """解码最优个体，返回结果字典。"""
        pos, phases, amps = self._decode(x_opt)
        af = self._compute_af(pos, amps, phases)
        af_db = Pattern.to_dB(af)

        return {
            "positions_wl": pos,
            "phases_rad": phases,
            "amplitudes": amps,
            "af": af,
            "af_db": af_db,
            "n_vars": self.n_vars,
            "n_pos": self.n_pos,
            "n_phase": self.n_phase,
            "n_amp": self.n_amp,
        }
