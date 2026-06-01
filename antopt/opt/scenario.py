"""场景组装器 — 从 BeamformingConfig 构建完整的优化问题。

职责：
  1. 解析三态逻辑（position / amplitude / phase 各自独立），补全默认值
  2. 创建 LMMapper、Pattern 实例
  3. 预计算性能 flag（对称/默认幅相/EP 加载）
  4. 构建组件列表 → BeamformingProblem
  5. 生成优化变量 bounds

三态模式默认值：写 Config.json 时只需指定 source，其余自动补全。
  position: ""(优化) → Ne=16, L=7.5λ, dmin=0.5λ, symmetric=false, fixedAperture=true
  amplitude: "optimize" → bounds=[0,20]dB, inDB=true, step=0.5
  phase: "optimize" → step=5.625° (5-bit移相器)
"""

from pathlib import Path
from typing import Optional
import numpy as np

from .config import BeamformingConfig
from ..mapping import LMMapper
from ..antenna.pattern import Pattern
from ..antenna.element_pattern import ElementPattern
from .problem import BeamformingProblem, _COMPONENT_REGISTRY
from ..utils import load_array_config

# ═══════════════════════════════════════════════════
#  各模式的默认值 — 用户只写 source 即可
# ═══════════════════════════════════════════════════

_POSITION_DEFAULTS = {
    "optimize": dict(Ne=16, L_wavelength=15.5, dmin_wavelength=0.5,
                     symmetric=False, fixedAperture=True),
    "import":   dict(symmetric=False, fixedAperture=False),
}

_AMPLITUDE_DEFAULTS = {
    "optimize": dict(bounds=(0.0, 20.0), inDB=True, step=0.5, optimizeHalf=False),
    "default":  dict(),
    "import":   dict(),
}

_PHASE_DEFAULTS = {
    "optimize": dict(step=5.625, optimizeHalf=False),
    "default":  dict(),
    "import":   dict(),
}


def _classify_source(source: str, opt_keywords=("optimize",), def_keywords=("default",)):
    """判断 source 属于哪种模式: 'optimize' | 'uniform' | 'default' | 'import'。"""
    if source in opt_keywords:
        return source
    if source in def_keywords:
        return "default"
    if source:
        return "import"
    return "optimize"


def _merge_defaults(raw_value, defaults: dict, cfg_value=None):
    """将 raw_value 与 defaults 合并, raw_value 优先。cfg_value 为 config 中的显式值。"""
    result = dict(defaults)
    result.update({k: v for k, v in raw_value.items() if k in defaults})
    return result


def assemble_scenario(cfg: BeamformingConfig) -> tuple:
    """从配置组装优化场景。

    Returns:
        (problem, bounds, x0)
    """
    # ── 1. 解析位置模式并补默认值 ──
    freqs_sorted = np.sort(np.array(cfg.frequenciesGHz, dtype=float))
    f0 = freqs_sorted[0]

    pos_source = cfg.position_source
    pos_mode = _classify_source(pos_source, opt_keywords=("optimize", "uniform"),
                                 def_keywords=())
    pos_defaults = _POSITION_DEFAULTS.get(pos_mode, {})
    pos_symmetric = cfg.position_symmetric if cfg.position_symmetric else pos_defaults.get("symmetric", False)
    pos_fixed = cfg.position_fixed_aperture if cfg.position_fixed_aperture else pos_defaults.get("fixedAperture", True)

    mapper = None
    init_positions = None

    if pos_mode == "uniform":
        Ne = cfg.position_Ne
        dmin = cfg.position_dmin
        unit = cfg.position_unit
        if unit == "lam":
            d_wl = dmin
        elif unit == "mm":
            d_wl = dmin * 1e-3 / (299792458.0 / (f0 * 1e9))
        elif unit == "m":
            d_wl = dmin / (299792458.0 / (f0 * 1e9))
        else:
            raise ValueError(f"uniform Unit 不支持: {unit!r}, 应为 m/mm/lam")
        L = (Ne - 1) * d_wl
        init_positions = np.linspace(-L/2, L/2, Ne)
        pos_symmetric = True
        pos_fixed = True

    elif pos_mode == "import":
        try:
            init_positions = load_array_config(pos_source, "xCenters")
        except (FileNotFoundError, KeyError):
            init_positions = load_array_config(pos_source, "positions")
        if init_positions is not None:
            lam0 = 299792458.0 / (f0 * 1e9)
            init_positions = np.asarray(init_positions, dtype=float) / lam0
            _p = init_positions
            if len(_p) > 1 and np.allclose(_p, -_p[::-1], atol=1e-6):
                pos_symmetric = True

    # ── 2. 解析幅度模式并补默认值 ──
    amp_source = cfg.amplitude_source
    amp_mode = _classify_source(amp_source)
    amp_defaults = _AMPLITUDE_DEFAULTS.get(amp_mode, {})
    amp_bounds = cfg.amplitude_bounds if cfg.amplitude_bounds != (0.0, 1.0) else amp_defaults.get("bounds", (0.0, 1.0))
    amp_in_db = cfg.amplitude_in_db if cfg.amplitude_in_db is not None else amp_defaults.get("inDB", False)
    amp_step = cfg.amplitude_step if cfg.amplitude_step is not None else (amp_defaults.get("step") if cfg.amplitude_in_db else None)
    amp_opt_half = cfg.amplitude_optimize_half if cfg.amplitude_optimize_half else amp_defaults.get("optimizeHalf", False)

    init_amplitudes = None
    if amp_mode == "import":
        init_amplitudes = load_array_config(amp_source, "amplitudes")

    # ── 3. 解析相位模式并补默认值 ──
    phase_source = cfg.phase_source
    phase_mode = _classify_source(phase_source)
    phase_defaults = _PHASE_DEFAULTS.get(phase_mode, {})
    phase_step = cfg.phase_step if cfg.phase_step is not None else phase_defaults.get("step")
    phase_opt_half = cfg.phase_optimize_half if cfg.phase_optimize_half else phase_defaults.get("optimizeHalf", False)

    init_phases_deg = None
    if phase_mode == "import":
        init_phases_deg = load_array_config(phase_source, "phases")

    # ── 4. 创建 Pattern ──
    is_planar = False
    pat = Pattern(
        array_type="planar" if is_planar else "linear",
        symmetric=pos_symmetric,
        theta_deg_start=cfg.theta_start,
        theta_deg_end=cfg.theta_end,
        theta_deg_step=cfg.theta_step,
        theta0s_deg=np.array(cfg.theta0s),
        frequenciesGHz=freqs_sorted,
    )

    # ── 5. 加载单元方向图 ──
    element_patterns = None
    if cfg.ep_enabled and cfg.ep_csv_dir:
        if cfg.ep_aep_mode:
            element_patterns = ElementPattern.from_hfss_aep(
                cfg.ep_csv_dir,
                frequenciesGHz=freqs_sorted,
                theta=pat.theta_deg,
                num_elements=cfg.position_Ne,
            )
        else:
            th_deg = pat.theta_deg
            element_patterns = [ElementPattern.from_hfss_multi_freq(
                cfg.ep_csv_dir,
                frequenciesGHz=freqs_sorted,
                theta=(float(th_deg[0]), float(th_deg[-1]), abs(float(th_deg[1]-th_deg[0]))),
                input_theta_range=tuple(cfg.ep_theta_range),
                is_gain=cfg.ep_is_gain, in_dB=cfg.ep_in_db,
            )[0]]

    # ── 6. 创建 LMMapper ──
    if mapper is None:
        if init_positions is not None:
            Ne = len(init_positions)
            L = float(np.ptp(init_positions))
            dmin = float(np.min(np.diff(np.sort(init_positions)))) if Ne > 1 else 0.5
        elif cfg.position_Ne > 0:
            Ne = cfg.position_Ne
            L = cfg.position_L_wavelength
            dmin = cfg.position_dmin_wavelength
        else:
            Ne = pos_defaults.get("Ne", 16)
            L = pos_defaults.get("L_wavelength", 7.5)
            dmin = pos_defaults.get("dmin_wavelength", 0.5)

        mapper = LMMapper(
            Ne=Ne, L=L, dmin=dmin,
            is_symmetric=pos_symmetric,
            is_fixed_aperture=pos_fixed,
        )

    # ── 7. 预计算性能 flag ──
    is_default_excitation = (amp_mode == "default" and phase_mode == "default")

    # ── 8. 构建组件列表（从 target.components 读取）──
    components = {}
    theta0_default = cfg.theta0s[0] if cfg.theta0s else 0.0
    tgt_components = cfg.target_components
    theta_range = cfg.theta_range

    _PARAM_MAP = {
        "sidelobe":       [("target_psll", "targetDb", -30.0),
                           ("mainlobe_region", "mainlobeNullWidthDeg", None)],
        "null_steering":  [("null_angles_deg", "anglesDeg", []),
                           ("null_target_db", "targetDb", -80.0),
                           ("null_window_half_deg", "windowHalfDeg", [3.0])],
        "directivity":    [("target_dbi", "targetDbi", 10.0)],
        "hpbw":           [("target_hpbw", "targetDeg", 10.0)],
    }

    for name, comp_cfg in tgt_components.items():
        if name not in _COMPONENT_REGISTRY:
            continue
        fn = _COMPONENT_REGISTRY[name]
        weight = float(comp_cfg.get("weight", 1.0))
        if weight <= 0:
            continue

        params = {}
        if name == "main_lobe_pointing":
            params["theta0_deg"] = theta0_default
        elif name == "difference_beam":
            params["theta0_deg"] = theta0_default
            params["diff_null_target"] = float(comp_cfg.get("nullTargetDb", -60.0))
        elif name in _PARAM_MAP:
            for py_key, json_key, default in _PARAM_MAP[name]:
                val = comp_cfg.get(json_key, default)
                if val is not None:
                    params[py_key] = val
            if name == "sidelobe":
                ml_width = comp_cfg.get("mainlobeNullWidthDeg")
                if ml_width and ml_width > 0:
                    params["mainlobe_region"] = (theta0_default - ml_width / 2,
                                                  theta0_default + ml_width / 2)

        components[name] = (fn, weight, params)

    # ── 9. 创建 BeamformingProblem ──
    problem = BeamformingProblem(
        mapper=mapper,
        pattern=pat,
        components=components,
        position_source="optimize" if pos_mode == "optimize" else (pos_source or "optimize"),
        amplitude_source=amp_source if amp_mode in ("optimize", "default") else amp_source,
        amplitude_bounds=amp_bounds,
        amplitude_in_db=amp_in_db,
        amplitude_step=amp_step,
        phase_source=phase_source if phase_mode in ("optimize", "default") else phase_source,
        phase_step_deg=phase_step,
        optimize_half_amp=amp_opt_half,
        optimize_half_phase=phase_opt_half,
        init_positions=init_positions,
        init_phases_deg=init_phases_deg,
        init_amplitudes=init_amplitudes,
        element_patterns=element_patterns,
        is_default_excitation=is_default_excitation,
        theta0_default=cfg.theta0s[0] if cfg.theta0s else 0.0,
    )

    # ── 10. 构建变量 bounds ──
    n_pos, n_phase, n_amp = problem.n_pos, problem.n_phase, problem.n_amp
    lb = np.empty(problem.n_vars)
    ub = np.empty(problem.n_vars)
    idx = 0

    if n_pos > 0:
        lb[:n_pos] = 0.0; ub[:n_pos] = 1.0
        idx = n_pos
    if n_phase > 0:
        phs_ub = 2 * np.pi
        if phase_step is not None and phase_step > 0:
            phs_ub = 2 * np.pi - np.deg2rad(phase_step)
        lb[idx:idx + n_phase] = 0.0
        ub[idx:idx + n_phase] = phs_ub
        idx += n_phase
    if n_amp > 0:
        lb[idx:] = amp_bounds[0]
        ub[idx:] = amp_bounds[1]

    # ── 11. 初始解 x0 ──
    x0 = None
    if cfg.opt_import_init_individual and cfg.opt_init_individual_dir:
        try:
            x0 = np.loadtxt(cfg.opt_init_individual_dir)
        except (FileNotFoundError, ValueError):
            pass

    return problem, (lb, ub), x0
