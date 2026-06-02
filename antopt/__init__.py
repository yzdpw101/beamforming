"""antopt — 阵列天线波束成形优化库。

核心能力：低副瓣、任意零陷、差波束、方向性约束。

子包：
  antenna        — 天线基础模型（Pattern / Geometry / ElementPattern）
  mapping        — 变量 → 物理位置映射（LMMapper / PlanarMapper）
  opt            — 优化管线（Config → Scenario → Problem → Solver）
  analysis       — 方向图分析 + 可视化（PSLL / 方向性系数 / 绘图）
  io             — HFSS 读写 + 结果存取
  space_mapping  — 空间映射（ASM / MSM）
"""

# ── antenna ──
from .antenna import Pattern
from .antenna import Element, ArrayGeometry, LinearArray, UniformLinearArray, PlanarArray
from .antenna import ElementPattern

# ── mapping ──
from .mapping import BaseMapper, LMMapper, PlanarMapper, MGOM

# ── opt ──
from .opt import (
    BaseConfig, BeamformingConfig, ASMConfig, MSMConfig,
    assemble_scenario,
    BeamformingProblem,
    main_lobe_pointing, sidelobe, null_steering,
    directivity, hpbw, difference_beam, _COMPONENT_REGISTRY,
    minimize, CMA, DE, GWO,
)

# ── analysis ──
from .analysis import (
    find_peaks, get_psll, get_overall_psll,
    compute_directivity, compute_gain_drop,
    PatternPlot, plot_pattern_comparison, plot_convergence,
    plot_amplitude_distribution, plot_positions_1d, plot_positions_2d,
    plot_3d_spherical, plot_3d_uv,
)

# ── io ──
from .io import (
    read_hfss_csv, read_hfss_csv_db, read_hfss_dir,
    load_aep_patterns, compute_total_phases,
    ScenarioResult, load_elements,
)

# ── utils ──
from .utils import compute_pattern, to_json_flat, load_array_config, fmt_freq_angle

# ── experiment ──
from .experiment import run_single, run_batch, collect_results
