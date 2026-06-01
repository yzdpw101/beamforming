"""analysis — 方向图分析与可视化。"""

from .metrics import (
    find_peaks, get_psll, get_overall_psll,
    compute_directivity, compute_gain_drop,
    _find_peaks,  # space_mapping 依赖
)
from .plotting import (
    PatternPlot, plot_pattern_comparison, plot_convergence,
    plot_amplitude_distribution, plot_positions_1d, plot_positions_2d,
    plot_3d_spherical, plot_3d_uv,
)
