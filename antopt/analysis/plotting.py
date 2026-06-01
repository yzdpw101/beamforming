"""标准绘图工具: 方向图对比、收敛曲线、幅度分布 + PatternPlot 标注构建器。"""

from pathlib import Path

import numpy as np
import matplotlib
# ── 学术风格全局设置 ──
matplotlib.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'font.size': 10,
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'legend.framealpha': 0.7,
    'legend.edgecolor': '#CCCCCC',
    'legend.fontsize': 8,
    'legend.borderpad': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
import matplotlib.pyplot as plt

# ── 学术色板 ──
_C_BASE  = '#1F77B4'   # 方向图主曲线 深蓝
_C_PSLL  = '#C44E52'   # 副瓣标记 暗红
_C_NULL  = '#937860'   # 零陷窗口 暗紫褐
_C_HPBW  = '#DD8452'   # 波束宽度 暗橙
_C_POINT = '#55A868'   # 指向标记 暗绿
_C_DIFF  = '#4C72B0'   # 差波束 暗蓝
_C_DASH  = '#AAAAAA'   # 参考线 浅灰


def plot_pattern_comparison(theta, curves, title="", ylim=(-40, 3),
                            error_ylim=(-10, 10), save_path=None, figsize=(10, 8)):
    """方向图 + 误差图 (上下两子图)。

    Args:
        theta: 角度数组
        curves: dict, {label: (pattern_db, color, ls)}
            pattern_db 为归一化 dB 方向图
        title: 总标题
        ylim: 方向图 y 轴范围
        error_ylim: 误差图 y 轴范围
        save_path: 保存路径 (None=不保存)
        figsize: 图片尺寸

    Returns:
        (fig, axes)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 上图: 方向图对比
    ax = axes[0]
    ref_db = None
    for label, (db, color, ls) in curves.items():
        ax.plot(theta, db, color=color, ls=ls, lw=1.0, label=label)
        if ref_db is None:
            ref_db = db

    ax.set_title(title)
    ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 下图: 误差 (相对于第一条曲线)
    ax = axes[1]
    if ref_db is not None and len(curves) > 1:
        second_key = list(curves.keys())[1]
        second_db = curves[second_key][0]
        mask = ref_db > -40
        error = np.zeros_like(theta)
        error[mask] = ref_db[mask] - second_db[mask]
        rms = np.sqrt(np.mean(error[mask] ** 2)) if mask.any() else 0
        ax.plot(theta, error, 'k-', lw=0.8)
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_title(f'误差 (RMS={rms:.2f} dB)')
    else:
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_title('误差')

    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Error (dB)")
    ax.set_ylim(error_ylim)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_convergence(iters, values, xlabel="Iteration", ylabel="PSLL (dB)",
                     title="", save_path=None, figsize=(8, 4)):
    """收敛曲线 (单条)。

    Args:
        iters: 迭代号数组
        values: 对应值数组
        xlabel, ylabel: 坐标轴标签
        title: 标题
        save_path: 保存路径

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iters, values, "o-", lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_amplitude_distribution(amps, title="", save_path=None, figsize=(10, 4)):
    """幅度分布柱状图。

    Args:
        amps: 幅度数组
        title: 标题
        save_path: 保存路径

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(amps)), amps, width=0.8, color='steelblue')
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════
#  PatternPlot — 2D 方向图切面标注构建器
# ═══════════════════════════════════════════════════

class PatternPlot:
    """2D 方向图切面 — 链式标注组件叠加。

    用法:
        PatternPlot(theta, db).add_psll().add_hpbw().save("out.png")
        p = PatternPlot(theta, db, interactive=True)
        p.add_null([30], [3.0]).add_pointing(0).show()

    x 轴: 度 (°), y 轴: dB.
    """

    def __init__(self, theta, pattern_db, *,
                 title="", xlabel=r"$\theta$ (deg)", ylabel="dB",
                 db_floor=-60, figsize=(8, 5),
                 interactive=False, ax=None):
        self.theta = np.asarray(theta, dtype=float)
        self.db = np.asarray(pattern_db, dtype=float)
        self.db_floor = db_floor
        self.interactive = interactive
        self._owns_figure = (ax is None)
        # theta 采样步长 (用于角度显示精度)
        self._dtheta = abs(float(self.theta[1] - self.theta[0])) if len(self.theta) > 1 else 1.0

        if ax is not None:
            self.fig = ax.figure
            self.ax = ax
        else:
            if not interactive:
                matplotlib.use('Agg')
            self.fig, self.ax = plt.subplots(figsize=figsize)
        self._draw_base()
        self.ax.set_title(title, fontsize=12, fontweight='normal')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_ylim(db_floor, 3)
        self.ax.grid(True, which='major')
        self.ax.grid(True, which='minor', alpha=0.15)
        # -3 dB 参考线
        self.ax.axhline(-3, color=_C_DASH, ls='--', lw=0.6, alpha=0.6)

    def _draw_base(self):
        self.ax.plot(self.theta, self.db, color=_C_BASE, lw=1.2,
                     solid_capstyle='round')

    # ── 标注组件 ──

    @property
    def _theta_step(self):
        return self._dtheta

    def _snap(self, deg: float) -> float:
        """将角度四舍五入到 theta 采样步长精度。"""
        step = self._dtheta
        val = round(deg / step) * step
        return 0.0 if abs(val) < step * 0.5 else val

    def _fmt_deg(self, deg: float, force_sign: bool = False) -> str:
        """格式化角度，匹配 theta 采样精度。"""
        step = self._dtheta
        ndigits = max(0, -int(np.floor(np.log10(step))) if step < 1 else 0)
        val = self._snap(deg)
        return f'{val:+.{ndigits}f}°' if force_sign else f'{val:.{ndigits}f}°'

    def add_psll(self, mainlobe_region=None):
        """标记最大副瓣电平。"""
        from .metrics import get_psll
        psll_v, psll_a = get_psll(self.db, self.theta,
                                   mainlobe_region=mainlobe_region)
        if not np.isinf(psll_v):
            self.ax.axhline(psll_v, color=_C_PSLL, ls='--', lw=0.7,
                            label=f'PSLL = {psll_v:.1f} dB')
            self.ax.plot(self._snap(psll_a), psll_v, 'o', color=_C_PSLL, ms=5,
                         markeredgewidth=0.5, markeredgecolor='white')
        return self

    def add_null(self, angles_deg, window_half_deg=None):
        """标记零陷窗口。

        Args:
            angles_deg: 零陷中心角列表
            window_half_deg: 窗口半宽 (单值或与 angles_deg 等长列表)
        """
        if window_half_deg is None:
            window_half_deg = [3.0] * len(angles_deg)
        if isinstance(window_half_deg, (int, float)):
            window_half_deg = [float(window_half_deg)] * len(angles_deg)
        while len(window_half_deg) < len(angles_deg):
            window_half_deg.append(window_half_deg[-1])

        for ang, w2 in zip(angles_deg, window_half_deg):
            wL, wR = float(ang) - float(w2), float(ang) + float(w2)
            mask = (self.theta >= wL) & (self.theta <= wR)
            if mask.any():
                pk = float(np.max(self.db[mask]))
                pk_t = self._snap(float(self.theta[mask][np.argmax(self.db[mask])]))
                self.ax.plot(pk_t, pk, 'D', color=_C_NULL, ms=4,
                             markeredgewidth=0.3, markeredgecolor='white')
                self.ax.fill_betweenx([self.db_floor, 3], wL, wR,
                                      alpha=0.08, color=_C_NULL,
                                      label=f'null {ang}°: {pk:.1f} dB')
        return self

    def add_hpbw(self):
        """标记半功率波束宽度。"""
        pi = int(np.argmax(self.db))
        L, R = pi, pi
        while L > 0 and self.db[L] > -3.0: L -= 1
        while R < len(self.db) - 1 and self.db[R] > -3.0: R += 1
        tL, tR = self.theta[L], self.theta[R]
        self.ax.axvline(tL, color=_C_HPBW, ls=':', lw=0.7)
        self.ax.axvline(tR, color=_C_HPBW, ls=':', lw=0.7)
        self.ax.fill_betweenx([self.db_floor, 3], tL, tR,
                              alpha=0.06, color=_C_HPBW,
                              label=f'HPBW = {tR - tL:.1f}°')
        return self

    def add_pointing(self, theta0_deg):
        """标记主瓣指向。"""
        peak_t = self._snap(float(self.theta[int(np.argmax(self.db))]))
        self.ax.axvline(theta0_deg, color=_C_POINT, ls='-', lw=0.7, alpha=0.7,
                        label=f'target $\\theta_0$ = {self._fmt_deg(theta0_deg)}')
        self.ax.plot(peak_t, float(np.max(self.db)), '*', color=_C_POINT, ms=8,
                     markeredgewidth=0.3, markeredgecolor='white',
                     label=f'peak = {self._fmt_deg(peak_t)}')
        return self

    def add_diff_beam(self, theta0_deg):
        """差波束标记（中心零陷 + 左右峰对称度）。"""
        idx0 = int(np.argmin(np.abs(self.theta - theta0_deg)))
        mid = len(self.theta) // 2
        left_pk = float(np.max(self.db[:mid]))
        right_pk = float(np.max(self.db[mid:]))
        center_lvl = float(self.db[idx0])
        self.ax.axvline(theta0_deg, color=_C_DIFF, ls='-', lw=0.7, alpha=0.7,
                        label=f'null depth = {center_lvl:.1f} dB')
        self.ax.axhline(left_pk, color=_C_DIFF, ls=':', lw=0.5, alpha=0.5)
        self.ax.axhline(right_pk, color=_C_DIFF, ls=':', lw=0.5, alpha=0.5)
        return self

    def add_peaks(self, n=5):
        """标记前 n 个峰值。"""
        from .metrics import find_peaks
        pk_val, pk_ang = find_peaks(self.db, self.theta)
        for pv, pa in zip(pk_val[:n], pk_ang[:n]):
            self.ax.plot(pa, pv, '.', color=_C_BASE, ms=3, alpha=0.5)
        return self

    def add_directivity(self, dbi: float = None):
        """在图上标注方向性系数。"""
        if dbi is not None:
            self.ax.text(0.98, 0.03, f'D = {dbi:.1f} dBi',
                         transform=self.ax.transAxes, ha='right', va='bottom',
                         fontsize=9, color=_C_DASH,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=_C_DASH, alpha=0.7))
        return self

    def add_all(self, **kw):
        """一键叠加全部标记。"""
        self.add_psll(**{k: v for k, v in kw.items() if k in ('mainlobe_region',)})
        self.add_hpbw()
        return self

    # ── 输出 ──

    def save(self, path=None, dpi=300):
        """保存图片。path=None/False 时不保存。

        格式由后缀决定：
          .pdf → 矢量 PDF（推荐论文用）
          .svg → 矢量 SVG
          .png → 位图（dpi 控制清晰度，默认 300）
        """
        if path and self._owns_figure:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            h, _l = self.ax.get_legend_handles_labels()
            if h:
                self.ax.legend(loc='upper right', frameon=True)
            self.fig.tight_layout(pad=0.8)
            fmt = p.suffix.lstrip('.')
            if fmt in ('pdf', 'svg'):
                self.fig.savefig(str(p), format=fmt, bbox_inches="tight")
            else:
                self.fig.savefig(str(p), dpi=dpi, bbox_inches="tight",
                                 pil_kwargs={'compress_level': 6})
        return self

    def show(self):
        """弹出交互窗口。interactive=True 且拥有 Figure 时有效。"""
        if self.interactive and self._owns_figure:
            h, _l = self.ax.get_legend_handles_labels()
            if h:
                self.ax.legend(loc='upper right', frameon=True)
            self.fig.tight_layout(pad=0.8)
            plt.show()
        return self

    def figure(self):
        """返回底层 matplotlib Figure。"""
        return self.fig


# ═══════════════════════════════════════════════════
#  阵元位置图
# ═══════════════════════════════════════════════════

def plot_positions_1d(positions_m, color_values=None, *,
                      title="Element Positions", color_label="Value",
                      save_path=None, ax=None):
    """线阵阵元位置分布 — 散射点, 颜色+大小编码值。

    Args:
        positions_m: 阵元位置 (m)
        color_values: 用于颜色和大小编码的值 (幅度/相位等)
        title: 标题
        color_label: 颜色条标签
        save_path: 保存路径 (None=不保存)
        ax: 可选 matplotlib Axes (嵌入用)
    """
    pos = np.asarray(positions_m, dtype=float) * 1000  # m → mm
    Ne = len(pos)

    own_fig = (ax is None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2.5))
    else:
        fig = ax.figure

    if color_values is not None:
        vals = np.asarray(color_values, dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if abs(vmax - vmin) < 1e-6:
            # 值全相同: 用固定颜色, 无 colorbar
            ax.scatter(pos, np.zeros(Ne), s=90, color=_C_BASE,
                       edgecolors='#444444', linewidths=0.5, zorder=3)
            ax.text(0.98, 0.95, f'All = {vmin:.3g}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='#666666')
        else:
            sizes = 30 + 100 * (vals - vmin) / (vmax - vmin)
            sc = ax.scatter(pos, np.zeros(Ne), s=sizes, c=vals,
                            cmap='coolwarm', edgecolors='#444444',
                            linewidths=0.5, zorder=3)
            fig.colorbar(sc, ax=ax, shrink=0.6, label=color_label, aspect=30)

    # 坐标标注
    for i, x in enumerate(pos):
        ax.annotate(f'{x:.2f}', (x, 0), textcoords='offset points',
                    xytext=(0, -15), ha='center', fontsize=6,
                    color='#666666', rotation=45)

    ax.axhline(0, color=_C_DASH, lw=0.5, zorder=0)
    ax.set_ylim(-0.3, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Position (mm)')
    ax.set_title(title, fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.25, axis='x')

    if own_fig:
        fig.tight_layout(pad=0.5)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig, ax


def plot_positions_2d(positions_x_m, positions_y_m, color_values=None, *,
                      title="Element Positions", color_label="Value",
                      save_path=None):
    """平面阵阵元位置 — 散点, 颜色+大小编码值。

    Args:
        positions_x_m: x 坐标 (m)
        positions_y_m: y 坐标 (m)
        color_values: 用于颜色和大小编码的值
        title: 标题
        color_label: 颜色条标签
        save_path: 保存路径
    """
    px = np.asarray(positions_x_m, dtype=float) * 1000  # m → mm
    py = np.asarray(positions_y_m, dtype=float) * 1000
    Ne = len(px)

    fig, ax = plt.subplots(figsize=(7, 6))
    if color_values is not None:
        vals = np.asarray(color_values, dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if abs(vmax - vmin) < 1e-6:
            ax.scatter(px, py, s=70, color=_C_BASE,
                       edgecolors='#333333', linewidths=0.3, alpha=0.85)
            ax.text(0.98, 0.97, f'All = {vmin:.3g}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='#666666')
        else:
            sizes = 20 + 80 * (vals - vmin) / (vmax - vmin)
            sc = ax.scatter(px, py, c=vals, s=sizes, cmap='coolwarm',
                            edgecolors='#333333', linewidths=0.3, alpha=0.85)
            fig.colorbar(sc, ax=ax, shrink=0.8, label=color_label)

    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.25, lw=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════
#  3D 方向图
# ═══════════════════════════════════════════════════

def plot_3d_spherical(theta_deg, phi_deg, pattern_db, *,
                      db_floor=-50, title="", save_path=None,
                      view_elev=30, view_azim=-60):
    """球坐标 3D 方向图 (仿 MATLAB patternCustom)。

    以 θ, φ 为自变量, dB 值为因变量, 在直角坐标 3D 空间中绘制曲面。

    Args:
        theta_deg: θ 角度 (度), shape (Nθ,)
        phi_deg: φ 角度 (度), shape (Nφ,)
        pattern_db: dB 方向图, shape (Nθ, Nφ)
        db_floor: dB 下限
        title: 标题
        save_path: 保存路径
        view_elev, view_azim: 初始视角
    """
    TH, PH = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing='ij')
    DB = np.maximum(pattern_db, db_floor)
    # dB → 正半径: db_floor → 1.0, 0dB → 1.5
    R = 1.0 + (DB - db_floor) / abs(db_floor) * 0.5

    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(
        (DB - db_floor) / abs(db_floor)),
        linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title, fontsize=12, fontweight='normal')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=view_elev, azim=view_azim)

    mappable = plt.cm.ScalarMappable(
        cmap='jet', norm=plt.Normalize(vmin=db_floor, vmax=0))
    mappable.set_array(DB)
    fig.colorbar(mappable, ax=ax, shrink=0.5, label='dB')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_3d_uv(theta_deg, phi_deg, pattern_db, *,
               db_floor=-50, title="", save_path=None):
    """UV 空间 3D 方向图 (仿 MATLAB surf(u,v,Pattern))。

    u = sinθ·cosφ, v = sinθ·sinφ, 增益归一化 dB。
    保存时同时输出俯视图 (top_view)。

    Args:
        theta_deg: θ 角度 (度), shape (Nθ,)
        phi_deg: φ 角度 (度), shape (Nφ,)
        pattern_db: dB 方向图, shape (Nθ, Nφ)
        db_floor: dB 下限
        title: 标题
        save_path: 保存路径 (主图); 俯视图自动加 _top 后缀
    """
    TH, PH = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing='ij')
    U = np.sin(TH) * np.cos(PH)
    V = np.sin(TH) * np.sin(PH)
    P = np.maximum(pattern_db, db_floor)

    # ── 3D surf ──
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(U, V, P, cmap='jet', linewidth=0,
                           antialiased=True, alpha=0.9, shade=True)
    ax.set_xlabel(r'$u = \sin\theta\cos\phi$')
    ax.set_ylabel(r'$v = \sin\theta\sin\phi$')
    ax.set_zlabel('Gain (dB)')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(db_floor, 0)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title(title, fontsize=12, fontweight='normal')
    ax.view_init(elev=45, azim=-45)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='dB')

    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(sp), dpi=300, bbox_inches="tight")

        # ── 俯视图 ──
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        c = ax2.contourf(U, V, P, levels=50, cmap='jet',
                         vmin=db_floor, vmax=0)
        ax2.set_xlabel(r'$u = \sin\theta\cos\phi$')
        ax2.set_ylabel(r'$v = \sin\theta\sin\phi$')
        ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1)
        ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2.set_aspect('equal')
        ax2.set_title(title + ' (俯视)', fontsize=12, fontweight='normal')
        ax2.grid(True, alpha=0.2, lw=0.3)
        fig2.colorbar(c, ax=ax2, shrink=0.7, label='dB')
        fig2.tight_layout()
        fig2.savefig(str(sp.parent / (sp.stem + '_top' + sp.suffix)),
                     dpi=300, bbox_inches="tight")
        plt.close(fig2)

    plt.close(fig)
    return fig, ax
