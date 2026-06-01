"""方向图计算器。

构造时预计算所有电磁常量（波数）和角度网格（sinθ、deltaSin），
计算时只传阵元位置和激励，避免优化循环中重复计算。

位置坐标均以波长为单位。单频时 k=2π（不使用 wavelength），
多频时根据频率比缩放波数。

支持的维度组合（通过构造函数参数控制）：
  阵列类型: 线阵 / 平面阵 (array_type)
  对称性:   非对称 / 对称 (symmetric)
  扫描角:   单角度 (theta0s 标量) / 多角度 (theta0s 数组)
  频率:     单频 (frequenciesGHz 标量) / 多频 (frequenciesGHz 数组)
"""

from typing import Optional, Union, Literal
import numpy as np

TWO_PI = 2 * np.pi


class Pattern:
    """方向图计算器。

    构造时预计算角度网格和电磁常量，后续 compute 只传阵元配置。

    Args:
        array_type: 阵列类型, "linear"=线阵, "planar"=平面阵
        symmetric: 是否对称阵列，True 时只传半边位置
        theta_deg_start: 俯仰角起始（度）
        theta_deg_end: 俯仰角终止（度）
        theta_deg_step: 俯仰角步长（度）
        theta0s_deg: 波束指向俯仰角（度），标量=单角度，数组=多角度。
                     平面阵时与 phi0s_deg broadcast 为 (Nscan,) 配对。
        phi_deg_start: 方位角起始（度），array_type="planar" 时必须指定
        phi_deg_end: 方位角终止（度）
        phi_deg_step: 方位角步长（度）
        phi0s_deg: 波束指向方位角（度），标量=单个，数组=多个，默认 0°。
                   与 theta0s_deg broadcast：短的自动填充，等长则一一配对。
        frequenciesGHz: 工作频率（GHz），标量=单频，数组=多频
    """

    def __init__(
        self,
        array_type: Literal["linear", "planar"] = "linear",
        symmetric: bool = False,
        theta_deg_start: float = -90,
        theta_deg_end: float = 90,
        theta_deg_step: float = 0.1,
        theta0s_deg: Union[float, np.ndarray] = 0.0,
        phi_deg_start: Optional[float] = None,
        phi_deg_end: Optional[float] = None,
        phi_deg_step: Optional[float] = None,
        phi0s_deg: Union[float, np.ndarray, None] = None,
        frequenciesGHz: Union[float, np.ndarray] = 1.0,
    ):
        # ── 阵列类型 ──
        if array_type not in ("linear", "planar"):
            raise ValueError(f"array_type 必须是 'linear' 或 'planar', 实际 '{array_type}'")
        self._array_type = array_type
        self._symmetric = symmetric

        # ── 频率 ──
        self.frequenciesGHz = np.atleast_1d(
            np.asarray(frequenciesGHz, dtype=float)
        )
        self._n_freq = len(self.frequenciesGHz)
        # k = 2π * (f / f₀), 单频时 f/f₀ = 1 → k = 2π
        self._ks = TWO_PI * self.frequenciesGHz / self.frequenciesGHz[0]

        # ── theta 网格 ──
        self.theta_deg = np.arange(
            theta_deg_start, theta_deg_end + theta_deg_step / 2, theta_deg_step
        )
        self._sin_theta = np.sin(np.deg2rad(self.theta_deg))

        # ── phi 网格 ──
        if array_type == "planar":
            if phi_deg_start is None or phi_deg_end is None or phi_deg_step is None:
                raise ValueError("平面阵必须指定 phi_deg_start, phi_deg_end, phi_deg_step")
            self.phi_deg = np.arange(
                phi_deg_start, phi_deg_end + phi_deg_step / 2, phi_deg_step
            )
            self._n_phi = len(self.phi_deg)
            self._sin_phi = np.sin(np.deg2rad(self.phi_deg))
            self._cos_phi = np.cos(np.deg2rad(self.phi_deg))
        else:
            self.phi_deg = None

        # ── 扫描角 ──
        self.theta0s_deg = np.atleast_1d(
            np.asarray(theta0s_deg, dtype=float)
        )
        self._sin_theta0s = np.sin(np.deg2rad(self.theta0s_deg))

        # 平面阵：phi0s 与 theta0s broadcast 为配对扫描方向 (Nscan,)
        if array_type == "planar":
            phi0s = np.atleast_1d(
                np.asarray(phi0s_deg if phi0s_deg is not None else 0.0, dtype=float)
            )
            n_theta0 = len(self.theta0s_deg)
            n_phi0 = len(phi0s)
            if n_theta0 > 1 and n_phi0 > 1 and n_theta0 != n_phi0:
                raise ValueError(
                    f"theta0s ({n_theta0}) 和 phi0s ({n_phi0}) 长度不一致，"
                    f"无法配对为扫描方向"
                )
            self._n_scan = max(n_theta0, n_phi0)

            # broadcast theta0s
            if n_theta0 == 1 and self._n_scan > 1:
                self.theta0s_deg = np.full(self._n_scan, self.theta0s_deg[0])
                self._sin_theta0s = np.full(self._n_scan, self._sin_theta0s[0])
            # broadcast phi0s
            if n_phi0 == 1 and self._n_scan > 1:
                phi0s = np.full(self._n_scan, phi0s[0])
            self._phi0s_deg = phi0s
            self._sin_phi0s = np.sin(np.deg2rad(phi0s))
            self._cos_phi0s = np.cos(np.deg2rad(phi0s))
        else:
            self._n_scan = len(self.theta0s_deg)
            self._phi0s_deg = None

        # 预计算 delta_sin = sinθ − sinθ₀
        _delta = self._sin_theta[None, :] - self._sin_theta0s[:, None]
        if self._n_scan == 1:
            self._delta_sin = _delta[0]          # (Nθ,)
        else:
            self._delta_sin = _delta              # (Nscan, Nθ)

        # ── 平面阵 UV 空间预计算 ──
        if array_type == "planar":
            # deltaU = sinθ·cosφ − sinθ₀·cosφ₀
            # deltaV = sinθ·sinφ − sinθ₀·sinφ₀
            sin_th = self._sin_theta[None, :, None]   # (1, Nθ, 1)
            cos_ph = self._cos_phi[None, None, :]     # (1, 1, Nφ)
            sin_ph = self._sin_phi[None, None, :]     # (1, 1, Nφ)
            sin0s = self._sin_theta0s[:, None, None]   # (Nscan, 1, 1)
            c0s = self._cos_phi0s[:, None, None]      # (Nscan, 1, 1)
            s0s = self._sin_phi0s[:, None, None]      # (Nscan, 1, 1)
            du = sin_th * cos_ph - sin0s * c0s         # (Nscan, Nθ, Nφ)
            dv = sin_th * sin_ph - sin0s * s0s
            if self._n_scan == 1:
                self._delta_u = du[0]
                self._delta_v = dv[0]
            else:
                self._delta_u = du
                self._delta_v = dv
        else:
            self._delta_u = None
            self._delta_v = None

    # ============================================================
    #  属性
    # ============================================================

    @property
    def array_type(self) -> str:
        return self._array_type

    @property
    def is_planar(self) -> bool:
        return self._array_type == "planar"

    @property
    def symmetric(self) -> bool:
        return self._symmetric

    @property
    def is_multi_freq(self) -> bool:
        return self._n_freq > 1

    @property
    def is_multi_scan(self) -> bool:
        return self._n_scan > 1

    # ============================================================
    #  统一 AF 入口
    # ============================================================

    def af(
        self,
        wl_x: np.ndarray,
        wl_y: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        is_default_excitation: bool = False,
    ) -> np.ndarray:
        """统一阵因子入口，根据 array_type 自动分派。

        Args:
            wl_x: 阵元 x 坐标（以第一频率波长 λ₀ 为单位），shape (N,)
            wl_y: 阵元 y 坐标（波长单位），仅 array_type="planar" 时需要
            amplitudes: 激励幅度，shape (N,)，默认全 1
            phases: 激励相位（弧度），shape (N,)，默认全 0
            is_default_excitation: True 时跳过幅相乘法，走纯位置快速路径。
                                   由 scenario.py 预检测后传入。

        Returns:
            复数阵因子
        """
        if is_default_excitation:
            amplitudes = None
            phases = None

        if self._array_type == "linear":
            if self._symmetric:
                return self.linear_af_symmetric(wl_x, amplitudes=amplitudes,
                                                phases=phases)
            return self.linear_af(wl_x, amplitudes, phases)
        # planar
        if wl_y is None:
            raise ValueError("平面阵必须提供 wl_y")
        if self._symmetric:
            return self.planar_af_symmetric(wl_x, wl_y,
                                            amplitudes, phases)
        return self.planar_af(wl_x, wl_y, amplitudes, phases)

    # ============================================================
    #  线阵
    # ============================================================

    def linear_af(
        self,
        wl_positions: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """线阵非对称阵因子。

        AF(θ) = Σ A_n · exp(j · k · x_n · (sinθ − sinθ₀))

        Args:
            wl_positions: 阵元 x 坐标（以第一频率波长 λ₀ 为单位），shape (N,)
            amplitudes: 激励幅度，shape (N,)，None=全 1
            phases: 激励相位（弧度），shape (N,)，None=全 0

        Returns:
            复数阵因子
        """
        positions = np.asarray(wl_positions, dtype=float)
        n = len(positions)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        if self._n_freq == 1:
            k = TWO_PI
            phase = k * np.outer(positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(n, *self._delta_sin.shape)
            e = np.exp(1j * phase)

            if not has_amps and not has_phs:
                return np.sum(e, axis=0)
            if not has_phs:
                b = amps.reshape(n, *((1,) * (e.ndim - 1)))
                return np.sum(b * e, axis=0)
            return np.einsum("i,i...->...", exc, e)

        # 多频
        afs = []
        for k in self._ks:
            phase = k * np.outer(positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(n, *self._delta_sin.shape)
            e = np.exp(1j * phase)

            if not has_amps and not has_phs:
                afs.append(np.sum(e, axis=0))
            elif not has_phs:
                b = amps.reshape(n, *((1,) * (e.ndim - 1)))
                afs.append(np.sum(b * e, axis=0))
            else:
                afs.append(np.einsum("i,i...->...", exc, e))
        return np.array(afs)

    def linear_af_symmetric(
        self,
        half_wl_positions: np.ndarray,
        has_center: bool = False,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        center_amplitude: float = 1.0,
        center_phase: float = 0.0,
    ) -> np.ndarray:
        """线阵对称阵因子（关于原点 x→−x 对称）。

        AF = Σ 2·A_n·cos(k·x_n·(sinθ−sinθ₀))  (+ center)
            center = center_amplitude · exp(j·center_phase)

        Args:
            half_wl_positions: 半边阵元 x 坐标（x > 0，以 λ₀ 为单位），shape (Nh,)
            has_center: 是否有中心阵元（x=0）
            amplitudes: 半边激励幅度，shape (Nh,)，None=全 1（不含中心）
            phases: 半边激励相位（弧度），shape (Nh,)，None=全 0（不含中心）
            center_amplitude: 中心阵元幅度，默认 1.0
            center_phase: 中心阵元相位（弧度），默认 0.0

        Returns:
            复数阵因子
        """
        half_positions = np.asarray(half_wl_positions, dtype=float)
        nh = len(half_positions)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        center_exc = center_amplitude * np.exp(1j * center_phase) if has_center else 0j

        if self._n_freq == 1:
            k = TWO_PI
            phase = k * np.outer(half_positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(nh, *self._delta_sin.shape)
            c = np.cos(phase)

            if not has_amps and not has_phs:
                af = 2 * np.sum(c, axis=0)
            elif not has_phs:
                b = amps.reshape(nh, *((1,) * (c.ndim - 1)))
                af = 2 * np.sum(b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, 2 * c)

            if has_center:
                af = af + center_exc
            return af

        # 多频
        afs = []
        for k in self._ks:
            phase = k * np.outer(half_positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(nh, *self._delta_sin.shape)
            c = np.cos(phase)

            if not has_amps and not has_phs:
                af = 2 * np.sum(c, axis=0)
            elif not has_phs:
                b = amps.reshape(nh, *((1,) * (c.ndim - 1)))
                af = 2 * np.sum(b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, 2 * c)

            if has_center:
                af = af + center_exc
            afs.append(af)
        return np.array(afs)

    # ============================================================
    #  平面阵
    # ============================================================

    def planar_af(
        self,
        wl_x: np.ndarray,
        wl_y: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """平面阵非对称阵因子（UV 空间）。

        AF = Σ A_n · exp(j·k·(x_n·deltaU + y_n·deltaV))

        Args:
            wl_x: 阵元 x 坐标（以第一频率波长 λ₀ 为单位），shape (N,)
            wl_y: 阵元 y 坐标（波长单位），shape (N,)
            amplitudes: 激励幅度，shape (N,)，None=全 1
            phases: 激励相位（弧度），shape (N,)，None=全 0
            n_jobs: 并行线程数, N>5000 时生效（默认 1=串行）

        Returns:
            复数阵因子
              单频单角度: (Nθ, Nφ)
              单频多角度: (Nscan, Nθ, Nφ)
        """
        if self._array_type != "planar":
            raise ValueError("array_type='linear' 不支持 planar_af")

        x = np.asarray(wl_x, dtype=float)
        y = np.asarray(wl_y, dtype=float)
        n = len(x)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        w = np.ones(n, dtype=complex)
        if has_phs:
            w = exc.astype(complex)
        elif has_amps:
            w = amps.astype(complex)

        if self._n_freq == 1:
            return _planar_af_chunked(x, y, w, self._delta_u, self._delta_v, TWO_PI, n_jobs)

        afs = []
        for k in self._ks:
            afs.append(_planar_af_chunked(x, y, w, self._delta_u, self._delta_v, k, n_jobs))
        return np.array(afs)

    def planar_af_symmetric(
        self,
        quarter_wl_x: np.ndarray,
        quarter_wl_y: np.ndarray,
        has_center: bool = False,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """平面阵四象限对称阵因子（关于 x→−x, y→−y 对称）。

        AF = Σ sym_i · A_i · cos(k·x_i·Δu) · cos(k·y_i·Δv)  (+ center)

        其中 sym_i = 4（x>0,y>0）、2（轴上单元素）、原点由 has_center 处理。

        Args:
            quarter_wl_x: 第一象限阵元 x 坐标（以 λ₀ 为单位），shape (Nq,)
            quarter_wl_y: 第一象限阵元 y 坐标（以 λ₀ 为单位），shape (Nq,)
            has_center: 是否有中心阵元（x=0,y=0，激励 1+0j）
            amplitudes: 四分之一象限激励幅度，shape (Nq,)，None=全 1
            phases: 四分之一象限激励相位（弧度），shape (Nq,)，None=全 0

        Returns:
            复数阵因子
        """
        if self._array_type != "planar":
            raise ValueError("array_type='linear' 不支持 planar_af_symmetric")

        qx = np.asarray(quarter_wl_x, dtype=float)
        qy = np.asarray(quarter_wl_y, dtype=float)
        nq = len(qx)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        # 对称因子：x>0且y>0 → 4, 仅在一个轴上 → 2
        eps = 1e-12
        x_on_axis = qx <= eps
        y_on_axis = qy <= eps
        sym_factor = np.where(
            x_on_axis & y_on_axis, 0.0,          # 原点由 has_center 处理
            np.where(x_on_axis | y_on_axis, 2.0, 4.0),
        )

        if self._n_freq == 1:
            k = TWO_PI
            phase_u = k * np.outer(qx, self._delta_u.ravel())
            phase_v = k * np.outer(qy, self._delta_v.ravel())
            phase_u = phase_u.reshape(nq, *self._delta_u.shape)
            phase_v = phase_v.reshape(nq, *self._delta_v.shape)
            c = np.cos(phase_u) * np.cos(phase_v)

            sf = sym_factor.reshape(nq, *((1,) * (c.ndim - 1)))

            if not has_amps and not has_phs:
                af = np.sum(sf * c, axis=0)
            elif not has_phs:
                b = amps.reshape(nq, *((1,) * (c.ndim - 1)))
                af = np.sum(sf * b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, sf * c)

            if has_center:
                af = af + 1.0
            return af

        # 多频
        afs = []
        for k in self._ks:
            phase_u = k * np.outer(qx, self._delta_u.ravel())
            phase_v = k * np.outer(qy, self._delta_v.ravel())
            phase_u = phase_u.reshape(nq, *self._delta_u.shape)
            phase_v = phase_v.reshape(nq, *self._delta_v.shape)
            c = np.cos(phase_u) * np.cos(phase_v)

            sf = sym_factor.reshape(nq, *((1,) * (c.ndim - 1)))

            if not has_amps and not has_phs:
                af = np.sum(sf * c, axis=0)
            elif not has_phs:
                b = amps.reshape(nq, *((1,) * (c.ndim - 1)))
                af = np.sum(sf * b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, sf * c)

            if has_center:
                af = af + 1.0
            afs.append(af)
        return np.array(afs)

    # ============================================================
    #  平面阵 FFT 快速计算
    # ============================================================

    def planar_af_fft(
        self,
        w: np.ndarray,
        dx_wl: float,
        dy_wl: float,
        oversample: int = 8,
        order: int = 1,
    ) -> np.ndarray:
        """FFT 快速计算均匀矩形栅格平面阵阵因子。

        复杂度 O(M'N'log M'N')，比直接求和快 6~50×。
        精度：默认参数下 RMS 误差 ~0.004 dB（vs 直接求和）。

        Args:
            w: 复激励矩阵, shape (Ny, Nx)
            dx_wl: x 方向阵元间距 (波长单位)
            dy_wl: y 方向阵元间距 (波长单位)
            oversample: UV 过采样倍数, ≥8 推荐
            order: 插值阶数, 1=双线性, 3=三次样条 (推荐)

        Returns:
            复数阵因子。
              单频单角度: (Nθ, Nφ)
              单频多角度: (Nscan, Nθ, Nφ)
              多频:        (Nfreq, ...)
        """
        if self._array_type != "planar":
            raise ValueError("planar_af_fft 仅支持平面阵 (array_type='planar')")

        from scipy.ndimage import map_coordinates

        Ny, Nx = w.shape
        atol = 1e-12

        # 1) 预计算: FFT 尺寸、UV 网格 (与频率/扫描角无关)
        Mp = 1 << int(np.ceil(np.log2(max(Nx * oversample, 256))))
        Np_fft = 1 << int(np.ceil(np.log2(max(Ny * oversample, 256))))
        uv_period_u = 1.0 / dx_wl
        uv_period_v = 1.0 / dy_wl
        u_half = 0.5 / dx_wl
        v_half = 0.5 / dy_wl
        u_fft = np.fft.fftshift(np.fft.fftfreq(Mp, d=dx_wl))
        v_fft = np.fft.fftshift(np.fft.fftfreq(Np_fft, d=dy_wl))

        # 2) 目标 UV (planar_af 返回 (Nθ,Nφ), 此处以 θ 为行、φ 为列)
        TH, PH = np.meshgrid(self.theta_deg, self.phi_deg, indexing='ij')  # (Nθ, Nφ)
        u_tgt = np.sin(np.deg2rad(TH)) * np.cos(np.deg2rad(PH))
        v_tgt = np.sin(np.deg2rad(TH)) * np.sin(np.deg2rad(PH))

        # 3) UV 周期映射 + 索引
        u_mapped = ((u_tgt + u_half) % uv_period_u) - u_half
        v_mapped = ((v_tgt + v_half) % uv_period_v) - v_half
        u_idx = np.clip((u_mapped - u_fft[0]) / max(u_fft[1] - u_fft[0], atol),
                         0, len(u_fft) - 1.001)
        v_idx = np.clip((v_mapped - v_fft[0]) / max(v_fft[1] - v_fft[0], atol),
                         0, len(v_fft) - 1.001)
        coords = np.stack([v_idx.ravel(), u_idx.ravel()])

        # 4) 相位修正系数 (在目标 UV 上, 形状 (Nθ, Nφ))
        Nt, Np_phi = len(self.theta_deg), len(self.phi_deg)
        phase_corr = np.exp(-1j * np.pi *
                            ((Nx - 1) * dx_wl * u_tgt +
                             (Ny - 1) * dy_wl * v_tgt))  # (Nθ, Nφ)

        def _fft_one(w_exc, k_factor=1.0):
            """对单频单扫描角计算 AF。k_factor 用于多频缩放。"""
            # 零填充 + IFFT
            W_pad = np.zeros((Np_fft, Mp), dtype=complex)
            W_pad[:Ny, :Nx] = w_exc
            DFT = np.fft.fftshift(np.fft.ifft2(W_pad)) * (Mp * Np_fft)

            # 插值
            DFT_interp = (map_coordinates(DFT.real, coords, order=order,
                                          mode='wrap', cval=0.0) +
                          1j * map_coordinates(DFT.imag, coords, order=order,
                                                mode='wrap', cval=0.0))
            DFT_interp = DFT_interp.reshape(Nt, Np_phi)  # (Nθ, Nφ)

            # 多频: UV 使用 k·u, k·v (k 已隐含在 fftfreq 的 d 参数中)
            # 相位修正中的 u/v 也需要相应缩放
            if abs(k_factor - 1.0) > atol:
                corr = np.exp(-1j * np.pi * k_factor *
                              ((Nx - 1) * dx_wl * u_tgt +
                               (Ny - 1) * dy_wl * v_tgt))
            else:
                corr = phase_corr
            return DFT_interp * corr

        # ── 单频 ──
        if self._n_freq == 1:
            # 多扫描角: 对每个扫描角相位偏移激励后做 FFT
            if self._n_scan > 1:
                afs = []
                for s in range(self._n_scan):
                    u0 = self._sin_theta0s[s] * self._cos_phi0s[s]
                    v0 = self._sin_theta0s[s] * self._sin_phi0s[s]
                    # 相位偏移: w_scanned = w * exp(-j2π(x_m*u0 + y_n*v0))
                    xm = dx_wl * (np.arange(Nx) - (Nx - 1) / 2)
                    yn = dy_wl * (np.arange(Ny) - (Ny - 1) / 2)
                    phase_scan = np.exp(-1j * 2 * np.pi *
                                        (yn[:, None] * v0 + xm[None, :] * u0))
                    w_scan = w * phase_scan
                    afs.append(_fft_one(w_scan))
                return np.array(afs)
            return _fft_one(w)

        # ── 多频 ──
        afs = []
        for fi, k in enumerate(self._ks):
            # 多频: 调整 UV 映射 — 频率为 f₀·k/2π 时 d 等效 = d_wl * k/2π
            # 简化: 用 k/2π 缩放的 UV 坐标
            kf = k / (2 * np.pi)  # f/f₀
            # 重新计算 UV 映射 (带 kf 缩放)
            u_fft_k = np.fft.fftshift(np.fft.fftfreq(Mp, d=dx_wl))
            v_fft_k = np.fft.fftshift(np.fft.fftfreq(Np_fft, d=dy_wl))
            # 等效间距变化: d_k = d / kf, UV 周期 = kf/d
            uv_period_u_k = kf / dx_wl
            uv_period_v_k = kf / dy_wl
            u_half_k = 0.5 * kf / dx_wl
            v_half_k = 0.5 * kf / dy_wl

            u_m_k = ((u_tgt + u_half_k) % uv_period_u_k) - u_half_k
            v_m_k = ((v_tgt + v_half_k) % uv_period_v_k) - v_half_k
            u_idx_k = np.clip((u_m_k - u_fft_k[0]) /
                              max(u_fft_k[1] - u_fft_k[0], atol),
                              0, len(u_fft_k) - 1.001)
            v_idx_k = np.clip((v_m_k - v_fft_k[0]) /
                              max(v_fft_k[1] - v_fft_k[0], atol),
                              0, len(v_fft_k) - 1.001)
            coords_k = np.stack([v_idx_k.ravel(), u_idx_k.ravel()])

            W_pad = np.zeros((Np_fft, Mp), dtype=complex)
            W_pad[:Ny, :Nx] = w
            DFT = np.fft.fftshift(np.fft.ifft2(W_pad)) * (Mp * Np_fft)
            DFT_interp = (map_coordinates(DFT.real, coords_k, order=order,
                                          mode='wrap', cval=0.0) +
                          1j * map_coordinates(DFT.imag, coords_k, order=order,
                                                mode='wrap', cval=0.0))
            DFT_interp = DFT_interp.reshape(Nt, Np_phi)  # (Nθ, Nφ)
            corr = np.exp(-1j * np.pi * kf *
                          ((Nx - 1) * dx_wl * u_tgt +
                           (Ny - 1) * dy_wl * v_tgt))
            afs.append(DFT_interp * corr)
        return np.array(afs)

    # ============================================================
    #  通用后处理
    # ============================================================

    @staticmethod
    def normalize(af: np.ndarray) -> np.ndarray:
        """线性归一化: |af| / max(|af|), 范围 [0, 1]。

        Args:
            af: 复数阵因子

        Returns:
            归一化幅度
        """
        return np.abs(af) / (np.max(np.abs(af)) + 1e-30)

    @staticmethod
    def to_dB(af: np.ndarray, normalized: bool = True) -> np.ndarray:
        """阵因子转 dB。

        Args:
            af: 复数阵因子
            normalized: True 时先归一化再转 dB（峰值 0 dB），默认开启

        Returns:
            dB 方向图
        """
        if normalized:
            return 20 * np.log10(np.abs(af) / np.max(np.abs(af)) + 1e-30)
        return 20 * np.log10(np.abs(af) + 1e-30)


# ═══════════════════════════════════════════════════
#  平面阵分块计算（防 OOM — 大阵列按 θ 维度切片）
# ═══════════════════════════════════════════════════

def _planar_af_chunked(
    x, y, w, delta_u, delta_v, k, n_jobs=1,
    _max_mb=500,
):
    """平面阵 AF 分块 — 目标单块 < 500MB，自动并行。"""
    ntheta = delta_u.shape[-2]
    nphi = delta_v.shape[-1]
    n = len(x)
    is_3d = (delta_u.ndim == 3)

    max_cells = int(_max_mb * 1e6 / (nphi * 16))
    tc = min(50, ntheta)
    ec = min(n, max(200, max_cells // tc))

    if n <= ec and ntheta <= tc:
        return _planar_af_vec(x, y, w, delta_u, delta_v, k)

    af_shape = delta_u.shape if is_3d else (ntheta, nphi)
    af = np.zeros(af_shape, dtype=complex)

    # 并行：每个线程处理一批阵元
    if n_jobs > 1 and n > 5000:
        from concurrent.futures import ThreadPoolExecutor
        nj = min(n_jobs, n // max(ec, 1) + 1)
        elem_batches = [(e0, min(e0 + ec, n)) for e0 in range(0, n, ec)]
        def _job(e0, e1):
            xe, ye, we = x[e0:e1], y[e0:e1], w[e0:e1]
            if ntheta <= tc:
                return _planar_af_vec(xe, ye, we, delta_u, delta_v, k)
            part = np.zeros(af_shape, dtype=complex)
            for t0 in range(0, ntheta, tc):
                t1 = min(t0 + tc, ntheta)
                dc_u = delta_u[..., t0:t1, :] if is_3d else delta_u[t0:t1]
                dc_v = delta_v[..., t0:t1, :] if is_3d else delta_v[t0:t1]
                if is_3d:
                    part[..., t0:t1, :] += _planar_af_vec(xe, ye, we, dc_u, dc_v, k)
                else:
                    part[t0:t1] += _planar_af_vec(xe, ye, we, dc_u, dc_v, k)
            return part
        with ThreadPoolExecutor(max_workers=nj) as pool:
            futures = [pool.submit(_job, e0, e1) for e0, e1 in elem_batches]
            for f in futures:
                af += f.result()
        return af

    # 串行
    for e0 in range(0, n, ec):
        e1 = min(e0 + ec, n)
        xe, ye, we = x[e0:e1], y[e0:e1], w[e0:e1]
        if ntheta <= tc:
            af += _planar_af_vec(xe, ye, we, delta_u, delta_v, k)
            continue
        for t0 in range(0, ntheta, tc):
            t1 = min(t0 + tc, ntheta)
            dc_u = delta_u[..., t0:t1, :] if is_3d else delta_u[t0:t1]
            dc_v = delta_v[..., t0:t1, :] if is_3d else delta_v[t0:t1]
            if is_3d:
                af[..., t0:t1, :] += _planar_af_vec(xe, ye, we, dc_u, dc_v, k)
            else:
                af[t0:t1] += _planar_af_vec(xe, ye, we, dc_u, dc_v, k)
    return af


def _planar_af_vec(x, y, w, du, dv, k):
    """单块向量化计算。du/dv shape: (Nθ_block, Nφ) 或 (Nscan, Nθ_block, Nφ)。"""
    phase = k * (np.outer(x, du.ravel()) + np.outer(y, dv.ravel()))
    phase = phase.reshape(len(x), *du.shape)
    return np.einsum("i,i...->...", w, np.exp(1j * phase))
