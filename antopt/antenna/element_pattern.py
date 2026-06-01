"""单元方向图模型。

提供各向同性、cosine-q、微带贴片等单元方向图，
以及从 HFSS 导出的单元方向图文件导入功能。
"""

from typing import Optional
import numpy as np


class ElementPattern:
    """单元方向图模型。"""
    @staticmethod
    def isotropic(theta: np.ndarray) -> np.ndarray:
        """各向同性单元方向图，在所有方向增益为 1。"""
        return np.ones_like(theta)

    @staticmethod
    def cosine_q(theta: np.ndarray, q: float = 1.0) -> np.ndarray:
        """cos^q(theta) 单元方向图。

        半功率波束宽度随 q 增大而变窄。
        q=1: HPBW ≈ 90°,  q=2: HPBW ≈ 65°
        """
        theta_rad = np.deg2rad(theta)
        pattern = np.abs(np.cos(theta_rad)) ** q
        return pattern

    @staticmethod
    def patch(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """微带贴片天线近似方向图。

        E 面 (phi=0°):  cos(θ)
        H 面 (phi=90°): 1
        """
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        e_plane = np.abs(np.cos(theta_rad))
        pattern = e_plane[:, None] * np.ones_like(phi_rad[None, :])
        return pattern

    @staticmethod
    def from_hfss(filepath: str, theta: np.ndarray) -> np.ndarray:
        """从 HFSS 导出的单元方向图文件导入。

        Args:
            filepath: CSV 文件路径，格式应为 [theta_deg, gain_dB]
            theta: 需要插值到的 θ 角度网格

        Returns:
            线性插值后的单元方向图幅值 (线性刻度)
        """
        import csv
        thetas = []
        gains = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        t = float(row[0])
                        g = float(row[1])
                        thetas.append(t)
                        gains.append(g)
                    except ValueError:
                        continue

        if len(thetas) < 2:
            raise ValueError(f"HFSS 文件 {filepath} 数据不足")

        thetas = np.array(thetas)
        gains = np.array(gains)

        gain_interp = np.interp(theta, thetas, gains)
        pattern_linear = 10 ** (gain_interp / 20.0)
        return pattern_linear

    @staticmethod
    def from_csv(filepath: str, theta: np.ndarray) -> np.ndarray:
        """从 CSV 文件导入单元方向图（别名）。"""
        return ElementPattern.from_hfss(filepath, theta)

    @staticmethod
    def from_hfss_aep(
        aep_csv_directory: str,
        frequenciesGHz: np.ndarray,
        theta: np.ndarray,
        num_elements: int,
        phiIdx: int = 1,
        is_gain: bool = True,
        in_dB: bool = False,
        input_theta_range: tuple[float, float] = (-90.0, 90.0),
    ) -> list[list[np.ndarray]]:
        """AEP (Active Element Pattern) 多元多频方向图导入。

        自动从 CSV 行数计算采样间隔, 无需手动指定 oriDegStep。

        文件命名: aep_elem_{i}_{freq}GHz.csv (i=1..N)
        CSV 格式: 有表头, 第1列θ, 第2列起数据 (phiIdx 控制列索引)

        Returns:
            list[list[np.ndarray]]: result[freq_idx][elem_idx] → Fe(θ)
        """
        import csv as _csv, os

        target_step = abs(float(theta[1] - theta[0]))
        in_t0, in_t1 = input_theta_range

        result = []
        for fGHz in frequenciesGHz:
            freq_str = f"{fGHz:g}"
            elem_patterns_this_freq = []

            for elem_idx in range(1, num_elements + 1):
                exact_name = f"aep_elem_{elem_idx}_{freq_str}GHz.csv"
                exact_path = os.path.join(aep_csv_directory, exact_name)

                if os.path.exists(exact_path):
                    csv_path = exact_path
                else:
                    candidates = [
                        f for f in os.listdir(aep_csv_directory)
                        if f"elem_{elem_idx}" in f and f"{freq_str}GHz" in f
                        and f.endswith(".csv")
                    ]
                    if not candidates:
                        raise FileNotFoundError(
                            f"AEP CSV 缺失: 单元 {elem_idx}, 频率 {freq_str}GHz, "
                            f"期望: {exact_name}  目录: {aep_csv_directory}")
                    csv_path = os.path.join(aep_csv_directory, candidates[0])

                # 先读取全部数据, 自动计算步长
                all_vals = []
                with open(csv_path, "r") as f:
                    reader = _csv.reader(f)
                    for row in reader:
                        try:
                            val = float(row[phiIdx]) if len(row) > phiIdx else float(row[0])
                            all_vals.append(val)
                        except (ValueError, IndexError):
                            continue

                if not all_vals:
                    raise RuntimeError(f"无法从 {csv_path} 读取数据")

                n_csv = len(all_vals)
                csv_theta_range = abs(in_t1 - in_t0)
                oriDegStep = csv_theta_range / (n_csv - 1) if n_csv > 1 else target_step
                step_ratio = max(1, int(round(target_step / oriDegStep)))

                raw = np.array(all_vals[::step_ratio], dtype=float)

                # dB → 线性场方向图 Fe
                if in_dB:
                    if is_gain:
                        raw = 10.0 ** (raw / 20.0)
                    else:
                        raw = 10.0 ** (raw / 10.0)
                else:
                    if is_gain:
                        raw = np.sqrt(np.maximum(raw, 0.0))

                # 插值到目标 theta 网格
                csv_theta = np.linspace(in_t0, in_t1, len(raw))
                if len(raw) != len(theta):
                    raw = np.interp(theta, csv_theta, raw)

                elem_patterns_this_freq.append(raw)

            result.append(elem_patterns_this_freq)

        return result

    @staticmethod
    def from_hfss_multi_freq(
        eGainCsvDirectory: str,
        frequenciesGHz: np.ndarray,
        theta: tuple[float, float, float],
        input_theta_range: tuple[float, float],
        phiIdx: int = 1,
        is_gain: bool = True,
        in_dB: bool = False,
    ) -> list[np.ndarray]:
        """多频单元方向图导入（仅降采样，不插值）。

        theta = (start_deg, end_deg, step_deg)，输出长度 = round((end-start)/step) + 1。
        若 step 不能整除范围，自动圆整步长并警告。
        input_theta_range: (min_deg, max_deg) — CSV 文件中 theta 的覆盖范围，必填。
        """
        import csv, os, warnings

        t_start, t_end, t_step = float(theta[0]), float(theta[1]), float(theta[2])

        # ── 错误检查 ──
        if len(frequenciesGHz) == 0:
            raise ValueError("frequenciesGHz 不能为空")
        if t_start >= t_end or t_start < -180 or t_end > 180:
            raise ValueError(f"theta 范围 [{t_start},{t_end}] 无效, 必须在 [-180,180] 内")
        if t_step <= 0:
            raise ValueError(f"theta step {t_step} 必须 > 0")
        if phiIdx < 0:
            raise ValueError(f"phiIdx={phiIdx} 必须 >= 0")

        # ── step 圆整 ──
        span = t_end - t_start
        n_ideal = span / t_step
        n_int = round(n_ideal)
        if abs(n_int - n_ideal) > 1e-9:
            new_step = span / n_int
            warnings.warn(
                f"theta step {t_step} 不能整除范围 [{t_start},{t_end}]，"
                f"自动调整为 {new_step:.6f}（{n_int} 个点）")
            t_step = new_step

        # ── 目标 theta ──
        target_theta = np.linspace(t_start, t_end, n_int + 1)

        in_t0, in_t1 = input_theta_range
        csv_span_deg = abs(in_t1 - in_t0)

        result = []
        for fGHz in frequenciesGHz:
            freq_str = f"{fGHz:g}"
            candidates = [
                f for f in os.listdir(eGainCsvDirectory)
                if f.startswith(f"{freq_str}GHz") and f.endswith(".csv")
            ]
            if candidates:
                csv_path = os.path.join(eGainCsvDirectory, candidates[0])
            else:
                csv_path = os.path.join(eGainCsvDirectory, f"eGain_{freq_str}GHz.csv")
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(
                        f"单元方向图文件找不到: {freq_str}GHz*.csv 或 eGain_{freq_str}GHz.csv")

            # 读取全部数据
            all_vals = []
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        val = float(row[phiIdx]) if len(row) > phiIdx else float(row[0])
                        all_vals.append(val)
                    except (ValueError, IndexError):
                        continue
            if not all_vals:
                raise RuntimeError(f"无法从 {csv_path} 读取数据")

            n_csv = len(all_vals)
            ori_deg_step = csv_span_deg / (n_csv - 1) if n_csv > 1 else t_step

            # 降采样: step_ratio 向上取整
            step_ratio = max(1, int(round(t_step / ori_deg_step)))
            raw = np.array(all_vals[::step_ratio], dtype=float)
            if len(raw) != len(target_theta):
                if len(raw) > len(target_theta):
                    raw = raw[:len(target_theta)]
                else:
                    raw = np.pad(raw, (0, len(target_theta)-len(raw)), 'edge')

            # 转换为场方向图 Fe
            if in_dB:
                if is_gain:
                    raw = 10.0 ** (raw / 20.0)
                else:
                    raw = 10.0 ** (raw / 10.0)
            else:
                if is_gain:
                    raw = np.sqrt(np.maximum(raw, 0.0))

            result.append(raw)

        return result