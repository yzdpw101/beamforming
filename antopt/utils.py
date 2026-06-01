"""工具函数。"""

import json
from pathlib import Path
import numpy as np


def compute_pattern(pos, amps, phases, mapper, pat, fe_patterns=None,
                    is_default_excitation=False):
    """计算方向图乘积: pattern = Fe * |AF| (实数量)。

    Args:
        pos: 阵元位置数组 (Ne,)
        amps: 激励幅度 (Ne,)
        phases: 激励相位 (rad, Ne,)
        mapper: LMMapper 实例
        pat: Pattern 实例
        fe_patterns: 单元方向图列表 [Fe_array, ...] 或 None
        is_default_excitation: 幅相均为默认值时跳过乘法，由 scenario 预检测

    Returns:
        np.ndarray: 实值方向图 (未 dB, 未归一化)
    """
    if pat.is_planar or not mapper.is_symmetric:
        af = pat.linear_af(pos, amps, phases)
    else:
        halfNe = mapper._halfNe
        offset = 1 if mapper.has_center else 0
        kwargs = dict(has_center=mapper.has_center,
                      amplitudes=amps[halfNe + offset:],
                      phases=phases[halfNe + offset:])
        if mapper.has_center:
            kwargs["center_amplitude"] = amps[halfNe]
            kwargs["center_phase"] = phases[halfNe]
        af = pat.linear_af_symmetric(pos[halfNe + offset:], **kwargs)
    af_abs = np.abs(af)
    if fe_patterns is not None:
        af_abs = af_abs * fe_patterns[0]
    return af_abs


def to_json_flat(obj, indent=0):
    """仿 C++ toJsonFlatArrays: 对象换行, 数组紧凑单行。"""
    sp = " " * (indent + 2)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            items.append(f'{sp}{json.dumps(k)}: {to_json_flat(v, indent + 2)}')
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    elif isinstance(obj, list):
        return json.dumps(obj)
    else:
        return json.dumps(obj)


def load_array_config(filename, key, base_dir=None):
    """从 JSON 文件加载 array_config 格式或纯数组。支持相对/绝对路径。

    Args:
        filename: JSON 文件路径
        key: JSON 键名 (如 "xCenters", "phasesDeg", "amplitudes")
        base_dir: 相对路径的基准目录, None 表示当前工作目录

    Returns:
        np.ndarray or None
    """
    if not filename:
        return None
    path = Path(filename)
    if not path.is_absolute():
        path = (Path(base_dir) if base_dir else Path.cwd()) / filename
    if not path.exists():
        raise FileNotFoundError(f"导入文件不存在: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and key in data:
        data = data[key]
    return np.array(data, dtype=float) if data is not None else None


def fmt_freq_angle(freq_ghz, theta0_deg, phi0_deg=None):
    """格式化频率+角度为文件/目录名。F11p4_T0 或 F11p4_T0P0 (面阵)。

    Args:
        freq_ghz: 频率 (GHz)
        theta0_deg: θ 指向角 (度)
        phi0_deg: φ 指向角 (度), None=仅 θ
    """
    def _f(x):
        s = f"{x:.1f}".rstrip('0').rstrip('.')
        return s.replace('.', 'p')
    name = f"F{_f(freq_ghz)}_T{_f(theta0_deg)}"
    if phi0_deg is not None:
        name += f"P{_f(phi0_deg)}"
    return name
